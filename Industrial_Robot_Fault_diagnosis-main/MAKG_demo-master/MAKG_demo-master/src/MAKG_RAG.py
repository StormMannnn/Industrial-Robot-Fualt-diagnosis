import os
import csv
import time
import faiss
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from dotenv import load_dotenv
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from typing import List
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI


graph_path = 'Your path'
chunks_path = 'Your path'


graph_df = pd.read_csv(graph_path)
chunks_df = pd.read_csv(chunks_path)

graph_df['chunk_id'] = graph_df['chunk_id'].astype(str)
chunks_df['chunk_id'] = chunks_df['chunk_id'].astype(str)

print("✅ graph_df columns:", graph_df.columns.tolist())
print("✅ chunks_df columns:", chunks_df.columns.tolist())
print("✅ sample chunks_df rows:")
print(chunks_df.head())


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = 'bge-m3'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_tensor=False, normalize_embeddings=True).tolist()[0]


embedding_model = SentenceTransformerEmbeddings(model_name='Your model path')

knowledge_graph = nx.DiGraph()

for _, row in graph_df.iterrows():
    h = str(row["node_1"]).strip()
    r = str(row["edge"]).strip()
    t = str(row["node_2"]).strip()

    knowledge_graph.add_edge(
        h,
        t,
        relation=r,
        chunk_id=str(row["chunk_id"])
    )

def extract_subgraph(key_nodes, max_hops=2):

    if not key_nodes:
        return nx.DiGraph()

    visited = set(key_nodes)
    frontier = set(key_nodes)

    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            for nbr in knowledge_graph.successors(node):
                if nbr not in visited:
                    next_frontier.add(nbr)
            for nbr in knowledge_graph.predecessors(node):
                if nbr not in visited:
                    next_frontier.add(nbr)

        visited.update(next_frontier)
        frontier = next_frontier

    return knowledge_graph.subgraph(visited).copy()


index_file = 'vector_index.faiss'
chunk_ids_file = 'chunk_ids.csv'


def load_or_create_index():
    if os.path.exists(index_file) and os.path.exists(chunk_ids_file):
        index = faiss.read_index(index_file)
        with open(chunk_ids_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            chunk_ids = [row[0] for row in reader]
        triplets = [
            f"({row['node_1']} , {row['edge']} , {row['node_2']})"
            for _, row in graph_df.iterrows()
        ]
    else:
        embeddings, triplets, chunk_ids = [], [], []
        for _, row in graph_df.iterrows():
            triplet_text = f"({row['node_1']} , {row['edge']} , {row['node_2']})"
            triplet_embedding = embedding_model.embed_query(triplet_text)
            embeddings.append((triplet_embedding, row['chunk_id']))
            triplets.append(triplet_text)
        dimension = len(embeddings[0][0])
        index = faiss.IndexFlatIP(dimension)
        for embedding, chunk_id in embeddings:
            embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
            index.add(embedding_array)
            chunk_ids.append(str(chunk_id))
        faiss.write_index(index, index_file)
        with open(chunk_ids_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['chunk_id'])
            writer.writerows([[cid] for cid in chunk_ids])
    return index, chunk_ids, triplets


index, chunk_ids, triplets = load_or_create_index()



def query_database(querys, index=index, chunk_ids=chunk_ids, triplets=triplets, top_k=5, use_multihop=True):

    unique_triplets, chunk_scores = set(), []


    for query in querys:
        query_embedding = np.array(embedding_model.embed_query(query), dtype=np.float32).reshape(1, -1)
        D, I = index.search(query_embedding, k=top_k)
        for sim, idx in zip(D[0], I[0]):
            if idx < len(chunk_ids):
                chunk_scores.append((chunk_ids[idx], sim, triplets[idx]))
                unique_triplets.add(triplets[idx])

    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    selected_chunk_ids = [cid for cid, _, _ in chunk_scores[:top_k]]
    selected_triplets = [t for _, _, t in chunk_scores[:top_k]]
    print("\n🔹【Initial Retrieved Triplets】")
    for t in unique_triplets:
        print(f"  {t}")
    if use_multihop:
        key_nodes = set()
        for trip in selected_triplets:
            try:
                n1, r, n2 = trip.strip("()").split(",")
                key_nodes.add(n1.strip())
                key_nodes.add(n2.strip())
            except:
                continue

        subgraph = extract_subgraph(key_nodes, max_hops=2)

        one_hop_triplets = set()
        two_hop_triplets = set()

        for u, v, data in subgraph.edges(data=True):
            r = data.get("relation", "relatedTo")
            triplet_str = f"({u} , {r} , {v})"
            unique_triplets.add(triplet_str)

            if u in key_nodes or v in key_nodes:
                one_hop_triplets.add(triplet_str)
            else:
                two_hop_triplets.add(triplet_str)

        print("\n🔹【1-Hop Expanded Triplets】")
        for t in one_hop_triplets:
            print(f"  {t}")

        print("\n🔹【2-Hop Expanded Triplets】")
        for t in two_hop_triplets:
            print(f"  {t}")

    expanded_chunk_ids = set(selected_chunk_ids)

    related_chunk_ids = graph_df[graph_df.apply(
        lambda row: f"({row['node_1']} , {row['edge']} , {row['node_2']})" in unique_triplets, axis=1
    )]['chunk_id'].tolist()

    expanded_chunk_ids.update(related_chunk_ids)

    selected_chunks = chunks_df[chunks_df['chunk_id'].isin(expanded_chunk_ids)]['text'].tolist()

    print("\n--- Top matched chunks (with multihop) ---")
    for i, cid in enumerate(expanded_chunk_ids):
        text = chunks_df[chunks_df['chunk_id'] == cid]['text'].values
        if len(text) > 0:
            print(f"{i + 1}. [{cid}] {text[0][:200]}...")

    return list(unique_triplets), selected_chunks


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url=BASE_URL,
    api_key=OPENAI_API_KEY,
    temperature=0,
    max_tokens=4096,
)


def rewrite_and_split_query(query):
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        You are an intelligent assistant. I will give you a query, and you need to break the question down into short phrases 
        that can be used for querying in a knowledge graph vector database, in order to improve the accuracy of vector database matching.  
        You should only output the decomposed phrases, do not add or modify anything. Answer in English.

        Here are some examples:

        Example 1:
            Question:
                How does the direct-acting sequence valve achieve sequential operation of Cylinder I and Cylinder II through controlling hydraulic pressure?
            Phrases:
                Direct-acting sequence valve\n
                Hydraulic pressure\n
                Sequential operation of Cylinder I and Cylinder II\n

        Example 2:
            Question:
                In a hydraulic system, how can fault diagnosis be used to identify and solve the problem of 'system pressure fluctuation'?
            Phrases:
                Hydraulic system\n
                Fault diagnosis\n
                System pressure fluctuation\n

        Now, please decompose the following question into phrases.  
        You must strictly follow the phrase format shown above.  

        Question: {query}  
        Phrases:
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(query)
    sub_queries = [line.strip() for line in result.split('\n') if line.strip()]
    sub_queries.append(query)
    formatted = [{"question": sq, "weight": 1.0 / len(sub_queries)} for sq in sub_queries]
    return formatted


def optimize_contexts(query, chunks, top_k=5):

    if not chunks:
        return ""

    selected_chunks = chunks[:top_k]
    combined_text = "\n".join(selected_chunks)

    prompt = PromptTemplate(
        input_variables=["query", "contexts"],
        template="""
        You are an intelligent assistant. I will give you a query and some related contexts.  
        Your task is to **minimally edit** the contexts to remove only clearly irrelevant or redundant parts—**do not summarize, paraphrase, or condense**.

        query: {query}  
        contexts: {contexts}  

        Instructions:
        1. First, analyze the query to identify its core intent and required information.
        2. Then, go through the contexts sentence by sentence.
        3. **Keep every sentence that contains any information potentially useful for answering the query.**
        4. Only remove a sentence if it is completely unrelated to the query (e.g., about a different topic, generic filler, or repeated content).
        5. **Do not shorten sentences. Do not rephrase. Preserve original wording exactly.**
        6. The final output should be nearly identical to the input contexts—**remove no more than 50–60 English words in total**.
        7. Include all relevant triplets mentioned in the contexts, along with a brief explanation of their relevance.

        Output format (strictly follow):
        Triplets:  
        (node1, relation, node2) — Explanation of why this triplet helps answer the query.  
        (Repeat for each relevant triplet)

        Context:  
        The edited context text, with minimal removal (≤20 words). Preserve original phrasing and structure.

        Important:  
        - Output only in English.  
        - Do not add new information.  
        - Do not explain your editing process—just output the result.

        Begin extraction:

        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    optimized_contexts = chain.run({"query": query, "contexts": combined_text})

    return optimized_contexts


chunk_text_index_file = 'chunk_text_index.faiss'
chunk_text_ids_file = 'chunk_text_ids.csv'


def build_or_load_chunk_text_index():
    if os.path.exists(chunk_text_index_file) and os.path.exists(chunk_text_ids_file):
        index = faiss.read_index(chunk_text_index_file)
        with open(chunk_text_ids_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            chunk_ids = [row[0] for row in reader]
        return index, chunk_ids
    else:
        texts = chunks_df['text'].tolist()
        chunk_ids = chunks_df['chunk_id'].astype(str).tolist()
        embeddings = embedding_model.embed_documents(texts)

        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        embedding_array = np.array(embeddings, dtype=np.float32)
        index.add(embedding_array)

        faiss.write_index(index, chunk_text_index_file)
        with open(chunk_text_ids_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['chunk_id'])
            writer.writerows([[cid] for cid in chunk_ids])
        return index, chunk_ids


chunk_text_index, chunk_id_list_by_text = build_or_load_chunk_text_index()


def retrieve_from_chunk_text(query: str, top_k: int = 5):

    query_emb = np.array(embedding_model.embed_query(query), dtype=np.float32).reshape(1, -1)
    D, I = chunk_text_index.search(query_emb, k=top_k)
    results = []
    for sim, idx in zip(D[0], I[0]):
        if idx < len(chunk_id_list_by_text):
            cid = chunk_id_list_by_text[idx]
            text = chunks_df[chunks_df['chunk_id'] == cid]['text'].values
            if len(text) > 0:
                results.append((cid, float(sim), text[0]))
    return results  # list of (chunk_id, score, text)


df = pd.read_csv("Your path", encoding='')
queries = df['user_input']

PROMPT_TEMPLATE = """
Based on the following provided Triplets and context information to answer the question:
information: {context}
Answer the question according to the above context:
query: {question}.
First analyze the question, clearly identify the information needed to answer it, and then find the answer in the provided information. Think step by step.
Do not divide the answer into paragraphs or bullet points!!!
Provide a complete answer that addresses all parts of the query if there are multiple questions!
The answer must be highly relevant to the query! Ensure it is the most direct, accurate, and clear answer possible.
Do not provide any reasoning for your answer!!!
Do not include any information not mentioned in the context; do not fabricate anything.
Provide only the answer most directly related to solving the problem, without any reasoning process!!!
Do not say "according to the context" or "as mentioned in the context" or similar phrases.
You should only generate the answer itself, with no other irrelevant statements!!!
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

all_contexts, responses = [], []
total_time = 0

for q in queries:
    start = time.time()

    sub_q = rewrite_and_split_query(q)
    triplets, chunks_from_triplet = query_database([sq["question"] for sq in sub_q], top_k=5)

    text_retrieval_results = retrieve_from_chunk_text(q, top_k=5)
    chunks_from_text = [text for _, _, text in text_retrieval_results]
    chunk_ids_from_text = [cid for cid, _, _ in text_retrieval_results]
    print("\n🔍 [DEBUG] Triplet-based Retrieval Results (with multihop):")
    for i, t in enumerate(triplets[:20]):
        print(f"  T{i + 1}: {t}")

    print("\n🔍 [DEBUG] Chunk Text Retrieval Results (top 5):")
    for i, (cid, score, text) in enumerate(text_retrieval_results):
        print(f"  C{i + 1} [score={score:.4f}, id={cid}]: {text[:300]}...")

    merged_chunks_dict = {}  # chunk_id -> (text, score)

    for text in chunks_from_triplet:
        match_row = chunks_df[chunks_df['text'] == text]
        if not match_row.empty:
            cid = match_row['chunk_id'].iloc[0]
            merged_chunks_dict[cid] = (text, 0.9)

    for cid, score, text in zip(chunk_ids_from_text, [r[1] for r in text_retrieval_results], chunks_from_text):
        if cid not in merged_chunks_dict:
            merged_chunks_dict[cid] = (text, score)
        else:
            if score > merged_chunks_dict[cid][1]:
                merged_chunks_dict[cid] = (text, score)

    sorted_merged = sorted(merged_chunks_dict.items(), key=lambda x: x[1][1], reverse=True)
    final_chunks = [text for _, (text, _) in sorted_merged[:5]]

    for i, text in enumerate(final_chunks, start=1):
        print(f"  Chunk {i}: {text[:300]}{'...' if len(text) > 300 else ''}")
    retrival_context = optimize_contexts(q, final_chunks, top_k=5)

    print("\n=== Retrieval Context (Merged) ===")
    print(retrival_context[:1000], "...")

    prompt = prompt_template.format(context=retrival_context, question=q)
    response_text = llm.predict(prompt)

    print("\n--- Answer ---")
    print(response_text)

    all_contexts.append(retrival_context)
    responses.append(response_text)
    total_time += time.time() - start
    print(f"Time taken: {time.time() - start:.4f}s")

df['answer'] = responses
df["retrival_contexts"] = all_contexts
df.to_csv("Your path", index=False)

