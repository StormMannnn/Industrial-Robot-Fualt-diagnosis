# ==================================================
# HydeRAG: Hypothetical Document Embedding + RAG
# - LLM: deepseek-chat (via DeepSeek API)
# - Embedder: all-MiniLM-L6-v2
# - Vector DB: semantic_chunks.csv (chunk-based, Local Mode)
# ==================================================

import os
import csv
import logging
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ==============================
# Configuration
# ==============================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CHUNK_CSV = r"E:\agent\MAKG_demo-master\MAKG_demo-master\src\data\semantic_chunks.csv"
QA_CSV = "qa_pairs.csv"
OUTPUT_CSV = "hyde_rag_output3.csv"

# LLM via DeepSeek
DEEPSEEK_API_KEY = "sk-bfecce8c8b65475f9022731ddf969f8b"
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)
LLM_MODEL = "deepseek-chat"

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts):
    """Embed texts with normalization for cosine similarity."""
    return embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )


# ==============================
# Step 1: Load and embed chunks (build vector DB in memory)
# ==============================
logging.info("Loading semantic chunks and building vector database...")
chunk_ids = []
chunk_texts = []

with open(CHUNK_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        chunk_ids.append(row["chunk_id"])
        chunk_texts.append(row["text"])

chunk_embeddings = embed_texts(chunk_texts)
logging.info(f"Loaded and embedded {len(chunk_texts)} chunks.")


# ==============================
# Step 2: HyDE + Retrieval + Generation functions
# ==============================

def generate_hypothetical_answer(question: str) -> str:
    prompt = (
        "Please write a concise, plausible paragraph that directly answers the following question. "
        "Do not say 'I don't know' or 'not enough information'. Just provide a hypothetical but realistic answer.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256
    )
    return response.choices[0].message.content.strip()


def retrieve_top_k_chunks(query_text: str, top_k: int = 5):
    q_emb = embed_texts([query_text])[0]
    sims = chunk_embeddings @ q_emb
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "chunk_id": chunk_ids[idx],
            "text": chunk_texts[idx],
            "score": float(sims[idx])
        })
    return results


def generate_final_answer(question: str, contexts: list) -> str:
    context_text = "\n\n".join([ctx["text"] for ctx in contexts])
    prompt = f"""
Answer the question strictly based on the provided context.
If the context does not contain enough information, respond exactly: "Not enough information".

Context:
{context_text}

Question:
{question}
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You answer based only on the given context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()


# ==============================
# Step 3: Main QA Loop — Modified to preserve original QA columns
# ==============================
def main():
    # Step A: Load original QA data as list of dicts (preserves all columns)
    original_rows = []
    with open(QA_CSV, encoding="gbk") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames  # e.g., ['user_input', 'ground_truth', ...]
        for row in reader:
            original_rows.append(row)

    # Step B: Process each row
    output_rows = []
    for row in tqdm(original_rows, desc="Processing QA with HydeRAG"):
        question = row.get("user_input", "").strip()
        if not question:
            # If no question, append empty result
            new_row = row.copy()
            new_row.update({
                "answer": "",
                "retrieval_contexts": "",
                "retrieval_chunk_ids": ""
            })
            output_rows.append(new_row)
            continue

        try:
            # HyDE pipeline
            hypo_doc = generate_hypothetical_answer(question)
            retrieved = retrieve_top_k_chunks(hypo_doc, top_k=5)
            final_answer = generate_final_answer(question, retrieved)

            # Format retrieval results
            retrieval_contexts = "\n\n".join([r["text"] for r in retrieved])
            retrieval_chunk_ids = ",".join([r["chunk_id"] for r in retrieved])

            # Merge with original row
            new_row = row.copy()
            new_row.update({
                "answer": final_answer,
                "retrieval_contexts": retrieval_contexts,
                "retrieval_chunk_ids": retrieval_chunk_ids
            })
            output_rows.append(new_row)

        except Exception as e:
            logging.error(f"Error processing question: {question} | Error: {e}")
            new_row = row.copy()
            new_row.update({
                "answer": "[ERROR]",
                "retrieval_contexts": "",
                "retrieval_chunk_ids": ""
            })
            output_rows.append(new_row)

    # Step C: Write back with new columns
    # Determine final fieldnames (original + new columns if not present)
    final_fieldnames = list(fieldnames)  # start with original
    for col in ["answer", "retrieval_contexts", "retrieval_chunk_ids"]:
        if col not in final_fieldnames:
            final_fieldnames.append(col)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n✅ HydeRAG inference completed. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()