

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ==================================================
# Paths
# ==================================================
CHUNK_CSV = "Your path"
QA_CSV = "Your path"
OUTPUT_CSV = "Your path"

# ==================================================
# DeepSeek API
# ==================================================
client = OpenAI(
    api_key="Your Key",  # 安全起见使用环境变量
    base_url="Your URL"
)
LLM_MODEL = "deepseek-chat"

# ==================================================
# Embedding Model
# ==================================================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 384

def embed_texts(texts):
    return embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


chunk_ids = []
chunk_texts = []

with open(CHUNK_CSV, encoding="utf-8") as f:
    reader = pd.read_csv(CHUNK_CSV)
    for _, row in reader.iterrows():
        chunk_ids.append(row["chunk_id"])
        chunk_texts.append(row["text"])

chunk_embeddings = embed_texts(chunk_texts)

# ==================================================
# Simple similarity search
# ==================================================
def retrieve_chunks(question, top_k=5):
    q_emb = embed_texts([question])[0]
    sims = chunk_embeddings @ q_emb
    top_idx = np.argsort(sims)[::-1][:top_k]

    retrieved_texts = [str(chunk_texts[i]) for i in top_idx]
    retrieved_ids = [str(chunk_ids[i]) for i in top_idx]
    return retrieved_texts, retrieved_ids


# ==================================================
# LLM QA
# ==================================================
def llm_predict(question, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""
Answer the question strictly based on the given context.
If the context is insufficient, say "Not enough information".

Context:
{context_text}

Question:
{question}
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Answer based on provided context only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# ==================================================
# Main processing loop
# ==================================================
df = pd.read_csv(QA_CSV,encoding="")

answers = []
retrieval_contexts_list = []
retrieval_ids_list = []

for idx, question_text in tqdm(df['user_input'].items(), desc="Processing Questions", total=len(df), ncols=100):
    try:
        retrieved_texts, retrieved_ids = retrieve_chunks(question_text, top_k=5)
        answer_text = llm_predict(question_text, retrieved_texts)
        answers.append(answer_text.replace("\n", " ").strip())
        retrieval_contexts_list.append("\n\n".join(retrieved_texts))
        retrieval_ids_list.append(",".join(retrieved_ids))
    except Exception as e:
        print(f"\nError at index {idx}: {e}")
        answers.append("[ERROR]")
        retrieval_contexts_list.append("")
        retrieval_ids_list.append("")

df['answer'] = answers
df['retrival_contexts'] = retrieval_contexts_list
df['retrieval_chunk_ids'] = retrieval_ids_list

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Done! Results saved to {OUTPUT_CSV}")
