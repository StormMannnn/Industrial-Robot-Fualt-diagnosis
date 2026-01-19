import re

import pandas as pd
import ast
from pathlib import Path
import os
from dotenv import load_dotenv

from datasets import Dataset

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


def load_hf_dataset(csv_path: str) -> Dataset:
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']

    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"Successfully read file using {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"{encoding} encoding read failed, try the next encoding ..")
            continue
    else:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', errors='replace')
        except Exception as e:
            raise RuntimeError(f"Unable to read file {csv_path}: {e}")

    records = []

    for idx, row in df.iterrows():
        raw_reference_contexts = str(row.get("reference_contexts", "") or "")
        raw_retrival_contexts = str(row.get("retrival_contexts", "") or "")

        reference_contexts = []
        if raw_reference_contexts.strip():
            try:
                parsed_ref = ast.literal_eval(raw_reference_contexts)
                if isinstance(parsed_ref, list):
                    reference_contexts = [str(x).strip() for x in parsed_ref if str(x).strip()]
                else:
                    reference_contexts = [str(parsed_ref).strip()]
            except Exception:
                reference_contexts = [raw_reference_contexts.strip()]

        retrival_contexts = []
        if raw_retrival_contexts.strip():
            try:
                parsed = ast.literal_eval(raw_retrival_contexts)
                if isinstance(parsed, list):
                    retrival_contexts = [str(x).strip() for x in parsed if str(x).strip()]
                else:
                    retrival_contexts = [str(parsed).strip()]
            except Exception:
                retrival_contexts = [raw_retrival_contexts.strip()]

        answer = str(row.get("answer", "") or "").strip()
        reference = str(row.get("reference", "") or "").strip()
        question = str(row.get("user_input", "") or "").strip()

        record = {
            "question": question,
            "contexts": retrival_contexts,
            "answer": answer,
            "ground_truth": reference,
            "ground_truths": [reference] if reference else [],
            "reference_contexts": reference_contexts,
        }

        records.append(record)

    dataset = Dataset.from_list(records)
    return dataset


def build_llm_and_embeddings():
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL", "Your url")
    if not api_key:
        raise RuntimeError("error")
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_BASE_URL", base_url)

    lc_llm = ChatOpenAI(
        model="deepseek-chat",
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        timeout=120,
        max_retries=3
    )
    lc_emb = HuggingFaceEmbeddings(model_name="Your model path",
                                   model_kwargs={"device": "cuda"},
                                   encode_kwargs={"batch_size": 32})
    return LangchainLLMWrapper(lc_llm), LangchainEmbeddingsWrapper(lc_emb)


def evaluate_dataset(dataset, dataset_name, llm, embeddings, save_path_prefix="ragas"):

    print(f"\n Start evaluation {dataset_name}...")

    try:
        results = evaluate(dataset, llm=llm, embeddings=embeddings)
    except Exception as e:
        print(f"evaluation failed: {e}")
        return None, {}

    try:
        df = results.to_pandas()
    except Exception as e:
        print(f"DataFrame error: {e}")
        return results, {}

    print(f"📊 {dataset_name} Scores (average over dataset):")
    scores = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            avg = df[col].mean()
            scores[col] = avg
            print(f"  {col}: {avg:.4f}")

    csv_file = f"{save_path_prefix}_{dataset_name.replace(' ', '_')}.csv"
    try:
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        print(f"Save CSV detailed results: {csv_file}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    return df, scores


def main():
    csv_path = "Your path"
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Not found {csv_path}")

    # 加载数据
    dataset = load_hf_dataset(csv_path)
    llm, emb = build_llm_and_embeddings()

    # 评估
    results, scores = evaluate_dataset(dataset, "Your path", llm, emb)



if __name__ == "__main__":
    main()
