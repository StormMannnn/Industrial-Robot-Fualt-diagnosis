#%%
import pandas as pd
import json
import re
from pathlib import Path
from typing import Any, List, Dict
from tqdm import tqdm

chunks_csv = "data/semantic_chunks_cleaned.csv"
kg_csv = "data/knowledge_graph.csv"
output_csv = "data/qa_pairs.csv"

chunks_df = pd.read_csv(chunks_csv)
kg_df = pd.read_csv(kg_csv)

chunk_to_triples: Dict[int, List[Dict[str, str]]] = {}
for _, row in kg_df.iterrows():
    cid = row.get("chunk_id", None)
    if pd.isna(cid):
        continue
    try:
        cid_int = int(cid)
    except Exception:
        continue
    triple = {
        "head": str(row.get("node_1", "")).strip(),
        "relation": str(row.get("edge", "")).strip(),
        "tail": str(row.get("node_2", "")).strip(),
    }
    if not triple["head"] or not triple["relation"] or not triple["tail"]:
        continue
    chunk_to_triples.setdefault(cid_int, []).append(triple)

chunks = chunks_df["text"].tolist()
sources = chunks_df["source"].tolist() if "source" in chunks_df.columns else [""] * len(chunks)

total_target = 200
per_chunk = max(1, total_target // max(1, len(chunks)))
remainder = total_target % max(1, len(chunks))

indices = list(range(len(chunks)))


from typing import Optional
import os
from litellm import completion as llm_completion

DEFAULT_MODEL = "deepseek-chat"
BASE_URL = ""
API_KEY = ""

def llm_chat(messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, **kwargs) -> str:
    m = model or DEFAULT_MODEL
    try:
        resp = llm_completion(model=m, messages=messages, api_base=BASE_URL, api_key=API_KEY, temperature=temperature, **kwargs)
        content = None
        if isinstance(resp, dict):
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            choices = getattr(resp, "choices", [])
            if choices:
                msg = getattr(choices[0], "message", None)
                if msg:
                    try:
                        content = msg.get("content")
                    except Exception:
                        content = getattr(msg, "content", "")
        return (content or "").strip()
    except Exception:
        return ""


def _build_prompt(content: str, k: int) -> str:
    return (
        "You are an industrial document annotation assistant. Please generate high-quality English Q&A pairs based on the provided content.\n"
        "Strict Requirements:\n"
        "1. All questions and answers must be strictly based on the provided content, without introducing external knowledge;\n"
        "2. Each data item should be output as an object with fields: user_input (concise and natural English user question), reference_contexts (key sentences selected from the content), reference (detailed and content-rich English answer based on these contexts);\n"
        "3. The reference must be detailed, content-rich, and directly supported by reference_contexts;\n"
        "4. If the content is too brief, fragmented, or insufficient to form complete Q&A pairs (such as simple list items or phrases), do not generate any Q&A pairs;\n"
        "5. Avoid generating Q&A pairs from overly brief content (like simple entries such as '(5) Maintenance operations must be performed with power on...)';\n"
        "6. Output strictly as a JSON array, without any extra text or code blocks.\n\n"
        "Quality Requirements:\n"
        "- user_input should be concise and natural, like questions real users would ask\n"
        "- reference should be content-rich and detailed, providing complete explanations\n"
        "- Only generate high-quality, substantive Q&A pairs\n\n"
        f"Please generate 1 to {k} high-quality data items. If the content does not meet requirements, output an empty array [].\n\n"
        f"Content:\n{content}\n\n"
        "Output only JSON array:"
    )

results: List[Dict[str, Any]] = []
remaining = total_target
triples_per_doc = 3

for idx, i in enumerate(tqdm(indices, desc="Q&A pairs Generating")):
    if remaining <= 0:
        break

    txt = (chunks[i] or "").strip()
    src = sources[i] if i < len(sources) else ""

    triples_txt = ""
    tri_list = chunk_to_triples.get(i, [])
    if tri_list:
        lines: List[str] = []
        for t in tri_list[: max(1, triples_per_doc)]:
        # for t in tri_list:
            h = str(t.get("head", "")).strip()
            r = str(t.get("relation", "")).strip()
            ta = str(t.get("tail", "")).strip()
            if h and r and ta:
                lines.append(f"- ({h}) --{r}--> ({ta})")
        if lines:
            triples_txt = "\n\nKey Triples:\n" + "\n".join(lines)

    content = txt

    k_this = per_chunk + (1 if idx < remainder else 0)
    k_this = min(k_this, remaining)
    if k_this <= 0:
        continue

    prompt = _build_prompt(content, k_this)
    resp = llm_chat(
        messages=[
            {"role": "system", "content": "Output only a JSON array, with each element containing user_input, reference_contexts, and reference."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    arr: List[Any] = []
    try:
        m = re.search(r"\[.*\]", resp, flags=re.S)
        arr = json.loads(m.group(0) if m else resp)
    except Exception:
        arr = []

    for obj in arr:
        if remaining <= 0:
            break
        if not isinstance(obj, dict):
            continue
        q = str(obj.get("user_input", "")).strip()
        rc = obj.get("reference_contexts", [])
        if isinstance(rc, str):
            rc = [rc]
        if not isinstance(rc, list):
            rc = []
        rc_clean = []
        for x in rc:
            sx = str(x).strip()
            if sx:
                rc_clean.append(sx)
        rc_clean = rc_clean[:3]
        ans = str(obj.get("reference", "")).strip()
        if not q or not ans or not rc_clean:
            continue
        results.append({
            "user_input": q,
            "reference_contexts": rc_clean,
            "reference": ans,
            "source": src,
            "chunk_id": i,
        })
        remaining -= 1

outp = Path(output_csv)
outp.parent.mkdir(parents=True, exist_ok=True)

if results:
    df_output = pd.DataFrame(results)
    df_output.to_csv(outp, index=False, encoding='utf-8')
    print({"ok": True, "count": len(results), "out_path": str(outp)})
else:
    print({"ok": False, "count": 0, "out_path": str(outp), "message": "No valid Q&A pairs generated"})

