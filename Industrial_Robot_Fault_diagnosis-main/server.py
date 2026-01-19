import os
import csv
import time
from pathlib import Path
from typing import List
import faiss
import networkx as nx
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from models.TimeKAN import Model as TimeKANModel


mcp = FastMCP(name="test")



class DiagnoseInput(BaseModel):
    query: str = Field(..., description="CSV file path")
    window_size: int = Field(100, description="Time window size")
    device_str: str = Field('cuda', description="Device selection: 'CPU' or 'CUDA'")


@mcp.tool()
def predict_timeseries_classification(
    query: str,
    model_path: str = './checkpoint/checkpoint.pth',
    output_path: str = "./predictions",
    seq_len: int = 96,
        batch_size: int = 32,
        num_workers: int = 0,
        enc_in: int = 12,
        scaler_path: str = './checkpoint/scaler.pkl',

) -> dict:
    import os
    import tempfile
    from urllib.parse import urlparse
    import requests
    import pandas as pd
    import numpy as np
    import torch
    import pickle
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from types import SimpleNamespace

    csv_source = query

    parsed = urlparse(csv_source)
    if parsed.scheme in ('http', 'https'):
        print(f"Downloading file from URL:{csv_source}")
        try:
            response = requests.get(csv_source, timeout=30)
            response.raise_for_status()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            temp_file.write(response.content)
            temp_file.close()
            csv_file_path = temp_file.name
            print(f"The file has been downloaded to a temporary path:{csv_file_path}")
        except Exception as e:
            return {"status": "error", "message": f"Failed to download file from URL: {e}"}
    else:
        # 假设是本地路径
        if not os.path.exists(csv_source):
            return {"status": "error", "message": f"The local file does not exist:{csv_source}"}
        csv_file_path = csv_source

    def csv_to_ts_unlabeled(csv_path, out_ts_path, seq_len=96, expected_dim=12):
        df = pd.read_csv(csv_path)
        n_rows, n_cols = df.shape
        if n_cols != expected_dim:
            raise ValueError(f"CSV Column Count {n_cols} != expected_dim {expected_dim}Please check the CSV column order/quantity.")

        if n_rows < seq_len:
            pad_rows = seq_len - n_rows
            last_row = df.iloc[[-1]]
            pad_df = pd.concat([last_row] * pad_rows, ignore_index=True)
            df = pd.concat([df, pad_df], ignore_index=True)
            n_rows = len(df)
            num_seqs = 1
        else:
            num_seqs = n_rows - seq_len + 1

        lines = []
        for k in range(num_seqs):
            start = k
            end = start + seq_len
            block = df.iloc[start:end]
            dim_parts = []
            for dim in range(expected_dim):
                col_vals = block.iloc[:, dim].astype(float).astype(str).tolist()
                dim_parts.append(','.join(col_vals))
            line = ':'.join(dim_parts)
            lines.append(line)

        with open(out_ts_path, 'w') as f:
            f.write("@problemName converted_from_csv_unlabeled\n")
            f.write("@timeStamps false\n")
            f.write("@missing false\n")
            f.write("@univariate false\n")
            f.write(f"@dimesion {expected_dim}\n")
            f.write("@equallength true\n")
            f.write(f"@serieslength {seq_len}\n")
            f.write("@targetlabel false\n")
            f.write("@classlabel false\n")
            f.write("@data\n")
            for ln in lines:
                f.write(ln + "\n")
        print(f"Generated unlabeled. ts:{out_ts_path}(Number of sequences:{len(lines)})")
        return out_ts_path

    def parse_ts_file_unlabeled(file_path, default_dim=12, default_length=96):
        sequences = []
        labels = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            raise RuntimeError(f"open {file_path} failed : {e}")

        dimension = default_dim
        series_length = default_length
        data_start = False
        data_lines = []

        for line in lines:
            s = line.strip()
            if not s:
                continue
            low = s.lower()
            if low.startswith('@dimension') or low.startswith('@dimesion'):
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        dimension = int(parts[1])
                    except:
                        pass
                continue
            if low.startswith('@serieslength'):
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        series_length = int(parts[1])
                    except:
                        pass
                continue
            if low.startswith('@data'):
                data_start = True
                continue
            if data_start:
                data_lines.append(s)

        for lineno, line in enumerate(data_lines, start=1):
            parts = line.split(':')
            if len(parts) < 1:
                continue

            has_label = False
            if len(parts) == dimension:
                dim_parts = parts[:]
                has_label = False
            elif len(parts) == dimension + 1:
                dim_parts = parts[:-1]
                label_part = parts[-1].strip()
                has_label = True
            elif len(parts) > dimension + 1:
                label_part = parts[-1].strip()
                dim_parts = parts[:-1]
                if len(dim_parts) != dimension:
                    if len(dim_parts) > dimension:
                        extra = dim_parts[dimension - 1:]
                        dim_parts = dim_parts[:dimension - 1] + [':'.join(extra)]
                    else:
                        print(f"[WARN] Skip line {lineno}: Number of dimension segments ({len (dim_parts)})! =Expectations ({dimension})")
                        continue
                has_label = True
            else:
                print(
                    f"[WARN] Skip line {lineno}: The number of segments ({len (parts)}) is too small, expect {dimension} segments (or {dimension+1} to include labels)")
                continue

            data_values = []
            bad = False
            for dp in dim_parts:
                toks = [t.strip() for t in dp.split(',') if t.strip() != '']
                try:
                    data_values.extend([float(t) for t in toks])
                except Exception:
                    bad = True
                    break
            if bad:
                print(f"[WARN] Dimension data parsing failed on line {lineno}, skip")
                continue

            label_val = None
            if has_label:
                try:
                    label_val = float(label_part)
                except Exception:
                    label_val = None
                    has_label = False

            if len(data_values) != series_length * dimension:
                print(
                    f"[WARN] Line {lineno} length mismatch: obtained {len(data_values)} != {series_length * dimension} (series_len*dim)，skip")
                continue

            arr = np.array(data_values, dtype=np.float32).reshape(series_length, dimension)
            sequences.append(arr)
            if has_label:
                labels.append(int(label_val))
            else:
                labels.append(None)

        return sequences, labels

    # ---------- helper: build configs ----------
    def build_configs_from_checkpoint_or_defaults(checkpoint_obj, seq_len, enc_in, num_classes=7):

        if isinstance(checkpoint_obj, dict):
            for key in ('args', 'config', 'configs', 'best_args', 'settings'):
                if key in checkpoint_obj and checkpoint_obj[key] is not None:
                    cand = checkpoint_obj[key]
                    if hasattr(cand, '__dict__'):
                        return cand
                    if isinstance(cand, dict):
                        return SimpleNamespace(**cand)
        cfg = SimpleNamespace()
        cfg.task_name = 'classification'
        cfg.seq_len = int(seq_len)
        cfg.label_len = 0
        cfg.pred_len = 0
        cfg.down_sampling_window = 2
        cfg.channel_independence = False
        cfg.e_layers = 3
        cfg.down_sampling_layers = 3
        cfg.moving_avg = 25
        cfg.enc_in = int(enc_in)
        cfg.use_future_temporal_feature = False
        cfg.d_model = 64
        cfg.embed = 'timeF'
        cfg.freq = 'h'
        cfg.dropout = 0.1
        cfg.use_norm = 1
        cfg.features = 'M'
        cfg.begin_order = 1
        cfg.c_out = max(2, int(num_classes))
        cfg.num_classes = cfg.c_out
        cfg.pred_len = 0
        return cfg

    try:
        os.makedirs(output_path, exist_ok=True)

        ts_path = os.path.join(output_path, 'converted_unlabeled.ts')
        csv_to_ts_unlabeled(csv_file_path, ts_path, seq_len=seq_len, expected_dim=enc_in)

        seqs, labels = parse_ts_file_unlabeled(ts_path, default_dim=enc_in, default_length=seq_len)
        if len(seqs) == 0:
            return {"status": "error", "message": "After parsing the. ts file, no sequence was obtained. Please check the format (dimension/length/delimiter)."}

        print(f"✅ Parse to {len(seqs)} sequences, each with the following shape:{seqs[0].shape}")

        scaler = None
        if scaler_path is not None and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("✅ When training has been loaded scaler:", scaler_path)
        else:
            print("⚠️ No scaler_path provided or file does not exist, the current data will be fit()")
            sc = StandardScaler()
            all_data = np.vstack(seqs)
            sc.fit(all_data)
            scaler = sc
            print("⚠️ A temporary scaler has been generated for the current data fit() (for inference only)")

        seqs_t = [scaler.transform(s) for s in seqs]
        arr = np.stack(seqs_t, axis=0)  # shape [N, seq_len, dim]
        tensor = torch.FloatTensor(arr)

        dataset = TensorDataset(tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using equipment:", device)

        if not os.path.exists(model_path):
            return {"status": "error", "message": f"The model file does not exist: {model_path}"}

        checkpoint = torch.load(model_path, map_location='cpu')
        cfg = build_configs_from_checkpoint_or_defaults(checkpoint, seq_len=seq_len, enc_in=enc_in, num_classes=7)
        print(
            f"[DEBUG] configuration usage: task_name={getattr(cfg, 'task_name', None)}, seq_len={getattr(cfg, 'seq_len', None)}, enc_in={getattr(cfg, 'enc_in', None)}, e_layers={getattr(cfg, 'e_layers', None)}")

        # create model
        model = TimeKANModel(cfg).to(device)

        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state = checkpoint['model']
            else:
                state = checkpoint
        else:
            state = checkpoint

        if isinstance(state, dict):
            if not any(isinstance(v, torch.Tensor) for v in list(state.values())[:5]):
                for key in ('model_state_dict', 'state_dict', 'state'):
                    if key in state and isinstance(state[key], dict):
                        state = state[key]
                        break

        try:
            model.load_state_dict(state)
        except Exception as e:
            if isinstance(state, dict):
                new_state = {}
                for k, v in state.items():
                    new_key = k.replace('module.', '') if isinstance(k, str) else k
                    new_state[new_key] = v
                try:
                    model.load_state_dict(new_state)
                except Exception as e2:
                    return {"status": "error", "message": f"Failed to load model weights：{e2}\nOriginal error：{e}"}
            else:
                return {"status": "error", "message": f"Unrecognized state type: {type(state)}"}

        model.eval()
        print("✅ The model has been loaded and inference has begun ...")

        all_preds = []
        all_probs = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    batch_x = batch[0].to(device)
                else:
                    batch_x = batch.to(device)

                outputs = model(batch_x, None, None, None)

                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = torch.tensor(outputs).to(device)

                if logits.dim() == 3 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                if logits.dim() != 2:
                    logits = logits.view(logits.size(0), -1)

                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds.tolist())
                all_probs.extend(probs.tolist())
                if (i + 1) % 10 == 0:
                    print(f"Processed batches: {i + 1}/{len(dataloader)}")


        if len(all_probs) == 0:
            return {"status": "error", "message": "No predictions were generated (all_debs is empty)"}

        df = pd.DataFrame(all_probs, columns=[f'prob_class_{k}' for k in range(len(all_probs[0]))])
        df.insert(0, 'prediction', all_preds)
        out_csv = os.path.join(output_path, 'predictions_unlabeled_ts.csv')
        df.to_csv(out_csv, index=False)

        vc = pd.Series(all_preds).value_counts().sort_index()
        results_df = pd.read_csv("./predictions/predictions_unlabeled_ts.csv")

        analysis_report = analyze_predictions(results_df)
        print(analysis_report)
        result = {
            "status": "success",
            "message": f"Time series classification prediction completed: {len(all_preds)} sequences",
            "analysis_report": analysis_report,
        }

        print(f"✅ Inference completed, results saved to: {out_csv}")
        print("predictive distribution:")
        for idx, cnt in vc.items():
            print(f"  class {int(idx)}: {cnt} ({cnt / len(all_preds) * 100:.2f}%)")

    except Exception as e:
        result = {
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }

    return result



def analyze_predictions(results_df):

    class_mapping = {
        0: "Normal",
        1: "Axis 1 Gearbox Fault and Axis 2 Motor Fault",
        2: "Axis 1 and Axis 3 Gearbox Faults",
        3: "Axis 3 and Axis 4 Gearbox Faults",
        4: "Axis 3 Gearbox Fault",
        5: "Axis 2 Motor Fault",
        6: "Axis 4 Gearbox Fault"
    }

    total_predictions = len(results_df)

    class_distribution = results_df['prediction'].value_counts().sort_index()
    class_percentage = (class_distribution / total_predictions * 100).round(2)


    prob_columns = [col for col in results_df.columns if col.startswith('prob_class_')]
    results_df['confidence'] = results_df[prob_columns].max(axis=1)

    class_avg_confidence = results_df.groupby('prediction')['confidence'].mean().round(4)

    most_common_class = class_percentage.idxmax()
    most_common_percentage = class_percentage.max()
    most_common_avg_confidence = class_avg_confidence[most_common_class]

    overall_avg_confidence = results_df['confidence'].mean().round(4)

    class_details = []
    for class_id in sorted(class_distribution.index):
        count = class_distribution[class_id]
        percentage = class_percentage[class_id]
        avg_conf = class_avg_confidence.get(class_id, 0)
        class_name = class_mapping.get(class_id, f"Unknown category {class_id}")
        class_details.append(f"- {class_name}: {count} ({percentage}%), Average Confidence: {avg_conf:.4f}")

    report = f"""
Time Series Classification Analysis Report
====================

Basic statistics:
- Total Predictions:{total_predictions}
- Overall average confidence level: {overall_avg_confidence:.4f}

Main diagnostic results:
- Main status: {class_mapping[most_common_class]}
- Status proportion: {most_common_percentage:.2f}%
- Average confidence level of main states: {most_common_avg_confidence:.4f}

Detailed category statistics:
{chr(10).join(class_details)}

Data quality assessment:
- Prediction consistency: {'high' if most_common_percentage > 80 else 'medium' if most_common_percentage > 60 else 'low'}
- Confidence level: {'high' if overall_avg_confidence > 0.9 else 'medium' if overall_avg_confidence > 0.7 else 'low'}
"""


    main_status = class_mapping[most_common_class]

    fault_percentages = {
        class_mapping[k]: v
        for k, v in class_percentage.items()
        if class_mapping[k] != "Normal"
    }

    if main_status == "Normal" and most_common_percentage >= 98:
        return "The robot is working normally without any abnormalities!"

    if main_status == "Normal":
        over_10_faults = {k: v for k, v in fault_percentages.items() if v > 10}
        if over_10_faults:
            top_fault = max(over_10_faults.items(), key=lambda x: x[1])[0]
            return f"In industrial robots{top_fault},how should we handle it?"
        else:
            return "The robot is working normally without any abnormalities!"

    sorted_faults = sorted(
        fault_percentages.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_fault, top_pct = sorted_faults[0]

    dominant = True
    for _, pct in sorted_faults[1:]:
        if top_pct - pct < 10:
            dominant = False
            break

    if dominant:
        return f"In industrial robots{top_fault},how should we handle it?"

    similar_faults = [
        fault for fault, pct in sorted_faults
        if abs(pct - top_pct) <= 10
    ]

    if len(similar_faults) >= 2:
        fault_text = "，".join(similar_faults)
        return (
            f"In industrial robots{fault_text},"
            f"how to deal with these multiple composite faults occurring simultaneously？"
        )

    return f"In industrial robots{top_fault},how should we handle it?"








graph_path = '.\data\KG_data\knowledge_graph.csv'
chunks_path = '.\data\KG_data\semantic_chunks.csv'
index_file = './checkpoint/vector_index.faiss'
chunk_ids_file = './checkpoint/chunk_ids.csv'
chunk_text_index_file='.\checkpoint\chunk_text_index.faiss'
chunk_text_ids_file='.\checkpoint\chunk_text_ids.csv'
graph_df = pd.read_csv(graph_path)
chunks_df = pd.read_csv(chunks_path)

graph_df['chunk_id'] = graph_df['chunk_id'].astype(str)
chunks_df['chunk_id'] = chunks_df['chunk_id'].astype(str)

print("✅ graph_df columns:", graph_df.columns.tolist())
print("✅ chunks_df columns:", chunks_df.columns.tolist())


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = 'bge-m3'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_tensor=False, normalize_embeddings=True).tolist()[0]


embedding_model = SentenceTransformerEmbeddings(model_name='./checkpoint/bge-m3/checkpoint-200')

knowledge_graph = nx.DiGraph()
knowledge_graph.add_edges_from([("n1", "n2"), ("n2", "n3"), ("n3", "n1")])


def extract_subgraph(key_nodes):
    if not key_nodes:
        return nx.DiGraph()
    subgraph_nodes = set()
    for key_node in key_nodes:
        subgraph_nodes.add(key_node)
        one_hop_nodes = {edge[0] if edge[1] == key_node else edge[1]
                         for edge in knowledge_graph.edges() if key_node in edge}
        two_hop_nodes = set()
        for one_hop_node in one_hop_nodes:
            adjacent_nodes = set(knowledge_graph.neighbors(one_hop_node))
            adjacent_nodes.discard(key_node)
            two_hop_nodes.update(adjacent_nodes)
        subgraph_nodes.update(one_hop_nodes)
        subgraph_nodes.update(two_hop_nodes)
    return knowledge_graph.subgraph(subgraph_nodes)



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

        subgraph = extract_subgraph(key_nodes)

        one_hop_triplets = set()
        two_hop_triplets = set()

        for u, v in subgraph.edges():
            t = f"({u}, connected, {v})"
            unique_triplets.add(t)

            if u in key_nodes or v in key_nodes:
                one_hop_triplets.add(t)
            else:
                two_hop_triplets.add(t)

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


# ===================== LLM =====================
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


# ===================== Query process =====================
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


# ===================== Optimize context (directly concatenate the top 5 chunks) =====================
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
        The edited context text, with minimal removal (≤150 words). Preserve original phrasing and structure.

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


# ===================== Chunk text vector index (independent of triplet index) =====================

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
        embeddings = embedding_model.embed_documents(texts)  # list of list[float]

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


@mcp.tool()
def query_knowledge_graph(query: str) -> dict:
    """
        A question answering tool based on knowledge graph (supporting dual path retrieval: triplet+chunk text vector)
        Args:
        Query: User's question
        Returns:
        Dict: a dictionary containing answers and related information
    """
    try:
        start_time = time.time()
        print(f"The received issue is：{query}")
        print("🔍 Decomposing query...")
        sub_queries = rewrite_and_split_query(query)
        print(f"The rewritten sub question is as follows：{sub_queries}")
        print("🔍 Performing Triplet based retrieval (including multi hop extensions)...")
        triplets, chunks_from_triplet = query_database([sq["question"] for sq in sub_queries], top_k=5)

        print("🔍 Executing Chunk Text vector retrieval...")
        text_retrieval_results = retrieve_from_chunk_text(query, top_k=5)
        chunks_from_text = [text for _, _, text in text_retrieval_results]
        chunk_ids_from_text = [cid for cid, _, _ in text_retrieval_results]
        print("\n📄 Chunk Text Vector Retrieval Results (Top-5):")
        for i, (cid, score, text) in enumerate(text_retrieval_results, start=1):
            print(f"  [{i}] Score: {score:.4f} | Chunk ID: {cid}")
            print(f"      Text: {text[:200]}{'...' if len(text) > 200 else ''}\n")
        merged_chunks_dict = {}

        for text in chunks_from_triplet:
            match_row = chunks_df[chunks_df['text'] == text]
            if not match_row.empty:
                cid = match_row['chunk_id'].iloc[0]
                merged_chunks_dict[cid] = (text, 0.3)  # Confidence score

        for cid, score, text in zip(chunk_ids_from_text, [r[1] for r in text_retrieval_results], chunks_from_text):
            if cid not in merged_chunks_dict:
                merged_chunks_dict[cid] = (text, score)
            else:
                if score > merged_chunks_dict[cid][1]:
                    merged_chunks_dict[cid] = (text, score)

        sorted_merged = sorted(merged_chunks_dict.items(), key=lambda x: x[1][1], reverse=True)
        final_chunks = [text for _, (text, _) in sorted_merged[:5]]
        print("\n✅ After final merging and reordering Top-5 Chunks:")
        for i, (cid, (text, score)) in enumerate(sorted_merged[:5], start=1):
            print(f"  [{i}] Score: {score:.4f} | Chunk ID: {cid}")
            print(f"      Text: {text[:200]}{'...' if len(text) > 200 else ''}\n")
        print("📄 Final_chunks content (plain text list):")
        for i, text in enumerate(final_chunks, start=1):
            print(f"  Chunk {i}: {text[:200]}{'...' if len(text) > 200 else ''}")
        print()
        print("🔍 Optimizing context...")
        retrieval_context = optimize_contexts(query, final_chunks, top_k=5)
        print(f"The optimized context is：{retrieval_context}")
        print("🤖 Generating answers...")

        PROMPT_TEMPLATE ="""
         You are required to answer the question strictly based on the provided Triplets and contextual information.
        
        Information:
        {context}
        
        Question:
        {question}
        
        Task Instructions (must be followed strictly):
        
        1. First, carefully analyze the question and clearly identify the exact fault type(s) mentioned in the query.
           The fault type(s) explicitly stated in {question} must be used as-is.
           You are strictly forbidden from modifying, reinterpreting, generalizing, or substituting the fault type(s) based on retrieved information.
        
        2. If the question involves multiple faults, you must provide one single, unified solution that comprehensively addresses all mentioned faults together.
           Do NOT answer them separately.
        
        3. Before presenting the solution, you must restate the fault in the following fixed template:
           “According to the fault described in {question}, where a specific axis of the robot exhibits the specified fault, the solution is as follows:”
        
        4. The solution must be written as ONE continuous paragraph.
           Do NOT split the solution into bullet points, numbered lists, sections, or line breaks.
        
        5. To ensure sufficient detail and completeness, the solution paragraph should explicitly and cohesively cover:
           - practical handling or treatment methods for the fault,
           - inspection and verification actions that should be carried out,
           - adjustment, repair, or replacement measures if applicable,
           - operational considerations to restore normal robot functionality.
        
        6. The solution must be highly specific, direct, and tightly aligned with the fault described in the question.
           Avoid generic statements and avoid mentioning unrelated fault types.
        
        7. Do NOT provide any reasoning, explanation, justification, or analysis.
           Do NOT explain why the solution works.
        
        8. Do NOT reference the context explicitly.
           Avoid phrases such as “according to the context”, “based on the provided information”, or similar expressions.
        
        9. Do NOT include any information that is not present in the given context.
           Do NOT fabricate details, assumptions, or external knowledge.
        
        10. Output ONLY the final answer.
            Do NOT include your thinking process, intermediate steps, or any meta commentary.
        
        The final output must be written entirely in English.
        """

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=retrieval_context, question=query)
        response_text = llm.predict(prompt)
        print(f"The answer is: {response_text}")
        processing_time = time.time() - start_time

        result = {
            "status": "success",
            "answer": response_text,
            "processing_time": f"{processing_time:.2f}s",
            "retrieved_chunks_count": len(final_chunks),
            "retrieved_triplets_count": len(triplets)
        }

        print(f"✅ Answer generation completed, time required:{processing_time:.2f}s")

    except Exception as e:
        result = {
            "status": "error",
            "message": f"Error processing query: {str(e)}"
        }
        print(f"❌ error: {str(e)}")

    return result


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    mcp.run(
        transport="sse",host="0.0.0.0",port=8000
    )


