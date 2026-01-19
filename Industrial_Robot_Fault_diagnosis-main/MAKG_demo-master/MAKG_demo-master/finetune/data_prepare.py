# -*- coding: utf-8 -*-
# @file: make_ft_corpus.py
import glob
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from llama_index.legacy.finetuning import (
    generate_qa_embedding_pairs
)

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sklearn.model_selection import train_test_split
from llama_index.llms.litellm import LiteLLM as OpenAI
import os
project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]

DATA_DIR = os.path.join(project_dir, "data")

txt_FILES = glob.glob(os.path.join(DATA_DIR, "*.txt"))

TRAIN_CORPUS_FPATH = os.path.join(project_dir, "data/ft_train_corpus.json")
VAL_CORPUS_FPATH = os.path.join(project_dir, "data/ft_val_corpus.json")


def load_and_concatenate_md_files(md_files, verbose=False):

    if verbose:
        print(f"Loading txt files: {md_files}")

    all_content = []
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    all_content.append(content)
                    if verbose:
                        print(f"Loaded {file_path} - {len(content)} characters")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    concatenated_content = "\n\n".join(all_content)

    if verbose:
        print(f"Total concatenated content length: {len(concatenated_content)} characters")
        print(f"Number of files processed: {len(all_content)}")

    return concatenated_content


def create_documents_from_content(content, verbose=False):
    from llama_index.core import Document

    document = Document(text=content)
    docs = [document]

    if verbose:
        print(f"Created {len(docs)} document from concatenated content")

    return docs


def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter(chunk_size=250, chunk_overlap=0)
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


all_nodes = load_corpus(txt_FILES, verbose=True)
train_nodes, val_nodes = train_test_split(
    all_nodes,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
print(f"Train set size: {len(train_nodes)} nodes")
print(f"Validation set size: {len(val_nodes)} nodes")

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
if not OPENAI_KEY:
    raise RuntimeError("error")
llm = OpenAI(
    model="deepseek-chat",
    api_key=OPENAI_KEY,
    api_base=BASE_URL,
    temperature=0,
    timeout=120
)

qa_generate_prompt_tmpl = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a professor. Your task is to create {num_questions_per_chunk} questions for an upcoming quiz/examination based on the provided mechanical fault diagnosis-related text. The questions should cover diverse aspects of the content, ensuring a variety of question types without repetition. The questions should focus on mechanical fault diagnosis, without options, and should not start with "Q1" or "Q2". They should be closely aligned with the provided context, aiming to assess the students' understanding of mechanical fault diagnosis theory and practice.
"""

train_dataset = generate_qa_embedding_pairs(
    nodes=train_nodes,
    llm=llm,
    num_questions_per_chunk=1,
    qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
)
val_dataset = generate_qa_embedding_pairs(
    nodes=val_nodes,
    llm=llm,
    num_questions_per_chunk=1,
    qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
)

train_dataset.save_json(TRAIN_CORPUS_FPATH)
val_dataset.save_json(VAL_CORPUS_FPATH)
print(f"Training dataset saved to: {TRAIN_CORPUS_FPATH}")
print(f"Validation dataset saved to: {VAL_CORPUS_FPATH}")
