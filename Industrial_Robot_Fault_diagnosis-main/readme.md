Industrial_robot_fault_diagnosis

An MCP-Enabled Multi-Agent Framework with TimeKAN and MAKG-RAG for Industrial Robot Fault Diagnosis

1. Introduction

This repository provides the official implementation of Industrial_robot_fault_diagnosis, a multi-agent collaborative fault diagnosis system proposed in our paper.
The system is built upon the MAKG (Multi-Agent and Synergistic Knowledge Graph) paradigm and aims to address the challenges of accurate, interpretable, and efficient fault diagnosis in industrial robotic systems operating under complex and harsh conditions.

Unlike conventional model-centric diagnostic approaches or standalone large language model (LLM) reasoning systems, Industrial_robot_fault_diagnosis tightly integrates:

Lightweight time-series fault classification

Knowledge graph–enhanced retrieval-augmented generation (MAKG-RAG)

Multi-agent collaboration enabled by the Model Context Protocol (MCP)

This unified design enables closed-loop diagnosis that combines real-time perception with domain knowledge reasoning, supporting practical industrial maintenance decision-making.

2. System Overview

The proposed system consists of three tightly coupled components:

Time-Series Fault Diagnosis Agent (Diagnosis Agent)

Identifies fault categories from multivariate sensor time-series data using TimeKAN.

Maintenance Knowledge Retrieval Agent (MKRA)

Retrieves and organizes structured and unstructured maintenance knowledge using MAKG-RAG.

LLM Coordination Agent (via MCP)

Acts as a high-level planner and summarizer, coordinating tool calls and generating interpretable diagnostic reports.

All agents communicate through the Model Context Protocol (MCP), ensuring consistent context sharing and reliable tool integration.

3. Software Environment
3.1 Basic Environment

Python: 3.11 (base interpreter)

Environment Management: Conda (recommended)

3.2 Required Dependencies

The following Python packages are required:

4. Time-Series Fault Classification (TimeKAN)
4.1 Model Description

The system employs a lightweight TimeKAN (KAN-based Time–Frequency Decomposition Network) for industrial robot fault classification.

Key characteristics:

Multi-scale frequency decomposition

Multi-order KAN representation learning

Grouped depthwise convolutions for efficiency

High robustness with extremely low MACs

4.2 Model Files

The trained TimeKAN model and preprocessing artifacts are provided in:
checkpoint/
├── checkpoint.pth   # Trained TimeKAN classifier
└── scaler.pkl       # Standardization parameters
The scaler must be applied during inference to ensure consistency with training data.

5. Embedding Model and Fine-Tuning
5.1 Base Embedding Model

Model: BAAI/bge-m3

Embedding Dimension: 1024

This embedding model is used for:

Knowledge graph entity embedding

Semantic chunk embedding

Query embedding in MAKG-RAG

5.2 Fine-Tuning Configuration

Fine-Tuning Dataset:
datasets/
Fine-Tuning Script:
finetune/finetune.py
Only dataset paths and output model paths need to be specified in finetune.py.
No other code modifications are required.
Due to the large storage size of the fine-tuned embedding model, it is not included in this GitHub repository; users are required to perform the fine-tuning process locally to obtain the adapted vector model.

6. Knowledge Base for MAKG-RAG

6.1 Knowledge Graph
Knowledge_Graph.csv
The knowledge graph encodes structured industrial maintenance knowledge, including:

Fault types

Symptoms

Root causes

Maintenance actions

6.2 Semantic Chunks
Semantic_chunks.csv
This file contains semantically merged text chunks derived from industrial manuals, expert knowledge, and maintenance documents.

6.3 Vector Index Files

Vector indexes are constructed for:

Knowledge graph entities

Semantic chunks

These indexes enable efficient dense retrieval during MAKG-RAG inference.

7. Rebuilding the RAG Knowledge Base

If the base RAG corpus is modified, the following components must be rebuilt:

Knowledge Graph (KG.csv)

Semantic Chunks (SC.csv)

Corresponding Vector Index Files

Failing to rebuild these components may lead to inconsistent retrieval results and degraded diagnostic performance.

8. Multi-Agent Collaboration via MCP

The system adopts the Model Context Protocol (MCP) to coordinate heterogeneous agents and tools.

MCP enables:

Standardized context delivery to agents

Reliable invocation of lightweight models (e.g., TimeKAN)

Seamless integration of retrieval, reasoning, and generation

LLMs serve as high-level coordinators, while lightweight models act as efficient executors, significantly reducing computation cost and deployment complexity.

9. Experimental Data and Evaluation
9.1 Test Data

The complete end-to-end evaluation datasets are provided in:
TIMEKAN-main/data/

These datasets include:

Single-fault scenarios

Labeled multivariate time-series signals

They are used to evaluate:

TimeKAN classification accuracy and robustness

MAKG-RAG retrieval quality

Multi-agent diagnosis effectiveness

10. Reproducibility Guidelines

To reproduce the results reported in the paper:

Create a Python 3.11 environment

Install all required dependencies

(Optional) Fine-tune the embedding model

Ensure TimeKAN model and scaler are correctly placed

Verify KG, semantic chunks, and vector indexes

Run system-level diagnosis and evaluation

11. Notes and Limitations

This repository is intended for academic research and experimental validation.

The system is not optimized for real-time industrial deployment.

The architecture is modular and supports:

Alternative classifiers

Different embedding models

Multiple LLM backends

Domain-specific knowledge graph extensions

12. Citation

If you use Industrial_robot_fault_diagnosis in your research, please cite the corresponding paper describing the TimeKAN and MAKG-RAG framework.
