import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm


DASHSCOPE_API_KEY = "Your Key"
if not DASHSCOPE_API_KEY:
    raise RuntimeError("Missing DASHSCOPE_API_KEY. Please set it in .env file.")

import pandas as pd
df = pd.read_csv("ragas1/data/qa_pairs.csv")
print(df.columns.tolist())
print(df.head(2))

# Use Qwen3 via DashScope (LangChain integration)
from langchain_community.chat_models import ChatTongyi

llm = ChatTongyi(
    model="qwen3-next-80b-a3b-thinking",  # or "qwen-max", "qwen-plus" if you prefer; "qwen3" is the latest open-weight version
    dashscope_api_key=DASHSCOPE_API_KEY,
    temperature=0,
    max_tokens=4096,
)

from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
You are a professional mechanical fault diagnosis expert capable of answering the question: {question}.
Please answer in English.
Provide only the final answer without any additional explanations or descriptions.
Think step by step.
"""
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

queries = df['user_input']
responses = []

for idx, question_text in tqdm(
    enumerate(queries),
    desc="Processing Questions",
    total=len(queries),
    ncols=100
):
    try:
        prompt = prompt_template.format(question=question_text)
        response_text = llm.predict(prompt)
        clean_answer = response_text.replace('\n', ' ').strip()
        responses.append(clean_answer)
    except Exception as e:
        print(f"\n❌ Error at index {idx}: {e}")
        responses.append("[ERROR]")

df['answer'] = responses
df.to_csv("Qwen3-thinking_output3.csv", index=False)

print('✅ Done! Results saved to Qwen3_output2.csv')