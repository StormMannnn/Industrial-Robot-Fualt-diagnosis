import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from tqdm import tqdm
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
if not OPENAI_API_KEY:
    raise RuntimeError("error")



import pandas as pd
df = pd.read_csv("ragas1/data/qa_pairs.csv", encoding='gbk')
# sample_df = df.sample(n=109, random_state=1)  # random_state for reproducibility
# print(sample_df)
# df = sample_df
print(df.columns.tolist())
print(df.head(2))
llm = ChatOpenAI(model="deepseek-chat",
    base_url=BASE_URL,
    api_key=OPENAI_API_KEY,
    temperature=0,
    max_tokens=4096,)

from langchain.prompts import ChatPromptTemplate
# Combine all contexts into a single string
#final_context = "\n\n".join(all_contexts)
#print(final_context)

# Create prompt template
PROMPT_TEMPLATE = """
You are a professional mechanical fault diagnosis specialist who can answer{question}.Please answer in English, thank you.
You only need to answer the question without adding any other description.
Please think step by step.
"""
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

query= df['user_input']

responses = []
for idx, question_text in tqdm(
    enumerate(query),
    desc="Processing Questions",
    total=len(query),
    ncols=100
):
    try:
        prompt = prompt_template.format(question=question_text)
        response_text = llm.predict(prompt)
        clean_answer = response_text.replace('\n', ' ').strip()
        responses.append(clean_answer)
    except Exception as e:
        print(f"\n❌ Error at index {idx}: {e}")
        responses.append(f"[ERROR]")

df['answer'] = responses

df.to_csv("Your path", index=False)

print('done!')
