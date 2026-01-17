import os
import numpy as np
import requests
from embeddings.embedder import embed_query
from vector_store.faiss_store import load_faiss
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN")

LLM_URL = (
    "https://api-inference.huggingface.co/models/"
    "mistralai/Mistral-7B-Instruct"
)

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

index, metadata = load_faiss()

def answer_question(query, k=5):
    q_emb = embed_query(query)
    D, I = index.search(np.array([q_emb]), k)

    context = ""
    for idx in I[0]:
        c = metadata[idx]
        context += f"(Page {c['page']} | {c['modality']}): {c['content']}\n"

    prompt = f"""
Answer ONLY using the context below.
Include page numbers.

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        LLM_URL,
        headers=headers,
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7
            }
        }
    )

    return response.json()[0]["generated_text"].split("Answer:")[-1].strip()
