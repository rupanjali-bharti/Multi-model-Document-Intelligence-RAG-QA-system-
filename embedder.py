import os
import requests
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")

# FIX: Force the 'feature-extraction' task directly in the URL
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}/pipeline/feature-extraction"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def _request_embeddings(inputs):
    """Core helper to get raw vectors from the API."""
    payload = {
        "inputs": inputs,
        "options": {"wait_for_model": True, "use_cache": True}
    }
    
    response = requests.post(EMBEDDING_URL, headers=HEADERS, json=payload)
    
    if response.status_code != 200:
        raise RuntimeError(f"HF API error {response.status_code}: {response.text}")
    
    data = response.json()

    # The feature-extraction pipeline returns a 3D list: [batch, sequence, hidden_dim]
    # We need to average these word-vectors into one 'Sentence Vector' (Mean Pooling)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list) and isinstance(data[0][0], list):
        # We manually average the vectors so we don't need the NumPy library
        pooled_vectors = []
        for sequence in data:
            n_tokens = len(sequence)
            n_dim = len(sequence[0])
            mean_vec = [sum(row[i] for row in sequence) / n_tokens for i in range(n_dim)]
            pooled_vectors.append(mean_vec)
        return pooled_vectors
    
    return data

def embed_documents(texts):
    """Standardize input and call the API."""
    if isinstance(texts, str): texts = [texts]
    return _request_embeddings(texts)

def embed_query(query):
    """Return a single vector for the user's query."""
    result = _request_embeddings([query])
    return result[0] if result else []

# --- Quick Test ---
if __name__ == "__main__":
    try:
        sample = ["Paris is the capital of France.", "Python is a language."]
        vectors = embed_documents(sample)
        print(f"✅ Success! Created {len(vectors)} vectors.")
        print(f"First vector length: {len(vectors[0])}") # Should be 384
    except Exception as e:
        print(f"❌ Failed: {e}")