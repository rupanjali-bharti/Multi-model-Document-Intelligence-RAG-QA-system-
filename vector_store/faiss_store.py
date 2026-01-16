import faiss
import numpy as np
import pickle

def save_faiss(vectors, metadata, index_path="faiss.index"):
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    faiss.write_index(index, index_path)

    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

def load_faiss(index_path="faiss.index"):
    index = faiss.read_index(index_path)

    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return index, metadata
