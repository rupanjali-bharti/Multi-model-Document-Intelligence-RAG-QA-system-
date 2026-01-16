from ingestion.extract_text import extract_text
from ingestion.extract_tables import extract_tables
from ingestion.extract_images_ocr import extract_images_ocr
from chunking.chunker import chunk_text
from embeddings.embedder import embed_documents
from vector_store.faiss_store import save_faiss

PDF_PATH = "data/source.pdf"

raw_chunks = []
raw_chunks.extend(extract_text(PDF_PATH))
raw_chunks.extend(extract_tables(PDF_PATH))
raw_chunks.extend(extract_images_ocr(PDF_PATH))

final_chunks = []
for chunk in raw_chunks:
    final_chunks.extend(
        chunk_text(
            chunk["content"],
            chunk["page"],
            chunk["modality"]
        )
    )

texts = [c["content"] for c in final_chunks]
metadata = final_chunks

vectors = embed_documents(texts)
save_faiss(vectors, metadata)

print("✅ FAISS index built successfully")
