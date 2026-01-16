import os
import uuid
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

# Import your local embedding logic
from embedder import embed_documents, embed_query

# ------------------------
# CONFIGURATION & ENV
# ------------------------
load_dotenv()

# Setup Hugging Face Client (Mistral v0.3 is the stable 2026 choice)
HF_TOKEN = os.getenv("HF_API_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# ------------------------
# 1. PDF INGESTION (OCR & TABLES)
# ------------------------
def ingest_pdf(pdf_path):
    loader = UnstructuredLoader(
        file_path=pdf_path,
        api_key=os.getenv("UNSTRUCTURED_API_KEY"),
        partition_via_api=True,      # Keeps RAM usage near zero
        strategy="hi_res",           # Extracts tables and performs OCR
        chunking_strategy="by_title" 
    )
    return loader.load()

# ---------------------------------
# EMBEDDING ADAPTER FOR CHROMA
# ---------------------------------
class HFEmbeddingWrapper:
    def embed_documents(self, texts):
        return embed_documents(texts)
    def embed_query(self, text):
        return embed_query(text)

# ------------------------
# 2. VECTOR STORE
# ------------------------
def build_vectorstore(chunks):
    vectordb = Chroma(
        collection_name="multimodal_rag",
        embedding_function=HFEmbeddingWrapper(),
        persist_directory="./chroma_db"
    )

    docs = []
    for i, chunk in enumerate(chunks):
        # We ensure content is string and add metadata for citations
        docs.append(
            Document(
                page_content=str(chunk.page_content),
                metadata={
                    "chunk_id": i,
                    "source": "PDF",
                    "type": chunk.metadata.get("category", "text")
                }
            )
        )

    vectordb.add_documents(docs)
    return vectordb

# ------------------------
# 3. QA PIPELINE
# ------------------------
def generate_answer(context, query):
    try:
        # Using chat_completion ensures compatibility with 2026 Inference Providers
        response = client.chat_completion(
            model=MODEL_ID,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Answer the question ONLY using the provided context. If the answer is not in the context, say you don't know. Cite your sources as [Source 1], [Source 2], etc."
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            max_tokens=512,
            temperature=0.3
        )
        # Access the content from the chat response object
        return response.choices[0].message.content
        
    except Exception as e:
        # Enhanced error message for common 2026 API issues
        error_msg = str(e)
        if "503" in error_msg or "loading" in error_msg.lower():
            return "The Qwen model is currently loading on the server. Please wait 30 seconds."
        elif "Authorization" in error_msg:
            return "HF_API_TOKEN error. Please check your token scopes in Hugging Face settings."
        return f"Hugging Face Error: {error_msg}"
    

    
def answer_query(vectordb, query):
    # Retrieve top 4 relevant chunks
    docs = vectordb.similarity_search(query, k=4)

    context = "\n\n".join(
        f"[Source {i+1}] {doc.page_content}"
        for i, doc in enumerate(docs)
    )

    answer = generate_answer(context, query)
    return answer, docs

# ------------------------
# 4. EXECUTION
# ------------------------
if __name__ == "__main__":
    pdf_path = "data/source.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found at {pdf_path}")
    else:
        print("📄 Reading PDF and extracting tables/images...")
        chunks = ingest_pdf(pdf_path)
        print(f"✅ Extracted {len(chunks)} elements")

        print("🔍 Building vector DB...")
        vectordb = build_vectorstore(chunks)
        print("✅ Vector store ready")

        query = "What are the main findings in the document?"
        print(f"❓ Query: {query}")
        
        answer, sources = answer_query(vectordb, query)

        print("\n🧠 ANSWER:\n")
        print(answer)

        print("\n📚 CITATIONS:")
        for i, src in enumerate(sources):
            # Show the type (Table, Title, NarrativeText, etc.)
            print(f"[Source {i+1}] Type: {src.metadata['type']}")