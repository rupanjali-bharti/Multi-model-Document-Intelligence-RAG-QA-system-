import streamlit as st
import os
# Import the correct names from your multimodal_rag.py
from multimodal_rag import ingest_pdf, build_vectorstore, answer_query

st.set_page_config(page_title="Multimodal RAG Bot", layout="wide")
st.title("🤖 Citation-Backed Multimodal Chatbot")

# Initialize retriever in session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Sidebar Upload
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Parsing text, tables, and images via Unstructured Cloud..."):
            # Save uploaded file temporarily
            temp_path = "temp_upload.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # 1. Ingest (OCR/Tables)
                chunks = ingest_pdf(temp_path)
                
                # 2. Build Vector DB
                st.session_state.retriever = build_vectorstore(chunks)
                
                st.success("Document Indexed Successfully!")
                
                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                st.error(f"Error processing document: {e}")

# Chat Interface
if query := st.chat_input("Ask a question about your document:"):
    if not st.session_state.retriever:
        st.warning("Please upload and process a document first.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.write(query)
            
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching and thinking..."):
                # Call the correct function name: answer_query
                answer, sources = answer_query(st.session_state.retriever, query)
                st.markdown(answer)
            
            # Show Citations
            with st.expander("🔍 View Citations & Metadata"):
                for i, doc in enumerate(sources):
                    # Pulling 'type' from the metadata we set in multimodal_rag.py
                    source_type = doc.metadata.get('type', 'Unknown')
                    st.markdown(f"**Source {i+1} (Category: {source_type})**")
                    st.info(doc.page_content)
                    st.divider()