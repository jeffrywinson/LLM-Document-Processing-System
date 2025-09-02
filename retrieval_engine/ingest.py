# your-llm-project/retrieval_engine/ingest.py

import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(data_dir):
    """Loads all supported documents from a given directory."""
    documents = []
    print(f"Loading documents from '{data_dir}'...")
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                continue  # Skip unsupported files
            documents.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Warning: Could not load file '{filename}'. Reason: {e}")
    return documents

def split_into_chunks(documents):
    """Splits the loaded documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} document pages into {len(chunks)} chunks.")
    return chunks