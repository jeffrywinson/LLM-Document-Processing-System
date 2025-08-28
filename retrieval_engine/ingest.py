# ingest.py

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_documents(data_dir):
    """
    Loads all supported documents from a given directory.
    """
    documents = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
    
    print(f"Successfully loaded {len(documents)} documents.")
    return documents

def split_into_chunks(documents):
    """
    Splits the loaded documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks