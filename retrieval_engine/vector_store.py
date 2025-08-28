# vector_store.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Define the path for the FAISS vector store
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_store(chunks):
    """
    Creates a FAISS vector store from the given text chunks and saves it to disk.
    """
    # Use a pre-trained model from Hugging Face to create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    
    print("Creating FAISS vector store...")
    # Create the vector store from the documents and embeddings
    db = FAISS.from_documents(chunks, embeddings)
    
    # Save the vector store locally
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS index created and saved to {DB_FAISS_PATH}")
    return db