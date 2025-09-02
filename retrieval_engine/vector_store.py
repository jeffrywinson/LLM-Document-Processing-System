# your-llm-project/retrieval_engine/vector_store.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import config

def create_vector_store(chunks):
    """Creates a FAISS vector store from document chunks and saves it locally."""
    print("Creating FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': config.EMBEDDING_DEVICE}
    )
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(config.DB_FAISS_PATH)
    print(f"FAISS index created and saved to '{config.DB_FAISS_PATH}'")
    return db

def get_retriever():
    """Loads the saved FAISS vector store and returns a retriever."""
    print(f"Loading FAISS index from '{config.DB_FAISS_PATH}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': config.EMBEDDING_DEVICE}
    )
    db = FAISS.load_local(
        config.DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db.as_retriever(search_kwargs={'k': config.RETRIEVER_K})