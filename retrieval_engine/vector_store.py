# retrieval_engine/vector_store.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define paths and model names
DB_FAISS_PATH = 'vectorstore/db_faiss'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def create_vector_store(chunks):
    """Creates a FAISS vector store and saves it."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    print("Creating FAISS vector store...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS index created and saved to {DB_FAISS_PATH}")

# --- NEW FUNCTION FOR YOUR APPLICATION TO CALL ---
def get_retriever():
    """Loads the FAISS index and returns a retriever object."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={'k': 3}) # Retrieve top 3 chunks