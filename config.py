# your-llm-project/config.py

import torch

# --- Directory and Path Settings ---
# Assumes your documents are in a subfolder within the retrieval_engine
DOCUMENTS_DIR = "retrieval_engine/documents"
DB_FAISS_PATH = "vectorstore/db_faiss"

# --- Embedding Model Settings ---
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- LLM Settings ---
LLM_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# --- Retriever Settings ---
RETRIEVER_K = 4