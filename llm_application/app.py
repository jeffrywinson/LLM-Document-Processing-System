# llm_application/app.py

import json
import sys
import os

# This block allows this script to find the sibling 'retrieval_engine' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary functions
from retrieval_engine.vector_store import get_retriever
from llm_application.llm_handler import load_pipeline, generate_decision_json

def process_query(retriever, llm_pipeline, user_query: str) -> dict:
    """Orchestrates the entire RAG process."""
    print("🔎 Retrieving relevant documents...")
    retrieved_docs = retriever.invoke(user_query)
    
    print("\n🧠 Asking the LLM for a decision...")
    final_decision = generate_decision_json(llm_pipeline, user_query, retrieved_docs)
    
    return final_decision

if __name__ == "__main__":
    # Load the tools once at the start
    retriever = get_retriever()
    llm_pipeline = load_pipeline()
    
    # Example query to test the system
    query = "I am a 45-year-old man who needs knee replacement surgery. My policy has been active for 3 years. Am I covered?"
    
    result = process_query(retriever, llm_pipeline, query)
    
    print("\n--- FINAL DECISION ---")
    print(json.dumps(result, indent=2))