# your-llm-project/llm_application/app.py

import os
import sys
import json

# This boilerplate allows the script to find modules in the parent directory (your-llm-project/)
# This is necessary to import from the 'retrieval_engine' and 'config'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval_engine.vector_store import get_retriever
from llm_application.llm_handler import load_model_and_tokenizer, generate_decision_json

def main():
    """
    Main application function to ask a question and get a decision.
    """
    print("--- Initializing the RAG Pipeline ---")
    retriever = get_retriever()
    model, tokenizer = load_model_and_tokenizer()

    query = "I am a 45-year-old man who needs knee replacement surgery. My policy has been active for 3 years. Am I covered?"
    print(f"\n🚀 Processing Query: '{query}'")

    retrieved_docs = retriever.invoke(query)
    print("\n🔎 Retrieved relevant documents.")

    print("\n🧠 Asking the LLM for a decision...")
    final_decision = generate_decision_json(model, tokenizer, query, retrieved_docs)

    print("\n--- ✅ FINAL DECISION ---")
    print(json.dumps(final_decision, indent=2))
    print("------------------------\n")

if __name__ == "__main__":
    main()