# your-llm-project/retrieval_engine/main.py

import ingest
import vector_store
import config

def main():
    """
    Main function to run the data ingestion and processing pipeline.
    This creates the vector store that the LLM application will use.
    """
    print("--- Starting Data Ingestion and Vector Store Creation ---")
    documents = ingest.load_documents(config.DOCUMENTS_DIR)
    
    if not documents:
        print("No documents found. Please add PDF/DOCX files to the 'documents' folder.")
        return

    chunks = ingest.split_into_chunks(documents)
    vector_store.create_vector_store(chunks)
    print("--- Vector Store is ready. ---")

if __name__ == "__main__":
    main()