# main.py

import ingest
import vector_store

# Define the path to the directory containing the documents
DOCUMENTS_DIR = "documents"

def main():
    """
    Main function to run the data ingestion and processing pipeline.
    """
    print("Starting the data ingestion process...")
    
    # Step 1: Load the documents from the specified directory
    documents = ingest.load_documents(DOCUMENTS_DIR)
    
    # Step 2: Split the documents into chunks
    chunks = ingest.split_into_chunks(documents)
    
    # Step 3: Create the FAISS vector store from the chunks
    # This will create embeddings and save the index to a local directory.
    vector_store.create_vector_store(chunks)
    
    print("\nProcess completed. The vector store is ready.")

if __name__ == "__main__":
    main()