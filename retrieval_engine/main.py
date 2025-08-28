# main.py

import ingest

# Define the path to the directory containing the documents
# This assumes your 'documents' folder is in the same directory as main.py
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
    
    # We will add the next steps (embedding and indexing) here later.
    print("\n--- Example of a Chunk ---")
    print(chunks[1].page_content)
    print("\n--- Metadata of the Chunk ---")
    print(chunks[1].metadata)
    
    print("\nProcess completed.")

if __name__ == "__main__":
    main()