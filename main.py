import os
from src.config import DOCUMENTS_DIR
from src.document_loader import load_documents
from src.chunker import split_documents
from src.vector_store import (
    check_index_has_data, 
    load_vector_store, 
    create_vector_store
)
from src.chat import chat_loop


def main():
    print("🚀 RAG Chatbot (Pinecone) Starting...\n")
    
    if check_index_has_data():
        print("📚 Loading existing vectors from Pinecone...")
        vector_store = load_vector_store()
    else:
        print("📚 First time setup - indexing documents to Pinecone...")
        
        if not os.path.exists(DOCUMENTS_DIR):
            os.makedirs(DOCUMENTS_DIR)
            print(f"📁 Created '{DOCUMENTS_DIR}' folder. Add your documents there and restart!")
            return
        
        if not os.listdir(DOCUMENTS_DIR):
            print(f"⚠️  No documents found in '{DOCUMENTS_DIR}'. Add some files and restart!")
            return
        
        documents = load_documents()
        chunks = split_documents(documents)
        vector_store = create_vector_store(chunks)
    
    chat_loop(vector_store)


if __name__ == "__main__":
    main()

