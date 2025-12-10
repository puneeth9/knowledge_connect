import os
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.config import PINECONE_API_KEY, INDEX_NAME, EMBEDDING_MODEL, DOCUMENTS_DIR
from src.document_loader import load_documents
from src.chunker import split_documents, clear_chunks_folder

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


def get_embeddings():
    """Get HuggingFace BGE embeddings."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def create_vector_store(chunks):
    """Create Pinecone vector store from chunks."""
    embeddings = get_embeddings()
    
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    
    print(f"✅ Uploaded {len(chunks)} chunks to Pinecone")
    return vector_store


def load_vector_store():
    """Load existing Pinecone vector store."""
    embeddings = get_embeddings()
    
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    
    return vector_store


def check_index_has_data():
    """Check if Pinecone index has vectors."""
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    return stats.total_vector_count > 0


def clear_and_reindex():
    """Clear Pinecone index and re-index all documents."""
    print("\n🗑️  Clearing existing vectors from Pinecone...")
    
    try:
        index = pc.Index(INDEX_NAME)
        index.delete(delete_all=True)
        print("✅ Cleared all vectors")
    except Exception as e:
        print(f"⚠️  Error clearing index: {e}")
    
    clear_chunks_folder()
    
    print("📚 Re-indexing documents...")
    
    if not os.path.exists(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR):
        print(f"⚠️  No documents found in '{DOCUMENTS_DIR}'!")
        return None
    
    documents = load_documents()
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    
    print("✅ Re-indexing complete!\n")
    return vector_store

