import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings  # ← Changed
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="urllib3")

load_dotenv()

# === CONFIG ===
DOCUMENTS_DIR = "./documents"
CHUNKS_OUTPUT_DIR = "./chunks_output"  # ADD THIS LINE
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-connect")

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# === 1. LOAD DOCUMENTS ===
def load_documents():
    """Load all documents from the documents folder."""
    documents = []
    
    # Load .txt files
    if any(f.endswith('.txt') for f in os.listdir(DOCUMENTS_DIR)):
        txt_loader = DirectoryLoader(
            DOCUMENTS_DIR, 
            glob="**/*.txt", 
            loader_cls=TextLoader
        )
        documents.extend(txt_loader.load())
    
    # Load .md files
    if any(f.endswith('.md') for f in os.listdir(DOCUMENTS_DIR)):
        md_loader = DirectoryLoader(
            DOCUMENTS_DIR, 
            glob="**/*.md", 
            loader_cls=TextLoader
        )
        documents.extend(md_loader.load())
    
    # Load .pdf files (requires: pip install pypdf)
    if any(f.endswith('.pdf') for f in os.listdir(DOCUMENTS_DIR)):
        pdf_loader = DirectoryLoader(
            DOCUMENTS_DIR, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
    
    print(f"📄 Loaded {len(documents)} document(s)")
    return documents

# === 2. SPLIT INTO CHUNKS ===
def split_documents(documents):
    """Split documents into smaller chunks and save to files."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📦 Split into {len(chunks)} chunks")
    
    # Save chunks to files
    save_chunks_to_files(chunks)
    
    return chunks

def save_chunks_to_files(chunks):
    """Save chunks to organized .txt files by source document."""
    from pathlib import Path
    from collections import defaultdict
    
    # Create output directory
    os.makedirs(CHUNKS_OUTPUT_DIR, exist_ok=True)
    
    # Group chunks by source file
    chunks_by_source = defaultdict(list)
    for chunk in chunks:
        source = chunk.metadata.get('source', 'unknown')
        chunks_by_source[source].append(chunk)
    
    print(f"💾 Saving chunks to '{CHUNKS_OUTPUT_DIR}/'...")
    
    # Save each group to a separate file
    for source, source_chunks in chunks_by_source.items():
        # Get the original filename without path and extension
        source_path = Path(source)
        base_name = source_path.stem  # filename without extension
        
        # Create output filename
        output_file = os.path.join(CHUNKS_OUTPUT_DIR, f"{base_name}_chunks.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"SOURCE FILE: {source}\n")
            f.write(f"TOTAL CHUNKS: {len(source_chunks)}\n")
            f.write(f"CHUNK SIZE: {CHUNK_SIZE} | OVERLAP: {CHUNK_OVERLAP}\n")
            f.write("=" * 70 + "\n\n")
            
            for i, chunk in enumerate(source_chunks, 1):
                # Clean up the text
                clean_content = " ".join(chunk.page_content.split())
                
                f.write(f"┌{'─' * 68}┐\n")
                f.write(f"│ CHUNK {i} of {len(source_chunks)}\n")
                f.write(f"│ Characters: {len(chunk.page_content)}\n")
                if 'page' in chunk.metadata:
                    f.write(f"│ Page: {chunk.metadata['page']}\n")
                f.write(f"├{'─' * 68}┤\n")
                f.write(f"\n{clean_content}\n\n")
                f.write(f"└{'─' * 68}┘\n\n")
        
        print(f"   📄 Saved: {output_file} ({len(source_chunks)} chunks)")
    
    print(f"✅ All chunks saved to '{CHUNKS_OUTPUT_DIR}/'\n")

# === 3. CREATE/LOAD PINECONE VECTOR STORE ===
def get_embeddings():
    """Get OpenAI embeddings."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}  # BGE recommends this
    )

def create_vector_store(chunks):
    """Create Pinecone vector store from chunks."""
    embeddings = get_embeddings()
    
    # Upload chunks to Pinecone
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
    
    # Clear files in chunks output folder (keep the folder)
    if os.path.exists(CHUNKS_OUTPUT_DIR):
        for file in os.listdir(CHUNKS_OUTPUT_DIR):
            file_path = os.path.join(CHUNKS_OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"🗑️  Cleared files in '{CHUNKS_OUTPUT_DIR}/'")
    
    print("📚 Re-indexing documents...")
    
    if not os.path.exists(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR):
        print(f"⚠️  No documents found in '{DOCUMENTS_DIR}'!")
        return None
    
    documents = load_documents()
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    
    print("✅ Re-indexing complete!\n")
    return vector_store

# === 4. CREATE RAG CHAIN ===
def create_rag_chain(vector_store):
    """Create the conversational RAG chain."""
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    
    return chain

# === 5. MAIN CHATBOT ===
def main():
    print("🚀 RAG Chatbot (Pinecone) Starting...\n")
    
    # Check if Pinecone index has data
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
    
    chain = create_rag_chain(vector_store)
    
    print("\n🤖 Chatbot ready! Ask questions about your documents.")
    print("   Type 'quit' to exit, 'sources' to see last sources used.\n")
    
    last_sources = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break

        if user_input.lower() == 'reindex':
            print("🗑️  Clearing existing vectors from Pinecone...")
            vector_store = clear_and_reindex()
            chain = create_rag_chain(vector_store)
            print("✅ Re-indexing complete!\n")
            continue
        
        if user_input.lower() == 'sources' and last_sources:
            print("\n📖 Sources used:")
            for i, doc in enumerate(last_sources, 1):
                source = doc.metadata.get('source', 'Unknown')
                print(f"  {i}. {source}")
                print(f"     Preview: {doc.page_content[:100]}...")
            print()
            continue
        
        if not user_input:
            continue
        
        result = chain.invoke({"question": user_input})
        answer = result["answer"]
        last_sources = result.get("source_documents", [])

        print("\n" + "=" * 60)
        print("📄 RETRIEVED CHUNKS")
        print("=" * 60)

        for i, doc in enumerate(last_sources):
            # Clean up the text (remove extra whitespace)
            clean_content = " ".join(doc.page_content.split())
            
            print(f"\n┌─ Chunk {i} ─────────────────────────────────────────")
            print(f"│ Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"├─────────────────────────────────────────────────────")
            # Show first 300 chars, nicely wrapped
            preview = clean_content
            print(f"│ {preview}")
            print(f"└─────────────────────────────────────────────────────")
        # print("--- END DEBUG ---\n")
        
        print(f"\nAssistant: {answer}")
        if last_sources:
            print(f"  (Based on {len(last_sources)} source(s) - type 'sources' to see them)")
        print()

if __name__ == "__main__":
    main()