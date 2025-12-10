import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from src.config import DOCUMENTS_DIR


def load_documents():
    """Load all documents from the documents folder."""
    documents = []
    
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        return documents
    
    files = os.listdir(DOCUMENTS_DIR)
    
    # Load .txt files
    if any(f.endswith('.txt') for f in files):
        txt_loader = DirectoryLoader(
            DOCUMENTS_DIR, 
            glob="**/*.txt", 
            loader_cls=TextLoader
        )
        documents.extend(txt_loader.load())
    
    # Load .md files
    if any(f.endswith('.md') for f in files):
        md_loader = DirectoryLoader(
            DOCUMENTS_DIR, 
            glob="**/*.md", 
            loader_cls=TextLoader
        )
        documents.extend(md_loader.load())
    
    # Load .pdf files
    if any(f.endswith('.pdf') for f in files):
        pdf_loader = DirectoryLoader(
            DOCUMENTS_DIR, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
    
    print(f"📄 Loaded {len(documents)} document(s)")
    return documents

