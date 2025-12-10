import os
from pathlib import Path
from collections import defaultdict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKS_OUTPUT_DIR


def split_documents(documents):
    """Split documents into smaller chunks and save to files."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📦 Split into {len(chunks)} chunks")
    
    save_chunks_to_files(chunks)
    
    return chunks


def save_chunks_to_files(chunks):
    """Save chunks to organized .txt files by source document."""
    os.makedirs(CHUNKS_OUTPUT_DIR, exist_ok=True)
    
    # Group chunks by source file
    chunks_by_source = defaultdict(list)
    for chunk in chunks:
        source = chunk.metadata.get('source', 'unknown')
        chunks_by_source[source].append(chunk)
    
    print(f"💾 Saving chunks to '{CHUNKS_OUTPUT_DIR}/'...")
    
    for source, source_chunks in chunks_by_source.items():
        source_path = Path(source)
        base_name = source_path.stem
        output_file = os.path.join(CHUNKS_OUTPUT_DIR, f"{base_name}_chunks.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"SOURCE FILE: {source}\n")
            f.write(f"TOTAL CHUNKS: {len(source_chunks)}\n")
            f.write(f"CHUNK SIZE: {CHUNK_SIZE} | OVERLAP: {CHUNK_OVERLAP}\n")
            f.write("=" * 70 + "\n\n")
            
            for i, chunk in enumerate(source_chunks, 1):
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


def clear_chunks_folder():
    """Clear all files in chunks output folder."""
    if os.path.exists(CHUNKS_OUTPUT_DIR):
        for file in os.listdir(CHUNKS_OUTPUT_DIR):
            file_path = os.path.join(CHUNKS_OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"🗑️  Cleared files in '{CHUNKS_OUTPUT_DIR}/'")

