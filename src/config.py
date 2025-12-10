import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="urllib3")

from dotenv import load_dotenv
load_dotenv()

# === DIRECTORIES ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
CHUNKS_OUTPUT_DIR = os.path.join(BASE_DIR, "chunks_output")

# === CHUNKING ===
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# === PINECONE ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-connect")

# === EMBEDDINGS ===
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# === LLM ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.7

