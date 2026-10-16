import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Points to project root personal-rag-chatbot/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# define directories
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "personal_rag_chatbot"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4