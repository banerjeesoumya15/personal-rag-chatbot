from pathlib import Path

# Points to project root personal-rag-chatbot/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# define directories
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "personal_rag_chatbot"

EMBEDDING_MODEL = ""
CHAT_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4