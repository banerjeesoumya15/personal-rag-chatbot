from pathlib import Path

from personal_rag_chatbot import config

def test_project_root_is_path():
    assert isinstance(config.PROJECT_ROOT, Path)
    assert config.PROJECT_ROOT.exists()

def test_data_dir_is_under_project_root():
    assert config.DATA_DIR == config.PROJECT_ROOT / "data"

def test_chroma_dir_is_under_project_root():
    assert config.CHROMA_DIR == config.PROJECT_ROOT / "chroma_db"

def test_ollama_base_url_is_not_empty():
    assert config.OLLAMA_BASE_URL

def test_model_config_values_Are_not_empty():
    assert config.COLLECTION_NAME
    assert config.EMBEDDING_MODEL
    assert config.CHAT_MODEL

def test_chunking_config_values_Are_valid():
    assert config.CHUNK_SIZE > 0
    assert config.CHUNK_OVERLAP > 0
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

def test_retrieval_k_is_positive():
    assert config.RETRIEVAL_K > 0