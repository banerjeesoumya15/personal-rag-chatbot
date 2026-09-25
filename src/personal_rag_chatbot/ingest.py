import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from personal_rag_chatbot.config import (
    DATA_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

load_dotenv()

def load_documents(data_dir: Path):
    documents = []

    for file_path in data_dir.iterdir():
        if file_path.is_dir():
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
        elif suffix in [".txt", ".md"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())
        else:
            print(f"Skipping unsupported file: {file_path.name}")

    return documents