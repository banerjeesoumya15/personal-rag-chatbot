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

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)

def ingest_documents():
    if not os.getenv("OPENAP_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing.")

    DATA_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)

    print("Lodaing documents...")
    documents = load_documents(DATA_DIR)

    if not documents:
        print("No documents found in the data directory")
        return

    print(f"Loaded {len(documents)} documents")

    print("Splitting documents...")
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    print("Creating vector database...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print("Ingestion complete.")
    print(f"Vector database saved to: {CHROMA_DIR}")

if __name__=="__main__":
    ingest_documents()