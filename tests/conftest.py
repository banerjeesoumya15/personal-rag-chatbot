import pytest
from langchain_core.documents import Document

from personal_rag_chatbot.config import PROJECT_ROOT

@pytest.fixture
def project_root():
    return PROJECT_ROOT

@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Alex is an AI Engineer who builds RAG chatbots.",
            metadata={"source": "sample.txt"},
        ),
        Document(
            page_content="Alex uses Python, LangChain, ChromaDB, and Streamlit.",
            metadata={"source": "sample.txt"},
        ),
    ]

@pytest.fixture
def sample_pdf_documents():
    return [
        Document(
            page_content="This is page one of a PDF.",
            metadata={"source": "sample.pdf", "page": 0},
        ),
        Document(
            page_content="This is page two of a PDF.",
            metadata={"source": "sample.pdf", "page": 1},
        ),
    ]

@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    sample_file = data_dir / "sample.txt"
    sample_file.write_text(
        "Alex is an AI Engineer.\nHe builds RAG applications.",
        encoding="utf-8",
    )

    markdown_file = data_dir / "notes.md"
    markdown_file.write_text(
        "# Notes\n\nLangChain and ChromaDB are used for retrieval.",
        encoding="utf-8",
    )

    unsupported_file = data_dir / "image.png"
    unsupported_file.write_text("not a real image", encoding="utf-8")

    return data_dir