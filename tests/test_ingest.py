import pytest
from langchain_core.documents import Document

from personal_rag_chatbot import ingest


def test_load_documents_loads_txt_and_markdown_files(temp_data_dir):
    documents = ingest.load_documents(temp_data_dir)

    sources = [doc.metadata.get("source", "") for doc in documents]

    assert len(documents) == 2
    assert any("sample.txt" in source for source in sources)
    assert any("notes.md" in source for source in sources)


def test_load_documents_skips_unsupported_files(temp_data_dir, capsys):
    documents = ingest.load_documents(temp_data_dir)

    captured = capsys.readouterr()

    assert len(documents) == 2
    assert "Skipping unsupported file: image.png" in captured.out


def test_load_documents_returns_empty_list_for_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty_data"
    empty_dir.mkdir()

    documents = ingest.load_documents(empty_dir)

    assert documents == []


def test_split_documents_returns_chunks(sample_documents):
    chunks = ingest.split_documents(sample_documents)

    assert len(chunks) >= 1
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all(chunk.page_content for chunk in chunks)


def test_split_documents_preserves_metadata(sample_documents):
    chunks = ingest.split_documents(sample_documents)

    assert all("source" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["source"] == "sample.txt" for chunk in chunks)


def test_ingest_documents_raises_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is missing"):
        ingest.ingest_documents()


def test_ingest_documents_returns_when_no_documents_found(monkeypatch, tmp_path, capsys):
    data_dir = tmp_path / "data"
    chroma_dir = tmp_path / "chroma_db"

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "CHROMA_DIR", chroma_dir)

    ingest.ingest_documents()

    captured = capsys.readouterr()

    assert "No documents found in the data directory." in captured.out
    assert data_dir.exists()
    assert chroma_dir.exists()


def test_ingest_documents_creates_vector_database(monkeypatch, tmp_path, sample_documents):
    data_dir = tmp_path / "data"
    chroma_dir = tmp_path / "chroma_db"

    data_dir.mkdir()
    chroma_dir.mkdir()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(ingest, "load_documents", lambda _: sample_documents)
    monkeypatch.setattr(ingest, "split_documents", lambda docs: docs)

    class FakeEmbeddings:
        def __init__(self, model):
            self.model = model

    class FakeChroma:
        called = False
        received_documents = None
        received_embedding = None
        received_collection_name = None
        received_persist_directory = None

        @classmethod
        def from_documents(
            cls,
            documents,
            embedding,
            collection_name,
            persist_directory,
        ):
            cls.called = True
            cls.received_documents = documents
            cls.received_embedding = embedding
            cls.received_collection_name = collection_name
            cls.received_persist_directory = persist_directory
            return cls()

    monkeypatch.setattr(ingest, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(ingest, "Chroma", FakeChroma)

    ingest.ingest_documents()

    assert FakeChroma.called is True
    assert FakeChroma.received_documents == sample_documents
    assert isinstance(FakeChroma.received_embedding, FakeEmbeddings)
    assert FakeChroma.received_collection_name == ingest.COLLECTION_NAME
    assert FakeChroma.received_persist_directory == str(chroma_dir)


def test_load_documents_handles_pdf_files(monkeypatch, tmp_path, sample_pdf_documents):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    pdf_file = data_dir / "sample.pdf"
    pdf_file.write_bytes(b"%PDF fake content")

    class FakePDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return sample_pdf_documents

    monkeypatch.setattr(ingest, "PyPDFLoader", FakePDFLoader)

    documents = ingest.load_documents(data_dir)

    assert len(documents) == 2
    assert documents[0].metadata["source"] == "sample.pdf"
    assert documents[0].metadata["page"] == 0
