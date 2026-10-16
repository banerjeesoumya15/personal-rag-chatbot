import pytest
from langchain_core.documents import Document

from personal_rag_chatbot import rag


def test_format_docs_with_text_sources(sample_documents):
    formatted = rag.format_docs(sample_documents)

    assert "[Source 1: sample.txt]" in formatted
    assert "[Source 2: sample.txt]" in formatted
    assert "Alex is an AI Engineer" in formatted
    assert "Python, LangChain, ChromaDB" in formatted


def test_format_docs_with_pdf_page_numbers(sample_pdf_documents):
    formatted = rag.format_docs(sample_pdf_documents)

    assert "[Source 1: sample.pdf, page 1]" in formatted
    assert "[Source 2: sample.pdf, page 2]" in formatted
    assert "This is page one of a PDF." in formatted
    assert "This is page two of a PDF." in formatted


def test_format_docs_handles_missing_metadata():
    docs = [
        Document(
            page_content="Document with no metadata.",
            metadata={},
        )
    ]

    formatted = rag.format_docs(docs)

    assert "[Source 1: Unknown source]" in formatted
    assert "Document with no metadata." in formatted


def test_format_docs_handles_empty_list():
    formatted = rag.format_docs([])

    assert formatted == ""


def test_get_vectorstore_creates_chroma_instance(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "CHROMA_DIR", tmp_path / "chroma_db")

    class FakeEmbeddings:
        def __init__(self, model, base_url):
            self.model = model
            self.base_url = base_url

    class FakeChroma:
        def __init__(
            self,
            collection_name,
            embedding_function,
            persist_directory,
        ):
            self.collection_name = collection_name
            self.embedding_function = embedding_function
            self.persist_directory = persist_directory

    monkeypatch.setattr(rag, "OllamaEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(rag, "Chroma", FakeChroma)

    vectorstore = rag.get_vectorstore()

    assert isinstance(vectorstore, FakeChroma)
    assert vectorstore.collection_name == rag.COLLECTION_NAME
    assert isinstance(vectorstore.embedding_function, FakeEmbeddings)
    assert vectorstore.embedding_function.model == rag.EMBEDDING_MODEL
    assert vectorstore.embedding_function.base_url == rag.OLLAMA_BASE_URL
    assert vectorstore.persist_directory == str(tmp_path / "chroma_db")


def test_ask_question_returns_answer_and_sources(monkeypatch, sample_documents):

    class FakeRetriever:
        def invoke(self, question):
            assert question == "What does Alex build?"
            return sample_documents

    class FakeVectorStore:
        def as_retriever(self, search_kwargs):
            assert search_kwargs == {"k": rag.RETRIEVAL_K}
            return FakeRetriever()

    class FakePrompt:
        def __or__(self, other):
            return FakeChainStep()

    class FakeChainStep:
        def __or__(self, other):
            return FakeChain()

    class FakeChain:
        def invoke(self, inputs):
            assert inputs["question"] == "What does Alex build?"
            assert "Alex is an AI Engineer" in inputs["context"]
            return "Alex builds RAG chatbots."

    class FakeChatPromptTemplate:
        @staticmethod
        def from_messages(messages):
            assert len(messages) == 2
            assert messages[0][0] == "system"
            assert messages[1][0] == "user"
            return FakePrompt()

    class FakeChatOllama:
        def __init__(self, model, base_url, temperature):
            assert model == rag.CHAT_MODEL
            assert base_url == rag.OLLAMA_BASE_URL
            assert temperature == 0

    class FakeStrOutputParser:
        pass

    monkeypatch.setattr(rag, "get_vectorstore", lambda: FakeVectorStore())
    monkeypatch.setattr(rag, "ChatPromptTemplate", FakeChatPromptTemplate)
    monkeypatch.setattr(rag, "ChatOllama", FakeChatOllama)
    monkeypatch.setattr(rag, "StrOutputParser", FakeStrOutputParser)

    result = rag.ask_question("What does Alex build?")

    assert result["answer"] == "Alex builds RAG chatbots."
    assert result["sources"] == sample_documents


def test_ask_question_handles_no_retrieved_documents(monkeypatch):

    class FakeRetriever:
        def invoke(self, question):
            return []

    class FakeVectorStore:
        def as_retriever(self, search_kwargs):
            return FakeRetriever()

    class FakePrompt:
        def __or__(self, other):
            return FakeChainStep()

    class FakeChainStep:
        def __or__(self, other):
            return FakeChain()

    class FakeChain:
        def invoke(self, inputs):
            assert inputs["context"] == ""
            return "I do not know based on the provided documents."

    class FakeChatPromptTemplate:
        @staticmethod
        def from_messages(messages):
            return FakePrompt()

    class FakeChatOllama:
        def __init__(self, model, base_url, temperature):
            pass

    class FakeStrOutputParser:
        pass

    monkeypatch.setattr(rag, "get_vectorstore", lambda: FakeVectorStore())
    monkeypatch.setattr(rag, "ChatPromptTemplate", FakeChatPromptTemplate)
    monkeypatch.setattr(rag, "ChatOllama", FakeChatOllama)
    monkeypatch.setattr(rag, "StrOutputParser", FakeStrOutputParser)

    result = rag.ask_question("What is not in the docs?")

    assert result["answer"] == "I do not know based on the provided documents."
    assert result["sources"] == []
