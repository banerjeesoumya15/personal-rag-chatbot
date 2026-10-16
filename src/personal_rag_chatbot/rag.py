from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from personal_rag_chatbot.config import (
    DATA_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    OLLAMA_BASE_URL,
    RETRIEVAL_K,
)

SYSTEM_PROMPT = """
You are a helpful AI assistant answering questions using the provided context.

Rulse:
- Use only the context to answer.
- If the answer is not in the context, say: "I do not know based on the provided documents."
- Be consise and accurate.
- Cite the source documents when possible.
"""

USER_PROMPT = """
Question:
{question}

Context:
{context}

Answer:
"""

def get_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL,
                                  base_url=OLLAMA_BASE_URL)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    return vectorstore

def format_docs(docs):
    formatted_chunks = []

    for i,doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", None)

        if page is not None:
            source_info = f"{source}, page {page + 1}"
        else:
            source_info = source

        formatted_chunks.append(
            f"[Source {i}: {source_info}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted_chunks)

def ask_questions(question: str):
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )

    docs = retriever.invoke(question)
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT),
        ]
    )

    llm = ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return {
        "answer": answer,
        "sources": docs,
    }