import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from personal_rag_chatbot.ingest import ingest_documents
from personal_rag_chatbot.rag import ask_question
from personal_rag_chatbot.config import DATA_DIR, CHROMA_DIR


load_dotenv()


st.set_page_config(
    page_title="Personal RAG Chatbot",
    layout="wide",
)


st.title("Personal RAG Chatbot")
st.write("Upload documents, build a vector database, and ask questions from your files.")


with st.sidebar:
    st.header("Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Save Uploaded Files"):
        DATA_DIR.mkdir(exist_ok=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = DATA_DIR / uploaded_file.name

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            st.success(f"Saved {len(uploaded_files)} file(s) to the data folder.")
        else:
            st.warning("Please upload at least one file.")

    if st.button("Build Vector Database"):
        try:
            ingest_documents()
            st.success("Vector database built successfully.")
        except Exception as e:
            st.error(f"Error while building vector database: {e}")

    if st.button("Clear Documents and Database"):
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)

        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)

        DATA_DIR.mkdir(exist_ok=True)
        st.success("Documents and vector database cleared.")

    st.divider()

    st.subheader("Status")

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        st.success("OpenAI API key loaded.")
    else:
        st.error("OpenAI API key missing.")

    if CHROMA_DIR.exists():
        st.success("Vector database exists.")
    else:
        st.warning("Vector database not found.")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a question about your documents...")


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = ask_question(question)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                with st.expander("Sources"):
                    for i, doc in enumerate(sources, start=1):
                        source = doc.metadata.get("source", "Unknown source")
                        page = doc.metadata.get("page", None)

                        if page is not None:
                            st.write(f"**Source {i}:** `{source}`, page {page + 1}")
                        else:
                            st.write(f"**Source {i}:** `{source}`")

                        preview = doc.page_content[:500]
                        st.write(preview + "...")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as e:
                error_message = f"Error: {e}"
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
