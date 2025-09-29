import os
import streamlit as st
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from dotenv import load_dotenv

# ------------------------
# 1. Setup Groq Client
# ------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ------------------------
# 2. Function to build vector store from uploaded docs
# ------------------------
@st.cache_resource
def build_vector_store(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            st.warning(f"⚠️ Unsupported file type: {uploaded_file.name}")
            continue

        docs = loader.load()
        documents.extend(docs)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(documents)

    # Embeddings + Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore

# ------------------------
# 3. Streamlit UI
# ------------------------
st.set_page_config(page_title="Groq RAG Chatbot", page_icon="⚡", layout="wide")
st.title("⚡ Groq-powered Chatbot")

mode = st.radio("Choose Mode:", ["💬 Normal Chat", "📂 Chat with Documents"])

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

# ------------------------
# Normal Chat Mode
# ------------------------
if mode == "💬 Normal Chat":
    st.subheader("💬 Chat with Groq")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=st.session_state["messages"],
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = response.choices[0].message.content
            st.write(bot_reply)

        st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

# ------------------------
# RAG Mode (Chat with Docs)
# ------------------------
elif mode == "📂 Chat with Documents":
    st.subheader("📂 Upload Documents and Chat")

    uploaded_files = st.file_uploader("Upload PDF, TXT, or DOCX", type=["pdf", "txt", "docx"], accept_multiple_files=True)

    if uploaded_files:
        vectorstore = build_vector_store(uploaded_files)

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask something about your documents...")
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join([d.page_content for d in retrieved_docs])

            with st.chat_message("assistant"):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers using the provided context."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_input}"}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                bot_reply = response.choices[0].message.content
                st.write(bot_reply)

            st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
    else:
        st.info("📂 Please upload at least one document to start chatting with it.")


    # 4. Save reply
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})


