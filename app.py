# rag_chatbot.py
import os
import streamlit as st
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# ------------------------
# 1. Setup Groq Client
# ------------------------
client = Groq(api_key=os.getenv("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"))

# ------------------------
# 2. Load & Embed Documents
# ------------------------
@st.cache_resource
def load_vector_store():
    # Example: Replace with your own docs
    texts = [
        "Groq provides ultra-fast inference for LLMs.",
        "Streamlit helps build interactive AI apps easily.",
        "Retrieval-Augmented Generation improves factual accuracy."
    ]

    docs = [Document(page_content=t) for t in texts]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore

vectorstore = load_vector_store()

# ------------------------
# 3. Streamlit UI
# ------------------------
st.set_page_config(page_title="Groq RAG Chatbot", page_icon="⚡", layout="wide")
st.title("⚡ Groq-powered RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful AI assistant that answers using the given context."}
    ]

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # 1. Add user query
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # 2. Retrieve relevant docs
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([d.page_content for d in retrieved_docs])

    # 3. Ask Groq with context
    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # other Groq models: mixtral-8x7b, llama-3.1-70b
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers based on context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_input}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        bot_reply = response.choices[0].message.content
        st.write(bot_reply)

    # 4. Save reply
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
