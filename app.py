import streamlit as st
import os
import shutil
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import google.generativeai as genai
import openai
import anthropic
from langchain_core.vectorstores import InMemoryVectorStore

#leyout
st.set_page_config(page_title="LLM PDF Chatbot", layout="wide")
st.title("Multi-PDF Chatbot")

GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)


st.sidebar.header("LLM Settings")
model_choice = st.sidebar.selectbox("Choose LLM", ["Gemini","ChatGPT", "Claude"])

openai_api_key = None
anthropic_api_key = None

if model_choice == "ChatGPT":
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
elif model_choice == "Claude":
    anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
 

st.sidebar.markdown("---")
st.sidebar.header(" Controls")
clear_cache = st.sidebar.button(" Clear Cache & History")
force_refresh = st.sidebar.button("Force Rerun")
store_choice = st.sidebar.radio("Vector Store Type", ["FAISS (persistent)", "In-Memory (temporary)"])
max_files = st.sidebar.slider("PDF Upload Limit", min_value=1, max_value=10, value=5)


db_path = "faiss_index"
if clear_cache:
    st.session_state.clear()
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    st.success("Cleared cache and FAISS index.")

if force_refresh:
    st.rerun()


uploaded_files = st.file_uploader("Upload up to {} PDFs".format(max_files), type="pdf", accept_multiple_files=True)

raw_texts = []
if uploaded_files:
    if len(uploaded_files) > max_files:
        st.warning(f"Upload limit exceeded. Max allowed: {max_files}")
        uploaded_files = uploaded_files[:max_files]

    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
        raw_texts.append(text)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if st.button("Generate Vector Store"):
    if not raw_texts:
        st.warning("Upload PDFs first.")
    else:
        docs = [Document(page_content=txt) for txt in raw_texts]

        if store_choice == "FAISS (persistent)":
            faiss_store = FAISS.from_documents(docs, embeddings)
            faiss_store.save_local(db_path)
            st.session_state.vectorstore = faiss_store
            st.success(" FAISS Vector Store created and saved.")
        else:
            memory_store = InMemoryVectorStore.from_documents(docs, embedding=embeddings)
            st.session_state.vectorstore = memory_store
            st.success(" In-Memory Vector Store created.")

# Load FAISS (dep)

if store_choice == "FAISS (persistent)" and os.path.exists(db_path) and "vectorstore" not in st.session_state:
    st.session_state.vectorstore = FAISS.load_local(db_path, embeddings)

# chat ui
st.subheader(" Ask Questions About Your PDFs")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask something:")

# oii chat history
for entry in st.session_state.chat_history:
    st.chat_message("user").write(entry["user"])
    st.chat_message("assistant").write(entry["bot"])

# LLM Response 
def query_llm(question, context, model="Gemni"):
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"

    if model == "ChatGPT":
        if not openai_api_key:
            return " OpenAI API Key missing."
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']

    elif model == "Claude":
        if not anthropic_api_key:
            return " Claude API Key missing."
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=500,
            temperature=0.6,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    elif model == "Gemini":
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text

    return " Invalid Model Selected"

# Ask get ans
if st.button(" Get Answer"):
    if query:
        if "vectorstore" not in st.session_state:
            st.warning(" No vectorstore generated or loaded.")
        else:
            docs = st.session_state.vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            with st.spinner("Thinking..."):
                answer = query_llm(query, context, model=model_choice)
            st.session_state.chat_history.append({"user": query, "bot": answer})
            st.chat_message("user").write(query)
            st.chat_message("assistant").write(answer)
    else:
        st.warning("Enter a question to get an answer.")
