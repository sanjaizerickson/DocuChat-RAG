import streamlit as st
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # for PDF
import pandas as pd
import os

# --- Embedding & LLM ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Ollama(model="gemma3:4b", temperature=0.3)

# --- Memory ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Classifier Chain ---
classifier_prompt = PromptTemplate.from_template("""
You are a topic classifier.
Classify the user's query into one of the following categories:
- deforestation
- population
- ww2
- payslip
- general_chat

Respond with only the label.

Query: {query}
Label:
""")
classifier_chain = LLMChain(llm=llm, prompt=classifier_prompt)

# --- Helper: Ingest PDF/CSV and add to FAISS ---
def ingest_file(file, label):
    text = ""
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        text = "\n".join(df.astype(str).apply(lambda x: " ".join(x), axis=1))
    else:
        st.error("Unsupported file type.")
        return
    # Chunk and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(f"vectorstores/{label}_faiss")
    st.success(f"{label.capitalize()} vectorstore updated!")

# --- Main Chatbot Logic ---
def hybrid_qa_pipeline(user_query):
    # Step 1: Classify
    label = classifier_chain.run({"query": user_query}).strip().lower()
    # Step 2: Route
    if label in ["deforestation", "population", "ww2", "payslip"]:
        retriever = FAISS.load_local(
            f"vectorstores/{label}_faiss",
            embedding_model,
            allow_dangerous_deserialization=True
        ).as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory
        )
        response = qa_chain.invoke({"query": user_query, "chat_history": st.session_state.memory.buffer})
        return response['result']
    elif label == "general_chat":
        response = llm.predict(f"You are a helpful assistant. Answer this: {user_query}")
        return response
    else:
        return "Sorry, I couldn't classify your query."

# --- Streamlit UI ---
st.title("Conversational RAG Chatbot (Ollama)")
st.write("Upload your PDF/CSV, then ask questions. The bot remembers your conversation!")

# File upload
uploaded_file = st.file_uploader("Upload PDF or CSV", type=["pdf", "csv"])
label = st.selectbox("Select topic label for this file", ["deforestation", "population", "ww2", "payslip"])
if uploaded_file and st.button("Ingest File"):
    ingest_file(uploaded_file, label)

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="user_input")
if user_input:
    response = hybrid_qa_pipeline(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {msg}")

