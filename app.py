import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Title and subtitle
st.title("Intertek Sustainability Disclosure")
st.markdown("**Based on 2024 Intertek Reports**")

# Sidebar for OpenAI API Key
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Topic buttons (suggested based on PDFs)
topic = st.radio("Explore Topics:", [
    "1. Climate & Emissions Goals",
    "2. People & Culture Initiatives",
    "3. Governance & Compliance",
    "4. Community Engagement",
    "5. Total Sustainability Assurance (TSA) Framework"
])

# File uploader
uploaded_files = st.file_uploader("Upload Intertek 2024 Reports (PDF)", type="pdf", accept_multiple_files=True)

# Question input
query = st.text_input("Ask a question about Intertek's sustainability activities")

if openai_api_key and uploaded_files and query:
    st.info("Processing your question...")

    # Load and split documents
    docs = []
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    # Create vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Build RetrievalQA chain
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Run query
    answer = qa_chain.run(query)
    st.success("Answer:")
    st.write(answer)

    st.markdown("---")
    st.caption("Powered by LangChain + OpenAI + FAISS")
else:
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key on the left.")
    if not uploaded_files:
        st.warning("Please upload at least one Intertek report PDF.")
    if not query:
        st.info("Enter a question about Intertek's sustainability disclosures.")
