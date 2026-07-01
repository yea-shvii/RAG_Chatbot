import os
import time
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env")
    st.stop()

# ----------------------------
# Streamlit Configuration
# ----------------------------
st.set_page_config(
    page_title="Groq RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 ChatBot")

# ----------------------------
# Build Vector Store (Only Once)
# ----------------------------
if "vectors" not in st.session_state:

    with st.spinner("Loading documents..."):

        loader = WebBaseLoader(
            "https://docs.smith.langchain.com/"
        )

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        documents = splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        vectors = FAISS.from_documents(
            documents,
            embeddings
        )

        st.session_state.vectors = vectors

# ----------------------------
# Initialize Groq LLM
# ----------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# ----------------------------
# Prompt Template
# ----------------------------
prompt = ChatPromptTemplate.from_template(
    """
Answer the question ONLY from the provided context.

<context>
{context}
</context>

Question:
{input}

If the answer is not available in the context,
reply:
"I don't know based on the provided context."
"""
)

document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

retriever = st.session_state.vectors.as_retriever()

retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)

# ----------------------------
# User Input
# ----------------------------
user_question = st.text_input(
    "Ask a question about LangSmith"
)

if user_question:

    start = time.time()

    response = retrieval_chain.invoke(
        {
            "input": user_question
        }
    )

    end = time.time()

    st.subheader("Answer")

    st.write(response["answer"])

    st.success(
        f"Response Time: {end-start:.2f} seconds"
    )