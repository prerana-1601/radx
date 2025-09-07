# rag/retriever.py

import os
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()


def get_retriever(k: int = 5):
    """
    Load the persisted Chroma vectorstore and return a retriever.
    Args:
        k (int): Number of top documents to retrieve per query
    Returns:
        retriever object
    """
    persist_directory = "rag/db"
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Load Chroma vectorstore
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever
