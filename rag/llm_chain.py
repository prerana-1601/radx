# rag/llm_chain.py

import os
from typing import Any, Dict, Union

from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

from rag.retriever import get_retriever

# Load environment variables from .env (once)
load_dotenv()

def _require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Put it in your .env or export it in the shell."
        )
    return api_key

def _extract_text(response: Union[str, Dict[str, Any]]) -> str:
    """Be tolerant to different LangChain return shapes."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for k in ("result", "output_text", "text"):
            v = response.get(k)
            if isinstance(v, str):
                return v
    # Fallback
    return str(response)

def get_qa_chain(temperature: float = 0.0, k: int = 5) -> RetrievalQA:
    """
    Create a RetrievalQA chain backed by Chroma retriever and OpenAI LLM.
    """
    api_key = _require_api_key()
    retriever = get_retriever(k=k)

    llm = OpenAI(
        temperature=temperature,
        openai_api_key=api_key,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,  # set True if you want sources
    )
    return qa_chain

def ask_qa(question: str, qa_chain: RetrievalQA) -> str:
    """
    Ask a natural-language question and return the LLM's plain-text answer.
    """
    # RetrievalQA expects {"query": "..."} with invoke()
    resp = qa_chain.invoke({"query": question})
    return _extract_text(resp)

def generate_radiology_report(gradcam_summary: str, qa_chain: RetrievalQA) -> str:
    """
    Generate a radiology-style report by querying the QA chain with a prompt that
    includes the Grad-CAM/image summary.
    """
    prompt = (
        "Using authoritative radiology guidance, generate a concise, structured "
        "report with sections: Findings, Impression, and Recommendations. "
        "Context:\n"
        f"{gradcam_summary}\n\n"
        "Be specific but avoid overclaiming beyond the evidence."
    )
    resp = qa_chain.invoke({"query": prompt})
    return _extract_text(resp)

