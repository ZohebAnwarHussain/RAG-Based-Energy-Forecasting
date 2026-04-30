"""RAG generation pipeline (Phase 5).

Generates natural language energy demand insights by combining retrieved
KB context with Llama 3.3 70B via Groq. This is where the Groq API
enters the codebase.

Components:
    get_rag_llm()       Returns a configured ChatGroq instance
    RAG_PROMPT          ChatPromptTemplate for insight generation
    build_rag_chain()   Constructs an LCEL chain: retriever | prompt | llm
    generate_rag_answers()  Runs all golden queries through the chain

The LCEL (LangChain Expression Language) chain pattern:
    retriever → format retrieved docs → inject into prompt → LLM → parse output

Usage:
    from src.rag import get_rag_llm, build_rag_chain, generate_rag_answers
"""

from src.rag.chains import build_rag_chain, generate_rag_answers
from src.rag.llm import get_rag_llm
from src.rag.prompts import RAG_PROMPT

__all__ = [
    "get_rag_llm",
    "RAG_PROMPT",
    "build_rag_chain",
    "generate_rag_answers",
]
