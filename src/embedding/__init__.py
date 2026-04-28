"""Embedding and vector indexing pipeline (Phase 3).

This package loads KB summaries as LangChain Document objects, embeds
them using a local sentence-transformer model, and stores the results
in two vector databases — FAISS for dense similarity search and
ChromaDB for metadata-filtered retrieval.

This is where LangChain enters the codebase. Phases 1-2 (KB and Golden
Dataset) are pure Python + Gemini API. From this phase onwards, all
retrieval, generation, and evaluation logic uses LangChain interfaces.

Pipeline flow:
    combined_master_summaries.csv
        → CSVLoader → List[Document] with clean page_content
        → HuggingFaceEmbeddings (all-MiniLM-L6-v2, 384-dim, local CPU)
        → FAISS.from_documents() → outputs/indexes/faiss/
        → Chroma.from_documents() → outputs/indexes/chromadb/

No API calls are made in this phase. Everything runs locally.

Usage:
    from src.embedding import (
        load_kb_documents,
        get_embeddings_model,
        build_faiss_index,
        load_faiss_index,
        build_chroma_index,
        load_chroma_index,
    )
"""

#from src.embedding.chroma_store import build_chroma_index, load_chroma_index
from src.embedding.document_loader import load_kb_documents
from src.embedding.embedder import get_embeddings_model
from src.embedding.faiss_store import build_faiss_index, load_faiss_index

__all__ = [
    "load_kb_documents",
    "get_embeddings_model",
    "build_faiss_index",
    "load_faiss_index",
#    "build_chroma_index",
#    "load_chroma_index",
]
