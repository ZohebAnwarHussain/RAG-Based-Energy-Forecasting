"""LJMU Thesis — RAG-based Energy Forecasting source package.

This package contains all reusable Python modules for the thesis pipeline.
Modules are organised by pipeline stage:

    src.utils              Logging, timestamps, IO helpers (used everywhere)
    src.knowledge_base     Phase 1 — KB generation pipeline
    src.golden_dataset     Phase 2 — Golden dataset generation
    src.embedding          Phase 3 — LangChain document loading and indexing
    src.retrieval          Phase 4 — Three retrieval pipelines
    src.rag                Phase 5 — LCEL chains and RAG generation
    src.evaluation         Phase 6 — RAGAS metrics and hallucination checks

Import functions directly from each subpackage:

    from src.knowledge_base import generate_summaries
    from src.golden_dataset import generate_golden_dataset
    from src.embedding import build_chroma_index
    from src.retrieval import DenseRetriever
    from src.rag import build_rag_chain
    from src.evaluation import compute_recall_at_k
"""

__version__ = "0.1.0"