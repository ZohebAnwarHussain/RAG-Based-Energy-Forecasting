"""Retrieval pipelines (Phase 4).

Three retrieval strategies implemented using LangChain:

    Pipeline 1 — Dense
        Pure semantic similarity via FAISS. Embeds the query using
        all-MiniLM-L6-v2 and returns the k most similar KB summaries.
        Baseline pipeline — no keyword matching or metadata filtering.

    Pipeline 2 — Hybrid
        Combines BM25 sparse retrieval with FAISS dense retrieval via
        LangChain EnsembleRetriever. BM25 catches exact keyword matches
        (zone IDs, Sub_metering column names) that dense search may miss.
        Configurable alpha weight controls sparse/dense balance.
        Post-retrieval metadata filtering available for dataset/granularity.

    Pipeline 3 — Hierarchical
        Retrieves child documents via FAISS, then expands context by
        looking up parent documents using parent_id links. Daily summaries
        retrieve their weekly parent; weekly summaries retrieve their
        monthly parent. Provides broader temporal context for multi-scale
        queries.

All three pipelines return List[Document] with metadata intact,
making them directly compatible with the RAG chain in Phase 5
and RAGAS evaluation in Phase 6.

Usage:
    from src.retrieval import (
        DenseRetriever,
        HybridRetriever,
        HierarchicalRetriever,
    )
"""

from src.retrieval.dense import DenseRetriever
from src.retrieval.hierarchical import HierarchicalRetriever
from src.retrieval.hybrid import HybridRetriever

__all__ = [
    "DenseRetriever",
    "HybridRetriever",
    "HierarchicalRetriever",
]
