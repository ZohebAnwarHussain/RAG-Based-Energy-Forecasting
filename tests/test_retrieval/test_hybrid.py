"""Unit tests for the HybridRetriever class in exp_03_hybrid_rag.py.

Uses a mock FAISS store and verifies RRF fusion, retrieval output
shape, and score computation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unittest.mock import MagicMock
import numpy as np
from langchain_core.documents import Document

from experiments.exp_03_hybrid_rag import HybridRetriever


def _make_mock_faiss(n_docs=6):
    """Build a mock FAISS store with synthetic documents in its docstore."""
    docs = {}
    for i in range(n_docs):
        doc = Document(
            page_content=f"Zone {i+1} load data with {100 + i*10} MW average demand.",
            metadata={"row_id": f"doc_{i}", "dataset": "gefcom", "granularity": "daily"},
        )
        docs[str(i)] = doc

    store = MagicMock()
    store.docstore._dict = docs

    # similarity_search_by_vector returns Documents
    store.similarity_search_by_vector.return_value = list(docs.values())
    return store


def _make_mock_embeddings():
    """Mock embeddings object with embed_query returning a fixed vector."""
    emb = MagicMock()
    emb.embed_query.return_value = [0.1] * 384
    return emb


def test_hybrid_retrieve_returns_list():
    """retrieve must return a list of dicts."""
    store = _make_mock_faiss()
    emb = _make_mock_embeddings()
    retriever = HybridRetriever(store, emb, top_k=3)
    result = retriever.retrieve("peak demand Zone 1")
    assert isinstance(result, list)


def test_hybrid_retrieve_respects_top_k():
    """Number of results must not exceed top_k."""
    store = _make_mock_faiss(n_docs=10)
    emb = _make_mock_embeddings()
    retriever = HybridRetriever(store, emb, top_k=3)
    result = retriever.retrieve("demand")
    assert len(result) <= 3


def test_hybrid_retrieve_has_required_keys():
    """Each result dict must have id, row_id, text, score, metadata."""
    store = _make_mock_faiss()
    emb = _make_mock_embeddings()
    retriever = HybridRetriever(store, emb, top_k=3)
    result = retriever.retrieve("demand")
    for d in result:
        assert "id" in d
        assert "row_id" in d
        assert "text" in d
        assert "score" in d
        assert "metadata" in d


def test_hybrid_retrieve_scores_positive():
    """RRF scores must be positive."""
    store = _make_mock_faiss()
    emb = _make_mock_embeddings()
    retriever = HybridRetriever(store, emb, top_k=3)
    result = retriever.retrieve("demand")
    for d in result:
        assert d["score"] > 0


def test_hybrid_retrieve_as_documents():
    """retrieve_as_documents must return LangChain Document objects."""
    store = _make_mock_faiss()
    emb = _make_mock_embeddings()
    retriever = HybridRetriever(store, emb, top_k=3)
    result = retriever.retrieve_as_documents("demand")
    assert isinstance(result, list)
    for doc in result:
        assert isinstance(doc, Document)


def test_hybrid_rrf_k_constant():
    """RRF_K constant must be 60."""
    assert HybridRetriever.RRF_K == 60


def test_hybrid_bm25_initialized():
    """BM25 index must be initialized from document texts."""
    store = _make_mock_faiss()
    emb = _make_mock_embeddings()
    retriever = HybridRetriever(store, emb, top_k=3)
    assert retriever.bm25 is not None
