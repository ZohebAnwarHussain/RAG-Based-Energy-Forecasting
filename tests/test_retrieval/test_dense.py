"""Unit tests for src/retrieval/dense.py.

Uses a mock FAISS vector store to avoid loading real embeddings.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.retrieval.dense import DenseRetriever


def _mock_faiss(n_docs=5):
    """Build a mock FAISS store that returns synthetic documents."""
    docs = [
        Document(
            page_content=f"Zone {i} had {100 + i*10} MW load.",
            metadata={"row_id": f"doc_{i}", "source": f"doc_{i}", "dataset": "gefcom", "granularity": "daily"},
        )
        for i in range(n_docs)
    ]
    store = MagicMock()
    store.similarity_search.return_value = docs
    store.similarity_search_with_score.return_value = [(doc, 0.9 - i * 0.1) for i, doc in enumerate(docs)]
    return store


def test_retrieve_returns_list():
    """retrieve must return a list of Documents."""
    retriever = DenseRetriever(_mock_faiss(), k=3)
    result = retriever.retrieve("peak demand")
    assert isinstance(result, list)


def test_retrieve_returns_documents():
    """Each item must be a LangChain Document."""
    retriever = DenseRetriever(_mock_faiss(), k=3)
    result = retriever.retrieve("peak demand")
    for doc in result:
        assert isinstance(doc, Document)


def test_retrieve_passes_k_to_faiss():
    """retrieve must pass the k parameter to similarity_search."""
    store = _mock_faiss()
    retriever = DenseRetriever(store, k=7)
    retriever.retrieve("test query")
    store.similarity_search.assert_called_once_with("test query", k=7)


def test_retrieve_k_override():
    """k parameter in retrieve() overrides the instance default."""
    store = _mock_faiss()
    retriever = DenseRetriever(store, k=5)
    retriever.retrieve("test", k=3)
    store.similarity_search.assert_called_once_with("test", k=3)


def test_retrieve_with_scores_returns_list():
    """retrieve_with_scores must return a list of dicts."""
    retriever = DenseRetriever(_mock_faiss(), k=3)
    result = retriever.retrieve_with_scores("peak demand")
    assert isinstance(result, list)
    assert all(isinstance(d, dict) for d in result)


def test_retrieve_with_scores_has_required_keys():
    """Each dict must have row_id, score, page_content."""
    retriever = DenseRetriever(_mock_faiss(), k=3)
    result = retriever.retrieve_with_scores("test")
    for d in result:
        assert "row_id" in d
        assert "score" in d
        assert "page_content" in d


def test_retrieve_with_scores_score_is_float():
    """Score values must be floats."""
    retriever = DenseRetriever(_mock_faiss(), k=3)
    result = retriever.retrieve_with_scores("test")
    for d in result:
        assert isinstance(d["score"], float)


def test_default_k_is_5():
    """Default k should be 5."""
    retriever = DenseRetriever(_mock_faiss())
    assert retriever.k == 5
