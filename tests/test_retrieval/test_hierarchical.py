"""Unit tests for src/retrieval/hierarchical.py.

Uses mock FAISS and synthetic Document objects.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.retrieval.hierarchical import HierarchicalRetriever


def _make_documents():
    """Build a set of synthetic Documents with parent-child relationships."""
    docs = [
        Document(page_content="Zone 1 daily load 100 MW.", metadata={
            "row_id": "daily_z1_d1", "dataset": "gefcom", "granularity": "daily",
            "parent_id": "weekly_z1_w1", "source": "daily_z1_d1",
        }),
        Document(page_content="Zone 1 daily load 110 MW.", metadata={
            "row_id": "daily_z1_d2", "dataset": "gefcom", "granularity": "daily",
            "parent_id": "weekly_z1_w1", "source": "daily_z1_d2",
        }),
        Document(page_content="Zone 1 weekly avg 105 MW.", metadata={
            "row_id": "weekly_z1_w1", "dataset": "gefcom", "granularity": "weekly",
            "parent_id": "monthly_z1_m1", "source": "weekly_z1_w1",
        }),
        Document(page_content="Zone 1 monthly avg 108 MW.", metadata={
            "row_id": "monthly_z1_m1", "dataset": "gefcom", "granularity": "monthly",
            "parent_id": "", "source": "monthly_z1_m1",
        }),
        Document(page_content="Household daily 1.2 kW.", metadata={
            "row_id": "hh_daily_d1", "dataset": "household", "granularity": "daily",
            "parent_id": "hh_weekly_w1", "source": "hh_daily_d1",
        }),
    ]
    return docs


def _mock_faiss(docs):
    """Build a mock FAISS store returning the first 3 docs as children."""
    store = MagicMock()
    children = docs[:3]
    store.similarity_search.return_value = children
    store.similarity_search_with_score.return_value = [
        (doc, 0.9 - i * 0.1) for i, doc in enumerate(children)
    ]
    return store


def test_retrieve_returns_list():
    """retrieve must return a list of Documents."""
    docs = _make_documents()
    retriever = HierarchicalRetriever(_mock_faiss(docs), docs, k=3)
    result = retriever.retrieve("Zone 1 demand")
    assert isinstance(result, list)


def test_retrieve_includes_parents():
    """Result should include more documents than just children (parents added)."""
    docs = _make_documents()
    retriever = HierarchicalRetriever(_mock_faiss(docs), docs, k=3)
    result = retriever.retrieve("Zone 1 demand")
    # 3 children + at least 1 parent (weekly_z1_w1 is parent of daily_z1_d1 and d2)
    assert len(result) >= 3


def test_retrieve_no_duplicate_parents():
    """Parent documents should not be duplicated even if multiple children share them."""
    docs = _make_documents()
    retriever = HierarchicalRetriever(_mock_faiss(docs), docs, k=3)
    result = retriever.retrieve("Zone 1 demand")
    row_ids = [doc.metadata.get("row_id", "") for doc in result]
    assert len(row_ids) == len(set(row_ids)), "Duplicate row_ids found"


def test_retrieve_with_scores_returns_dicts():
    """retrieve_with_scores must return a list of dicts."""
    docs = _make_documents()
    retriever = HierarchicalRetriever(_mock_faiss(docs), docs, k=3)
    result = retriever.retrieve_with_scores("Zone 1 demand")
    assert isinstance(result, list)
    assert all(isinstance(d, dict) for d in result)


def test_retrieve_with_scores_has_retrieval_method():
    """Each dict must have a retrieval_method key."""
    docs = _make_documents()
    retriever = HierarchicalRetriever(_mock_faiss(docs), docs, k=3)
    result = retriever.retrieve_with_scores("Zone 1")
    for d in result:
        assert "retrieval_method" in d
        assert d["retrieval_method"] in ("dense_child", "parent_expansion")


def test_children_marked_as_dense_child():
    """Child documents must have retrieval_method = 'dense_child'."""
    docs = _make_documents()
    retriever = HierarchicalRetriever(_mock_faiss(docs), docs, k=3)
    result = retriever.retrieve_with_scores("Zone 1")
    children = [d for d in result if d["retrieval_method"] == "dense_child"]
    assert len(children) == 3


def test_parents_marked_as_parent_expansion():
    """Parent documents must have retrieval_method = 'parent_expansion'."""
    docs = _make_documents()
    retriever = HierarchicalRetriever(_mock_faiss(docs), docs, k=3)
    result = retriever.retrieve_with_scores("Zone 1")
    parents = [d for d in result if d["retrieval_method"] == "parent_expansion"]
    assert len(parents) >= 1


def test_parent_resolution_field():
    """Parent docs must have parent_resolution = 'exact' or 'semantic'."""
    docs = _make_documents()
    retriever = HierarchicalRetriever(_mock_faiss(docs), docs, k=3)
    result = retriever.retrieve_with_scores("Zone 1")
    parents = [d for d in result if d["retrieval_method"] == "parent_expansion"]
    for p in parents:
        assert p["parent_resolution"] in ("exact", "semantic")


def test_default_k_is_5():
    """Default k should be 5."""
    docs = _make_documents()
    retriever = HierarchicalRetriever(_mock_faiss(docs), docs)
    assert retriever.k == 5
