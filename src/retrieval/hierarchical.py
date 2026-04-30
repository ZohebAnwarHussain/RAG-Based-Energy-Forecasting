"""Pipeline 3 — Hierarchical Retrieval with parent-child expansion.

Retrieves child documents via FAISS dense search, then expands context
by looking up parent documents using the parent_id metadata links built
during KB generation.

Parent-child relationships:
    daily   → parent is the weekly summary for the same zone and ISO week
    weekly  → parent is the monthly summary for the same zone and month
    monthly, seasonal, system_level, appliance, yearly → no parent

How it works:
    1. FAISS retrieves top-k child documents for the query
    2. For each child with a non-empty parent_id, find the parent
       document in the full document list
    3. Deduplicate — if a parent was already retrieved as a child, skip
    4. Return children + parents as expanded context

This provides broader temporal context for multi-scale queries.
For example, a query about "daily demand patterns in January" retrieves
the specific daily summaries AND the weekly/monthly parent summaries
that give the broader seasonal picture.

Since ChromaDB is currently disabled (Windows SQLite issue), parent
lookup is done over the in-memory document list rather than via
ChromaDB metadata filtering. For 480 documents this is instantaneous.

Strengths:
    - Multi-scale context — child detail + parent overview
    - Directly uses parent_id links from KB generation
    - Essential for cross-granularity queries

Weaknesses:
    - More complex than pure dense retrieval
    - Returns more documents (children + parents) — may dilute context
    - Parent quality depends on KB parent_id accuracy
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class HierarchicalRetriever:
    """Hierarchical retrieval with parent document expansion.

    Retrieves child documents via FAISS, then expands context by
    appending parent documents found via parent_id metadata links.

    Args:
        faiss_store: Built or loaded FAISS vector store.
        documents: Full list of KB Document objects. Used for parent
            lookup by scanning metadata. For 480 documents this is
            a fast in-memory operation.
        k: Number of child documents to retrieve. Default 5.
            Total returned documents will be k + number of unique
            parents found (typically k + 2 to k + 4).

    Example:
        >>> retriever = HierarchicalRetriever(faiss_vs, documents, k=5)
        >>> docs = retriever.retrieve("daily demand patterns in January")
        >>> len(docs)  # 5 children + 2-3 parents
        7
    """

    def __init__(
        self,
        faiss_store: FAISS,
        documents: List[Document],
        k: int = 5,
    ) -> None:
        """Initialise hierarchical retriever.

        Args:
            faiss_store: FAISS vector store for child retrieval.
            documents: Full document list for parent lookup.
            k: Number of child documents to retrieve.
        """
        self.faiss_store = faiss_store
        self.k = k

        # Build a lookup dict: row_id → Document for fast parent resolution
        self._doc_lookup: Dict[str, Document] = {}
        for doc in documents:
            row_id = doc.metadata.get("row_id", "")
            if row_id:
                self._doc_lookup[row_id] = doc

        logger.info(
            "HierarchicalRetriever initialised: k=%d, "
            "document lookup size=%d.",
            k, len(self._doc_lookup),
        )

    def _find_parent(self, child_doc: Document) -> Optional[Document]:
        """Look up the parent document for a given child document.

        Uses the parent_id field in the child's metadata to find the
        parent in the pre-built lookup dictionary.

        Args:
            child_doc: A retrieved child Document with metadata
                containing a parent_id field.

        Returns:
            Parent Document if found, None otherwise.
        """
        parent_id = child_doc.metadata.get("parent_id", "")
        if not parent_id:
            return None
        return self._doc_lookup.get(parent_id)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Retrieve child documents and expand with parent context.

        Steps:
            1. FAISS retrieves top-k child documents
            2. For each child with a parent_id, look up the parent
            3. Deduplicate — skip parents already in child results
            4. Return children first, then parents (order matters
               for context window positioning in the RAG prompt)

        Args:
            query: Natural language query string.
            k: Override the default k for child retrieval.

        Returns:
            List of Document objects: children first, then parents.
            Total length is k + number of unique new parents found.
        """
        k = k or self.k

        # Step 1: Retrieve child documents via FAISS
        children = self.faiss_store.similarity_search(query, k=k)

        # Track row_ids already in results to avoid duplicates
        seen_ids = set()
        for doc in children:
            row_id = doc.metadata.get("source", "")
            if row_id:
                seen_ids.add(row_id)

        # Step 2 & 3: Find and deduplicate parents
        parents: List[Document] = []
        for child in children:
            parent = self._find_parent(child)
            if parent is not None:
                parent_id = parent.metadata.get("source", "")
                if parent_id and parent_id not in seen_ids:
                    parents.append(parent)
                    seen_ids.add(parent_id)

        # Step 4: Children first, then parents
        result = children + parents

        logger.info(
            "Hierarchical retrieval: query='%s...' → "
            "%d children + %d parents = %d total.",
            query[:50], len(children), len(parents), len(result),
        )
        return result

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve with metadata formatted for evaluation.

        Children are assigned their FAISS similarity score.
        Parents are assigned score=0.0 (they were not retrieved by
        similarity but by parent_id expansion) and flagged with
        retrieval_method='parent_expansion'.

        Args:
            query: Natural language query string.
            k: Override the default k.

        Returns:
            List of dicts with keys: row_id, score, dataset,
            granularity, retrieval_method, page_content.
        """
        k = k or self.k

        # Get children with scores
        children_with_scores = (
            self.faiss_store.similarity_search_with_score(query, k=k)
        )

        seen_ids = set()
        output: List[Dict[str, Any]] = []

        for doc, score in children_with_scores:
            row_id = doc.metadata.get("source", "")
            seen_ids.add(row_id)
            output.append({
                "row_id":           row_id,
                "score":            float(score),
                "dataset":          doc.metadata.get("dataset", ""),
                "granularity":      doc.metadata.get("granularity", ""),
                "retrieval_method": "dense_child",
                "page_content":     doc.page_content,
            })

        # Find parents
        for doc, _ in children_with_scores:
            parent = self._find_parent(
                Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                )
            )
            if parent is not None:
                parent_id = parent.metadata.get("source", "")
                if parent_id and parent_id not in seen_ids:
                    seen_ids.add(parent_id)
                    output.append({
                        "row_id":           parent_id,
                        "score":            0.0,
                        "dataset":          parent.metadata.get("dataset", ""),
                        "granularity":      parent.metadata.get("granularity", ""),
                        "retrieval_method": "parent_expansion",
                        "page_content":     parent.page_content,
                    })

        return output

    def as_langchain_retriever(self):
        """Return a callable retriever compatible with LCEL chains.

        Since HierarchicalRetriever is not a standard LangChain
        retriever, this returns the base FAISS retriever. For full
        hierarchical context in LCEL chains, use the retrieve() method
        directly and format the results before passing to the chain.

        Returns:
            FAISS retriever (child-only, no parent expansion).
            For full hierarchical context, call self.retrieve() instead.
        """
        logger.warning(
            "as_langchain_retriever() returns FAISS child-only retriever. "
            "For full hierarchical context, use retrieve() directly."
        )
        return self.faiss_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )
