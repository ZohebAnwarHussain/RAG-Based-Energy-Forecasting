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
    2. For each child, attempt exact parent_id lookup in the doc store
    3. If exact lookup fails (parent chunk not in KB sample), fall back
       to semantic parent retrieval: find the closest chunk of the
       next-higher granularity for the same dataset via FAISS search
    4. Deduplicate — if a parent was already retrieved as a child, skip
    5. Return children + parents as expanded context

IMPORTANT — KB SAMPLE LIMITATION
----------------------------------
The combined_master_summaries.csv KB is a sample. The specific weekly/
monthly chunks selected may not be the same time periods referenced by
the daily chunks' parent_id fields. In practice, 0/14 GEFCom daily
chunks and 0/5 household daily chunks can resolve their parent via
exact parent_id lookup.

The semantic fallback addresses this by finding the closest available
chunk at the next granularity level, giving EXP_04/07/09 meaningful
broader context even without exact parent_id matches.

Granularity hierarchy used for semantic fallback:
    daily      → weekly
    weekly     → monthly
    monthly    → seasonal
    seasonal   → system_level
    appliance  → monthly
    system_level, yearly → no parent (leaf nodes)

Since ChromaDB is currently disabled (Windows SQLite issue), parent
lookup is done over the in-memory document list rather than via
ChromaDB metadata filtering. For 140 documents this is instantaneous.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Granularity hierarchy for semantic parent fallback
# ---------------------------------------------------------------------------

_PARENT_GRANULARITY: Dict[str, Optional[str]] = {
    "daily":        "weekly",
    "weekly":       "monthly",
    "monthly":      "seasonal",
    "seasonal":     "system_level",
    "appliance":    "monthly",
    "system_level": None,
    "yearly":       None,
}


class HierarchicalRetriever:
    """Hierarchical retrieval with parent document expansion.

    Retrieves child documents via FAISS, then expands context by
    appending parent documents. Parent resolution uses two strategies:

    1. Exact lookup: child metadata parent_id → doc_lookup[parent_id]
    2. Semantic fallback: if exact lookup fails, find the closest chunk
       of the next-higher granularity (same dataset) via FAISS search.
       This handles the case where the KB sample does not contain the
       specific parent chunk referenced by parent_id.

    Args:
        faiss_store: Built or loaded FAISS vector store.
        documents: Full list of KB Document objects. Used for parent
            lookup by scanning metadata. For 140 documents this is
            a fast in-memory operation.
        k: Number of child documents to retrieve. Default 5.

    Example:
        >>> retriever = HierarchicalRetriever(faiss_vs, documents, k=5)
        >>> docs = retriever.retrieve("daily demand patterns in January")
        >>> len(docs)  # 5 children + parents (exact or semantic)
        7
    """

    def __init__(
        self,
        faiss_store: FAISS,
        documents: List[Document],
        k: int = 5,
    ) -> None:
        self.faiss_store = faiss_store
        self.k = k

        # Build lookup dicts for fast parent resolution
        # Primary: row_id → Document (for exact parent_id lookup)
        self._doc_lookup: Dict[str, Document] = {}
        # Secondary: (dataset, granularity) → [Documents] (for semantic fallback)
        self._gran_lookup: Dict[tuple, List[Document]] = {}

        for doc in documents:
            row_id = doc.metadata.get("row_id", "")
            if row_id:
                self._doc_lookup[row_id] = doc

            dataset = doc.metadata.get("dataset", "")
            gran    = doc.metadata.get("granularity", "")
            key     = (dataset, gran)
            self._gran_lookup.setdefault(key, []).append(doc)

        logger.info(
            "HierarchicalRetriever initialised: k=%d, "
            "doc_lookup size=%d, gran_lookup keys=%s.",
            k, len(self._doc_lookup),
            sorted(self._gran_lookup.keys()),
        )

    # -------------------------------------------------------------------------
    # Parent resolution
    # -------------------------------------------------------------------------

    def _find_parent_exact(self, child_doc: Document) -> Optional[Document]:
        """Exact parent_id lookup in the doc store."""
        parent_id = child_doc.metadata.get("parent_id", "")
        if not parent_id:
            return None
        return self._doc_lookup.get(parent_id)

    def _find_parent_semantic(
        self, child_doc: Document, query: str
    ) -> Optional[Document]:
        """
        Semantic fallback: find the closest chunk of the next-higher
        granularity for the same dataset via FAISS similarity search.

        Called only when exact parent_id lookup fails. This ensures
        EXP_04/07/09 always receive some broader-context parent document
        even when the KB sample does not include the exact parent week/month
        referenced by the child's parent_id field.

        Returns the best semantic match at the parent granularity,
        or None if no candidates exist for that granularity.
        """
        gran    = child_doc.metadata.get("granularity", "")
        dataset = child_doc.metadata.get("dataset", "")
        parent_gran = _PARENT_GRANULARITY.get(gran)

        if not parent_gran:
            return None   # leaf node — no parent granularity defined

        candidates = self._gran_lookup.get((dataset, parent_gran), [])
        if not candidates:
            # Try without dataset constraint as last resort
            for (ds, g), docs in self._gran_lookup.items():
                if g == parent_gran:
                    candidates.extend(docs)

        if not candidates:
            return None

        # Score candidates by text overlap with the child summary
        # (cheap proxy — avoids a second FAISS call)
        child_text = child_doc.page_content.lower()
        best_doc   = None
        best_score = -1.0

        for cand in candidates:
            cand_text = cand.page_content.lower()
            # Simple word overlap score
            child_words = set(child_text.split())
            cand_words  = set(cand_text.split())
            overlap = len(child_words & cand_words) / max(len(child_words), 1)
            if overlap > best_score:
                best_score = overlap
                best_doc   = cand

        return best_doc

    def _find_parent(
        self, child_doc: Document, query: str = ""
    ) -> tuple[Optional[Document], str]:
        """
        Resolve parent for a child document.

        Tries exact lookup first, falls back to semantic lookup.

        Returns:
            (parent_doc, method) where method is 'exact' or 'semantic'
            or (None, 'none') if no parent found.
        """
        parent = self._find_parent_exact(child_doc)
        if parent is not None:
            return parent, "exact"

        parent = self._find_parent_semantic(child_doc, query)
        if parent is not None:
            return parent, "semantic"

        return None, "none"

    # -------------------------------------------------------------------------
    # Public retrieval methods
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Retrieve child documents and expand with parent context.

        Args:
            query: Natural language query string.
            k: Override the default k for child retrieval.

        Returns:
            List of Document objects: children first, then parents.
        """
        k = k or self.k
        children = self.faiss_store.similarity_search(query, k=k)

        seen_ids: set = set()
        for doc in children:
            row_id = doc.metadata.get("row_id", doc.metadata.get("source", ""))
            if row_id:
                seen_ids.add(row_id)

        parents: List[Document] = []
        for child in children:
            parent, method = self._find_parent(child, query)
            if parent is not None:
                parent_id = parent.metadata.get("row_id", parent.metadata.get("source", ""))
                if parent_id and parent_id not in seen_ids:
                    parents.append(parent)
                    seen_ids.add(parent_id)

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

        Children carry their FAISS similarity score.
        Parents carry score=0.0 and retrieval_method='parent_expansion'.
        The parent_resolution field records 'exact' or 'semantic' to
        distinguish true parent_id matches from semantic fallbacks.

        Args:
            query: Natural language query string.
            k: Override the default k.

        Returns:
            List of dicts with keys: row_id, score, dataset,
            granularity, retrieval_method, parent_resolution,
            page_content.
        """
        k = k or self.k
        children_with_scores = (
            self.faiss_store.similarity_search_with_score(query, k=k)
        )

        seen_ids: set = set()
        output: List[Dict[str, Any]] = []

        for doc, score in children_with_scores:
            row_id = doc.metadata.get("row_id", doc.metadata.get("source", ""))
            seen_ids.add(row_id)
            output.append({
                "row_id":             row_id,
                "score":              float(score),
                "dataset":            doc.metadata.get("dataset", ""),
                "granularity":        doc.metadata.get("granularity", ""),
                "retrieval_method":   "dense_child",
                "parent_resolution":  "n/a",
                "page_content":       doc.page_content,
                "parent_id":          doc.metadata.get("parent_id", ""),
            })

        exact_count    = 0
        semantic_count = 0

        for doc, _ in children_with_scores:
            child_doc = Document(
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            parent, method = self._find_parent(child_doc, query)

            if parent is not None:
                parent_id = parent.metadata.get("row_id", parent.metadata.get("source", ""))
                if parent_id and parent_id not in seen_ids:
                    seen_ids.add(parent_id)
                    output.append({
                        "row_id":            parent_id,
                        "score":             0.0,
                        "dataset":           parent.metadata.get("dataset", ""),
                        "granularity":       parent.metadata.get("granularity", ""),
                        "retrieval_method":  "parent_expansion",
                        "parent_resolution": method,   # 'exact' or 'semantic'
                        "page_content":      parent.page_content,
                        "parent_id":         parent.metadata.get("parent_id", ""),
                    })
                    if method == "exact":
                        exact_count += 1
                    else:
                        semantic_count += 1

        logger.info(
            "retrieve_with_scores: %d children, %d parents "
            "(%d exact + %d semantic fallback).",
            sum(1 for d in output if d["retrieval_method"] == "dense_child"),
            exact_count + semantic_count,
            exact_count, semantic_count,
        )
        return output

    def as_langchain_retriever(self):
        """Return a callable retriever compatible with LCEL chains.

        Returns the base FAISS retriever (child-only, no parent expansion).
        For full hierarchical context, call self.retrieve() directly.
        """
        logger.warning(
            "as_langchain_retriever() returns FAISS child-only retriever. "
            "For full hierarchical context, use retrieve() directly."
        )
        return self.faiss_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )
