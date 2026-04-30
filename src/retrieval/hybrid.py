"""Pipeline 2 — Hybrid Retrieval (BM25 + FAISS).

Combines sparse keyword retrieval (BM25) with dense semantic retrieval
(FAISS) via LangChain's EnsembleRetriever. The two retrievers are
weighted by an alpha parameter:

    final_score = alpha * BM25_score + (1 - alpha) * FAISS_score

BM25 catches exact keyword matches that dense search may miss:
    - Zone identifiers: "Zone 4", "Zone 21", "system"
    - Appliance names: "Sub_metering_1", "kitchen", "HVAC"
    - Temporal keywords: "weekday", "weekend", "January"

FAISS catches semantic similarity that BM25 misses:
    - "peak demand" ↔ "high load" (different words, same meaning)
    - "energy consumption" ↔ "power usage" (synonyms)

The combination outperforms either component alone on queries that
need both keyword precision and semantic understanding — which is
most real-world energy domain queries.

Post-retrieval metadata filtering is available for dataset and
granularity constraints (replacing ChromaDB's query-time filtering).

Strengths:
    - Best of both worlds — keywords + semantics
    - Configurable alpha for sparse/dense balance
    - Metadata filtering available post-retrieval

Weaknesses:
    - Slower than pure dense (two retrieval passes)
    - Alpha tuning needed for optimal performance
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining BM25 sparse and FAISS dense search.

    Uses LangChain's EnsembleRetriever to fuse results from both
    retrieval methods using Reciprocal Rank Fusion (RRF).

    Args:
        faiss_store: Built or loaded FAISS vector store.
        documents: Full list of KB Document objects (needed by BM25
            which operates over raw text, not embeddings).
        k: Number of documents to retrieve per query. Default 5.
        weights: List of two floats [bm25_weight, faiss_weight].
            Must sum to 1.0. Default [0.3, 0.7] — biased toward
            dense retrieval with BM25 as a keyword safety net.

    Example:
        >>> retriever = HybridRetriever(
        ...     faiss_vs, documents, k=5, weights=[0.3, 0.7]
        ... )
        >>> docs = retriever.retrieve("Zone 4 winter peak demand")
        >>> len(docs)
        5
    """

    def __init__(
        self,
        faiss_store: FAISS,
        documents: List[Document],
        k: int = 5,
        weights: Optional[List[float]] = None,
    ) -> None:
        """Initialise hybrid retriever with BM25 and FAISS components.

        Args:
            faiss_store: FAISS vector store instance.
            documents: Full list of KB Document objects for BM25.
            k: Number of results to return per query.
            weights: [bm25_weight, faiss_weight]. Default [0.3, 0.7].
        """
        self.k = k
        self.weights = weights or [0.3, 0.7]

        # BM25 retriever over raw document texts
        self.bm25_retriever = BM25Retriever.from_documents(
            documents, k=k
        )

        # FAISS retriever for dense semantic search
        self.faiss_retriever = faiss_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

        # LangChain EnsembleRetriever fuses both using RRF
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=self.weights,
        )

        logger.info(
            "HybridRetriever initialised: k=%d, weights=%s "
            "(BM25=%.1f, FAISS=%.1f).",
            k, self.weights, self.weights[0], self.weights[1],
        )

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dataset: Optional[str] = None,
        filter_granularity: Optional[str] = None,
    ) -> List[Document]:
        """Retrieve documents using hybrid BM25 + FAISS fusion.

        Optionally applies post-retrieval metadata filtering to restrict
        results to a specific dataset or granularity. Over-fetches by 2x
        when filters are applied to ensure enough results survive
        filtering.

        Args:
            query: Natural language query string.
            k: Override the default k for this call.
            filter_dataset: Optional — restrict to 'gefcom' or 'household'.
            filter_granularity: Optional — restrict to 'daily', 'weekly', etc.

        Returns:
            List of Document objects, filtered and truncated to k.
        """
        k = k or self.k
        needs_filter = filter_dataset or filter_granularity

        # Over-fetch when filtering to ensure enough results survive
        if needs_filter:
            self.bm25_retriever.k = k * 3
            self.faiss_retriever.search_kwargs["k"] = k * 3

        results = self.ensemble_retriever.invoke(query)

        # Reset k after retrieval
        self.bm25_retriever.k = self.k
        self.faiss_retriever.search_kwargs["k"] = self.k

        # Post-retrieval metadata filtering
        if filter_dataset:
            results = [
                doc for doc in results
                if doc.metadata.get("dataset") == filter_dataset
            ]
        if filter_granularity:
            results = [
                doc for doc in results
                if doc.metadata.get("granularity") == filter_granularity
            ]

        results = results[:k]

        logger.info(
            "Hybrid retrieval: query='%s...' → %d results "
            "(filters: dataset=%s, granularity=%s).",
            query[:50], len(results),
            filter_dataset or "none",
            filter_granularity or "none",
        )
        return results

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dataset: Optional[str] = None,
        filter_granularity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve with metadata formatted for evaluation.

        Args:
            query: Natural language query string.
            k: Override the default k.
            filter_dataset: Optional dataset filter.
            filter_granularity: Optional granularity filter.

        Returns:
            List of dicts with keys: row_id, rank, dataset,
            granularity, page_content.
        """
        docs = self.retrieve(
            query, k=k,
            filter_dataset=filter_dataset,
            filter_granularity=filter_granularity,
        )
        output = []
        for rank, doc in enumerate(docs, 1):
            output.append({
                "row_id":       doc.metadata.get("source", ""),
                "rank":         rank,
                "dataset":      doc.metadata.get("dataset", ""),
                "granularity":  doc.metadata.get("granularity", ""),
                "page_content": doc.page_content,
            })
        return output

    def as_langchain_retriever(self):
        """Return the LangChain EnsembleRetriever for LCEL chains.

        Returns:
            EnsembleRetriever instance usable with the pipe operator.
        """
        return self.ensemble_retriever
