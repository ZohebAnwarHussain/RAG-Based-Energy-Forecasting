"""
src/retrieval/reranker.py
==========================
Cross-Encoder Re-Ranker for EXP_02 and EXP_03.

WHY THIS EXISTS
---------------
FAISS (bi-encoder) and BM25 both score query–document relevance
*independently* — the query and document are embedded separately and
compared by cosine similarity or term overlap. This is fast but
imprecise: the bi-encoder never sees the query and document *together*.

A cross-encoder reads the concatenated [query, document] pair in one
forward pass and outputs a calibrated relevance score. It is far more
accurate at ranking, but too slow to run over the full KB (140 docs
× 50 queries = 7,000 forward passes per K value). The solution is a
two-stage pipeline:

    Stage 1 — Recall:  FAISS/BM25 retrieves top K * RECALL_MULTIPLIER
                        candidates quickly (e.g. K=3 → fetch 15 docs)
    Stage 2 — Rerank:  Cross-encoder scores all 15, keeps top K

This is the standard production RAG pattern and is expected to
substantially improve Context Precision@K (currently 0.000–0.04).

MODEL CHOICE
------------
cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS-MARCO passage ranking (IR task, close to our use case)
  - 22M parameters — fast inference on CPU (~5–15ms per pair)
  - No API key needed — runs fully locally via sentence-transformers
  - Proven strong performance on domain-specific retrieval tasks

Alternative: cross-encoder/ms-marco-TinyBERT-L-2-v2 (even faster, slightly weaker)

USAGE
-----
    from src.retrieval.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()          # loads model once
    top_docs = reranker.rerank(query, candidates, top_k=3)

INSTALL
-------
    pip install sentence-transformers
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default cross-encoder model — small, fast, CPU-friendly
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates to fetch before reranking (multiplier × final K)
# e.g. K=3 → fetch 15 candidates → rerank → return top 3
RECALL_MULTIPLIER = 4


class CrossEncoderReranker:
    """
    Re-ranks a list of retrieved documents using a cross-encoder model.

    The cross-encoder scores each (query, document_text) pair jointly,
    producing calibrated relevance scores that are far more accurate than
    bi-encoder cosine similarities for ranking purposes.

    Args:
        model_name: HuggingFace model ID for the cross-encoder.
                    Default: cross-encoder/ms-marco-MiniLM-L-6-v2
        device:     'cpu' or 'cuda'. Default 'cpu' (safe for Colab free tier).

    Example:
        >>> reranker = CrossEncoderReranker()
        >>> docs = dense_retriever.retrieve(query, k=15)   # over-fetch
        >>> top3 = reranker.rerank(query, docs, top_k=3)   # rerank → keep 3
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name, device=device)
            self._model_name = model_name
            logger.info(
                "CrossEncoderReranker loaded: model=%s device=%s",
                model_name, device,
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker.\n"
                "Install with: pip install sentence-transformers"
            )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate documents and return the top_k most relevant.

        Each candidate dict must contain a text field (default key: 'text').
        The cross-encoder score is added to each dict as 'rerank_score'.
        Results are sorted by descending rerank_score.

        Args:
            query:      The user query string.
            candidates: List of dicts from retrieve() or retrieve_with_scores().
                        Each must have at least a 'text' key (or set text_key).
            top_k:      Number of documents to return after reranking.
            text_key:   Key in each candidate dict that holds the document text.
                        DenseRetriever uses 'page_content'; HybridRetriever
                        uses 'text'. Default 'text'.

        Returns:
            List of top_k candidate dicts, each with an added 'rerank_score' field,
            sorted by descending rerank_score.
        """
        if not candidates:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [
            (query, c.get(text_key) or c.get("page_content") or "")
            for c in candidates
        ]

        scores = self._model.predict(pairs)

        # Attach score to each candidate
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        # Sort by rerank score descending, return top_k
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        top = reranked[:top_k]

        logger.debug(
            "Reranked %d candidates → top %d. Top score: %.4f",
            len(candidates), top_k, top[0]["rerank_score"] if top else 0.0,
        )
        return top

    def rerank_documents(
        self,
        query: str,
        documents: list,   # List[langchain_core.documents.Document]
        top_k: int,
    ) -> list:
        """
        Rerank a list of LangChain Document objects.

        Convenience wrapper for use with DenseRetriever.retrieve() which
        returns Document objects rather than dicts.

        Args:
            query:     The user query string.
            documents: List of LangChain Document objects.
            top_k:     Number of documents to return after reranking.

        Returns:
            List of top_k Document objects, sorted by relevance.
        """
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        scored = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        top = [doc for doc, _ in scored[:top_k]]

        logger.debug(
            "Reranked %d Documents → top %d.", len(documents), top_k
        )
        return top
