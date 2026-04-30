"""Pipeline 1 — Dense Retrieval via FAISS.

The simplest retrieval strategy. Embeds the user query using
all-MiniLM-L6-v2 and returns the k nearest KB summaries by
cosine similarity (implemented as inner product over normalised
vectors in FAISS IndexFlatIP).

This is the baseline pipeline. Every query in the golden dataset
is run through dense retrieval to establish a minimum performance
floor that the hybrid and hierarchical pipelines must beat.

Strengths:
    - Captures semantic similarity ("peak demand" matches "high load")
    - No configuration needed beyond k
    - Fast — single FAISS call

Weaknesses:
    - Cannot match exact keywords (zone IDs, Sub_metering column names)
    - No metadata filtering — may return household docs for GEFCom queries
    - No multi-scale context — returns individual chunks without parents
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense semantic retrieval using FAISS similarity search.

    Wraps a FAISS vector store and provides a consistent retrieve()
    interface used by the RAG chain and evaluation pipeline.

    Args:
        faiss_store: Built or loaded FAISS vector store from
            src.embedding.faiss_store.
        k: Number of documents to retrieve per query. Default 5.

    Example:
        >>> from src.embedding import load_faiss_index, get_embeddings_model
        >>> embeddings = get_embeddings_model()
        >>> faiss_vs = load_faiss_index(PATHS["faiss_index"], embeddings)
        >>> retriever = DenseRetriever(faiss_vs, k=5)
        >>> docs = retriever.retrieve("What was peak winter demand?")
        >>> len(docs)
        5
    """

    def __init__(self, faiss_store: FAISS, k: int = 5) -> None:
        """Initialise dense retriever with a FAISS vector store.

        Args:
            faiss_store: FAISS vector store instance.
            k: Number of results to return per query.
        """
        self.faiss_store = faiss_store
        self.k = k
        logger.info("DenseRetriever initialised with k=%d.", k)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Retrieve the k most semantically similar KB summaries.

        Args:
            query: Natural language query string.
            k: Override the default k for this call. If None, uses
                the instance default set at initialisation.

        Returns:
            List of Document objects sorted by descending similarity.
            Each document has page_content (summary text) and metadata
            (source, dataset, granularity, zone_id, etc.).
        """
        k = k or self.k
        results = self.faiss_store.similarity_search(query, k=k)
        logger.info(
            "Dense retrieval: query='%s...' → %d results.",
            query[:50], len(results),
        )
        return results

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve with similarity scores for evaluation.

        Returns a list of dicts with document metadata and score,
        formatted for direct use in retrieval metric calculation.

        Args:
            query: Natural language query string.
            k: Override the default k for this call.

        Returns:
            List of dicts with keys: row_id, score, dataset,
            granularity, page_content.
        """
        k = k or self.k
        results = self.faiss_store.similarity_search_with_score(
            query, k=k
        )
        output = []
        for doc, score in results:
            output.append({
                "row_id":       doc.metadata.get("source", ""),
                "score":        float(score),
                "dataset":      doc.metadata.get("dataset", ""),
                "granularity":  doc.metadata.get("granularity", ""),
                "page_content": doc.page_content,
            })
        return output

    def as_langchain_retriever(self):
        """Return a LangChain-compatible retriever for LCEL chains.

        Returns:
            LangChain retriever object that can be used with the
            pipe operator in LCEL chains:
            ``retriever | format_docs | prompt | llm | parser``
        """
        return self.faiss_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )
