"""ChromaDB vector store builder and loader.

ChromaDB is an open-source embedded vector database with built-in
metadata filtering. It is used for:

    Pipeline 2 (hybrid) — metadata-filtered retrieval combining
        semantic search with keyword/zone/granularity filters
    Pipeline 3 (hierarchical) — parent_id lookup to fetch parent
        context alongside child documents

Key advantage over FAISS:
    ChromaDB stores metadata (dataset, granularity, zone_id, year,
    season, parent_id) directly in the collection and supports
    filtering at query time:

        chroma.as_retriever(
            search_kwargs={"filter": {"zone_id": "4", "season": "Winter"}}
        )

    FAISS does not support metadata filtering — it returns the k most
    similar vectors regardless of metadata. This makes ChromaDB essential
    for the hybrid and hierarchical retrieval pipelines.

Persistence:
    ChromaDB collections are stored as a local SQLite-backed directory.
    They survive Python process restarts without re-embedding.
    Saved to: outputs/indexes/chromadb/

Collection name:
    "energy_kb" — a single collection containing all 480 KB summaries
    from both datasets. Metadata filters separate GEFCom from household
    at query time rather than using separate collections.
"""

from __future__ import annotations

import logging
import chromadb
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Single collection name used for all KB summaries
COLLECTION_NAME: str = "energy_kb"


def build_chroma_index(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_dir: Path,
) -> Chroma:
    """Build a ChromaDB collection from KB documents and persist to disk.

    Embeds all document page_content strings, stores them alongside
    their full metadata dicts in a single ChromaDB collection named
    ``energy_kb``. The collection is automatically persisted to
    ``persist_dir`` as a SQLite-backed directory.

    Args:
        documents: List of LangChain Document objects from
            load_kb_documents(). Each document's metadata dict is
            stored in ChromaDB for filtered retrieval.
        embeddings: Configured HuggingFaceEmbeddings instance from
            get_embeddings_model().
        persist_dir: Directory path where ChromaDB will persist the
            collection. Created if it does not exist.

    Returns:
        Built Chroma vector store instance, ready for .as_retriever()
        or .similarity_search() calls with optional metadata filters.

    Example:
        >>> chroma_vs = build_chroma_index(docs, embeddings, Path("indexes/chromadb"))
        >>> results = chroma_vs.similarity_search(
        ...     "peak winter demand",
        ...     k=5,
        ...     filter={"dataset": "gefcom", "season": "Winter"},
        ... )
        >>> len(results)
        5
    """
    logger.info(
        "Building ChromaDB collection '%s' from %d documents...",
        COLLECTION_NAME, len(documents),
    )

    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.Client()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME,
    )

    logger.info(
        "ChromaDB collection '%s' built and persisted to %s. "
        "Documents indexed: %d. "
        "Metadata fields available for filtering: "
        "dataset, granularity, zone_id, year, month, season, parent_id.",
        COLLECTION_NAME, persist_dir, len(documents),
    )
    return vector_store


def load_chroma_index(
    persist_dir: Path,
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    """Load a previously persisted ChromaDB collection from disk.

    Connects to the existing SQLite-backed collection at ``persist_dir``
    without re-embedding any documents. The same embeddings model must
    be used at query time as was used at build time — otherwise the
    query vector will be in a different embedding space from the stored
    document vectors.

    Args:
        persist_dir: Directory path containing the persisted ChromaDB
            files (chroma.sqlite3 etc.).
        embeddings: Same HuggingFaceEmbeddings instance used to build
            the collection.

    Returns:
        Loaded Chroma vector store instance.

    Example:
        >>> chroma_vs = load_chroma_index(Path("indexes/chromadb"), embeddings)
        >>> retriever = chroma_vs.as_retriever(
        ...     search_kwargs={
        ...         "k": 5,
        ...         "filter": {"granularity": "daily"},
        ...     }
        ... )
    """
    logger.info(
        "Loading ChromaDB collection '%s' from %s...",
        COLLECTION_NAME, persist_dir,
    )

    client = chromadb.Client()

    vector_store = Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    count = vector_store._collection.count()
    logger.info(
        "ChromaDB collection '%s' loaded. Documents in collection: %d.",
        COLLECTION_NAME, count,
    )
    return vector_store
