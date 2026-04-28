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

import shutil
import logging
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Single collection name used for all KB summaries
COLLECTION_NAME: str = "energy_kb"

def clean_metadata(metadata: dict) -> dict:
    """Chroma accepts only simple metadata types."""
    clean = {}
    for k, v in metadata.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean

def build_chroma_index(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_dir: Path,
    batch_size: int = 25,
    reset_index: bool = True,
) -> Chroma:

    logger.info(
        "Building ChromaDB collection '%s' from %d documents in batches...",
        COLLECTION_NAME, len(documents)
    )

    if reset_index and persist_dir.exists():
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    cleaned_docs = [
        Document(
            page_content=str(doc.page_content),
            metadata=clean_metadata(doc.metadata)
        )
        for doc in documents
        if doc.page_content and str(doc.page_content).strip()
    ]

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    for start in range(0, len(cleaned_docs), batch_size):
        batch = cleaned_docs[start:start + batch_size]

        logger.info(
            "Adding documents %d to %d of %d",
            start + 1,
            min(start + batch_size, len(cleaned_docs)),
            len(cleaned_docs)
        )

        vector_store.add_documents(batch)

    logger.info(
        "ChromaDB collection '%s' built successfully. Documents indexed: %d",
        COLLECTION_NAME,
        len(cleaned_docs)
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

    vector_store = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    count = vector_store._collection.count()
    logger.info(
        "ChromaDB collection '%s' loaded. Documents in collection: %d.",
        COLLECTION_NAME, count,
    )
    return vector_store
