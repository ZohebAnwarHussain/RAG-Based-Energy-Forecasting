"""FAISS vector store builder and loader.

FAISS (Facebook AI Similarity Search) is used for Pipeline 1 (dense
retrieval) and as the dense component of Pipeline 2 (hybrid retrieval).

Index type:
    IndexFlatIP — exact inner product search. With normalised embeddings
    (from embedder.py), inner product equals cosine similarity. Suitable
    for KB sizes up to ~10,000 entries. For larger KBs, upgrade to
    IndexIVFFlat for approximate nearest-neighbour search.

Persistence:
    FAISS indexes are saved as binary files to disk. They load
    instantaneously and do not require re-embedding on restart.
    Saved to: outputs/indexes/faiss/

LangChain integration:
    FAISS.from_documents() accepts the same Document list and
    HuggingFaceEmbeddings instance as Chroma.from_documents(),
    so one document loading step feeds both vector stores.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def build_faiss_index(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    save_path: Path,
) -> FAISS:
    """Build a FAISS index from KB documents and persist to disk.

    Embeds all document page_content strings using the provided
    embeddings model, builds an IndexFlatIP index, and saves the
    index + metadata to ``save_path``.

    Args:
        documents: List of LangChain Document objects from
            load_kb_documents(). Each document's page_content is
            embedded; metadata is stored alongside for retrieval.
        embeddings: Configured HuggingFaceEmbeddings instance from
            get_embeddings_model().
        save_path: Directory path where the FAISS index files will
            be saved. Created if it does not exist.

    Returns:
        Built FAISS vector store instance, ready for .as_retriever()
        or .similarity_search() calls.

    Example:
        >>> faiss_vs = build_faiss_index(docs, embeddings, Path("indexes/faiss"))
        >>> results = faiss_vs.similarity_search("peak winter demand", k=5)
        >>> len(results)
        5
    """
    logger.info(
        "Building FAISS index from %d documents...", len(documents)
    )

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    save_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_path))

    logger.info(
        "FAISS index built and saved to %s. "
        "Index type: IndexFlatIP (exact search). "
        "Documents indexed: %d.",
        save_path, len(documents),
    )
    return vector_store


def load_faiss_index(
    load_path: Path,
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """Load a previously saved FAISS index from disk.

    Loads the binary index file and the associated metadata (document
    texts + metadata dicts) from the directory specified by ``load_path``.

    The ``allow_dangerous_deserialization=True`` flag is required because
    FAISS uses pickle internally to store metadata alongside the index.
    This is safe when loading indexes you built yourself — do not load
    indexes from untrusted sources.

    Args:
        load_path: Directory path containing the saved FAISS index files.
        embeddings: Same HuggingFaceEmbeddings instance used to build
            the index. Required for query encoding at search time.

    Returns:
        Loaded FAISS vector store instance.

    Raises:
        ValueError: If the index files are not found at ``load_path``.

    Example:
        >>> faiss_vs = load_faiss_index(Path("indexes/faiss"), embeddings)
        >>> results = faiss_vs.similarity_search("Zone 4 daily load", k=3)
    """
    logger.info("Loading FAISS index from %s...", load_path)

    vector_store = FAISS.load_local(
        folder_path=str(load_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    logger.info("FAISS index loaded successfully from %s.", load_path)
    return vector_store
