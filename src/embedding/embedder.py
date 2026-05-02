"""Sentence-transformer embedding model wrapper.

Wraps the all-MiniLM-L6-v2 model in LangChain's HuggingFaceEmbeddings
interface so it can be passed directly to Chroma.from_documents() and
FAISS.from_documents().

Key properties of all-MiniLM-L6-v2:
    - 384-dimensional dense sentence embeddings
    - Runs entirely on local CPU — no API calls, no cost, no rate limits
    - Neutral relative to both Gemini and Llama — no model family bias
    - De facto research baseline for RAG retrieval benchmarks
    - Captures semantic similarity: "peak demand" and "high load" map to
      similar vectors even though they share no words

Why not Gemini or Groq embeddings:
    - Gemini embeddings would be tuned to Gemini-written text (KB bias)
    - Groq does not offer an embedding endpoint
    - Local embeddings avoid all API call overhead and rate limit concerns

Normalisation:
    encode_kwargs={"normalize_embeddings": True} ensures all vectors
    have unit length. This means cosine similarity and inner product
    give identical results, simplifying FAISS index type selection
    (IndexFlatIP is equivalent to cosine similarity with normalised vectors).
"""

from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Return the sentence-transformer embedding model wrapped in LangChain.

    The model is downloaded from HuggingFace Hub on first use and cached
    locally at ``~/.cache/huggingface/``. Subsequent calls load from
    cache with no internet access required.

    Returns:
        Configured HuggingFaceEmbeddings instance that accepts text
        strings and returns 384-dimensional normalised float vectors.

    Example:
        >>> embeddings = get_embeddings_model()
        >>> vector = embeddings.embed_query("What was peak winter demand?")
        >>> len(vector)
        384
        >>> abs(sum(v**2 for v in vector) - 1.0) < 0.01  # unit length
        True
    """
    logger.info(
        "Loading embedding model '%s' (runs locally on CPU).",
        EMBEDDING_MODEL_NAME,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info(
        "Embedding model loaded. Dimension: 384. "
        "Normalisation: enabled (cosine ≡ inner product)."
    )
    return embeddings
    
class Embedder:
    """Thin wrapper around get_embeddings_model() for experiments that
    expect a class-based interface."""

    def __init__(self):
        self._model = get_embeddings_model()

    def embed_query(self, text: str):
        return self._model.embed_query(text)

    def embed_documents(self, texts):
        return self._model.embed_documents(texts)

    # Allow passing the Embedder instance directly to LangChain/FAISS
    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)
