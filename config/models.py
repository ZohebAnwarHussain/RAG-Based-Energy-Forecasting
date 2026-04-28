"""Model configuration constants.

Defines the model strings, temperatures, and token limits used across
the four LLM stages of the pipeline. Centralising these here ensures
that any model change happens in one place and propagates consistently
through all dependent modules.

Model independence boundaries:
    - KB Generation (Gemini 3) ←→ Golden Dataset (Gemini 2.5)
      Different versions reduce style-matching contamination.
    - Golden Dataset (Gemini family) ←→ RAG Generation (Llama family)
      Primary independence boundary — different providers entirely.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base Generation — Phase 1
# ─────────────────────────────────────────────────────────────────────────────
# Model used to generate human-tone summaries from statistical aggregates.
KB_MODEL_NAME: str  = "gemini-3-flash-preview"
KB_TEMPERATURE: float = 0.2     # Low for factual, reproducible summaries
KB_MAX_TOKENS: int    = 1024    # Summaries are short paragraphs (3-5 sentences)


# ─────────────────────────────────────────────────────────────────────────────
# Golden Dataset Generation — Phase 2
# ─────────────────────────────────────────────────────────────────────────────
# Stable Gemini release used to generate reference answers.
# Different generation from KB model maintains evaluation independence.
GOLDEN_MODEL_NAME: str   = "gemini-2.5-flash"
GOLDEN_TEMPERATURE: float = 0.2  # Low for consistent reference answers
GOLDEN_MAX_TOKENS: int    = 2048  # Reference answers may be longer than summaries


# ─────────────────────────────────────────────────────────────────────────────
# RAG Generation — Phase 5
# ─────────────────────────────────────────────────────────────────────────────
# Llama 3.3 70B via Groq. Independent from Gemini family used in KB and Golden.
# Also serves as the RAGAS judge in Phase 6 (independent of both KB and golden).
RAG_MODEL_NAME: str    = "llama-3.3-70b-versatile"
RAG_TEMPERATURE: float = 0.2     # Low for factual RAG answers
RAG_MAX_TOKENS: int    = 1024    # RAG answers are paragraph-length insights


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Model — Phase 3
# ─────────────────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 — 384-dimensional sentence embeddings.
# Free, runs on CPU, neutral relative to both Gemini and Llama, and is the
# de facto research baseline for RAG retrieval evaluation.
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
