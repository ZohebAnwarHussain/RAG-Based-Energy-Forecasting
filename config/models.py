"""
config/models.py
=================
Central model configuration and API key management.

Single Gemini key (KB + golden dataset):
    from config.models import get_gemini_key

Rotating Groq client (all experiments):
    from src.experiments.groq_client import RotatingGroqClient
    client = RotatingGroqClient()

Legacy single-key accessor still works for non-experiment code:
    from config.models import get_groq_key   # returns first valid key
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _load_env() -> None:
    search = Path(__file__).resolve().parent
    for _ in range(4):
        candidate = search / ".env"
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            logger.debug("Loaded .env from %s", candidate)
            return
        search = search.parent
    load_dotenv(override=False)


_load_env()


# ---------------------------------------------------------------------------
# Gemini key
# ---------------------------------------------------------------------------

def get_gemini_key() -> str:
    """Return the Gemini API key from .env."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key.startswith("your_"):
        raise EnvironmentError(
            "GEMINI_API_KEY not set in .env\n"
            "Get yours at: https://aistudio.google.com/app/apikey"
        )
    return key


# ---------------------------------------------------------------------------
# Groq key — legacy single-key accessor (returns first valid key)
# For experiments, use RotatingGroqClient instead.
# ---------------------------------------------------------------------------

def get_groq_key() -> str:
    """
    Return the first valid Groq key found.
    Tries GROQ_API_KEY_1..6 then falls back to legacy GROQ_API_KEY.
    For multi-key rotation, use RotatingGroqClient directly.
    """
    # Try numbered keys first
    for i in range(1, 7):
        val = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
        if val and not val.startswith("your_"):
            return val

    # Fallback: legacy single key
    val = os.getenv("GROQ_API_KEY", "").strip()
    if val and not val.startswith("your_"):
        return val

    raise EnvironmentError(
        "No valid Groq API key found.\n"
        "Set GROQ_API_KEY_1 (through GROQ_API_KEY_6) in your .env file.\n"
        "Template: .env.template"
    )


# Convenience alias kept for backward compat with older notebooks
def get_api_key(provider: str) -> str:
    provider = provider.lower().strip()
    if provider == "gemini":
        return get_gemini_key()
    if provider == "groq":
        return get_groq_key()
    raise ValueError(f"Unknown provider '{provider}'. Use 'gemini' or 'groq'.")


# ---------------------------------------------------------------------------
# Model name constants
# ---------------------------------------------------------------------------

MODELS = {
    "gemini_kb":   os.getenv("GEMINI_KB_MODEL",  "gemini-2.0-flash-preview"),
    "gemini_gd":   os.getenv("GEMINI_GD_MODEL",  "gemini-2.5-flash"),
    "embedding":   os.getenv("EMBEDDING_MODEL",  "sentence-transformers/all-MiniLM-L6-v2"),
    "groq_rag":    os.getenv("GROQ_RAG_MODEL",   "llama-3.3-70b-versatile"),
    "groq_judge":  os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile"),
}

# ---------------------------------------------------------------------------
# Experiment defaults
# ---------------------------------------------------------------------------

EXP_DEFAULTS = {
    "top_k_values": [int(k) for k in
                     os.getenv("EXP_TOP_K_VALUES", "3,5,10").split(",")],
    "temperature":  float(os.getenv("EXP_TEMPERATURE", "0")),
    "max_tokens":   int(os.getenv("EXP_MAX_TOKENS",   "500")),
    "min_docs":     int(os.getenv("EXP_MIN_DOCS",     "200")),
}

# ---------------------------------------------------------------------------
# Legacy constants — kept for backward compatibility with notebooks 01–06
# These map the old import names to the new MODELS dict values
# ---------------------------------------------------------------------------

# Model names
KB_MODEL_NAME        = MODELS["gemini_kb"]
GOLDEN_MODEL_NAME    = MODELS["gemini_gd"]
RAG_MODEL_NAME       = MODELS["groq_rag"]
EMBEDDING_MODEL_NAME = MODELS["embedding"]

# Temperatures
KB_TEMPERATURE       = EXP_DEFAULTS["temperature"]
GOLDEN_TEMPERATURE   = EXP_DEFAULTS["temperature"]
RAG_TEMPERATURE      = EXP_DEFAULTS["temperature"]

# Max tokens
KB_MAX_TOKENS        = EXP_DEFAULTS["max_tokens"]
GOLDEN_MAX_TOKENS    = EXP_DEFAULTS["max_tokens"]
RAG_MAX_TOKENS       = EXP_DEFAULTS["max_tokens"]