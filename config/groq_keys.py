"""
config/groq_keys.py
====================
Loads all 46 Groq API keys from .env and exposes them as a list.

Used exclusively by src/experiments/groq_client.py (the rotation engine).
Nothing else should import from here directly — use RotatingGroqClient instead.

Keys are stored as:
    GROQ_API_KEY_1 ... GROQ_API_KEY_46
in your .env file.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _load_env() -> None:
    """Walk up from this file to find .env (same logic as config/models.py)."""
    search = Path(__file__).resolve().parent
    for _ in range(4):
        candidate = search / ".env"
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            return
        search = search.parent
    load_dotenv(override=False)


_load_env()


def get_all_groq_keys() -> list[str]:
    """
    Return a list of all configured Groq API keys (up to 6).

    Only returns keys that are set and not placeholder values.
    Raises EnvironmentError if no valid key is found at all.
    """
    keys: list[str] = []

    for i in range(1, 47):
        val = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
        if val and not val.startswith("your_"):
            keys.append(val)
        else:
            if val:
                logger.debug("GROQ_API_KEY_%d is a placeholder — skipping.", i)
            else:
                logger.debug("GROQ_API_KEY_%d not set — skipping.", i)

    if not keys:
        raise EnvironmentError(
            "No valid Groq API keys found.\n"
            "Set GROQ_API_KEY_1 through GROQ_API_KEY_6 in your .env file.\n"
            "Each key must be from a separate Groq account for true rate-limit rotation."
        )

    logger.info("Loaded %d Groq API key(s).", len(keys))
    return keys
