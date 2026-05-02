"""
src/experiments/groq_client.py
================================
RotatingGroqClient — thread-safe Groq client that cycles through
multiple API keys and handles rate limits automatically.

Behaviour
---------
1. Round-robin across all configured keys on every request.
2. On 429 (rate limit):
   - Marks the current key as cooling down for COOLDOWN_SECONDS (default 62s).
   - Immediately switches to the next available key and retries.
   - If ALL keys are cooling down simultaneously, waits for the soonest
     recovery and retries with exponential backoff.
3. Reads remaining quota from Groq response headers after every call
   and proactively rotates if remaining requests < LOW_REQUESTS_THRESHOLD.
4. Logs key index (never the key value itself), quota status, and
   cooldown events so you can monitor progress in the notebook.

Usage
-----
    from src.experiments.groq_client import RotatingGroqClient

    client = RotatingGroqClient()          # loads all keys from .env
    response = client.chat(
        messages=[{"role": "user", "content": "Hello"}],
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=500,
    )
    text = response.choices[0].message.content
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from groq import Groq, RateLimitError, APIStatusError


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants (override via constructor kwargs)
# ---------------------------------------------------------------------------

DEFAULT_COOLDOWN_SECONDS    = 62     # how long to wait after a 429 on one key
DEFAULT_MAX_RETRIES         = 8      # max attempts across all keys before giving up
DEFAULT_RETRY_BASE_DELAY    = 2.0    # base seconds for exponential backoff
LOW_REQUESTS_THRESHOLD      = 3      # proactively rotate if remaining RPM < this


# ---------------------------------------------------------------------------
# Per-key state
# ---------------------------------------------------------------------------

@dataclass
class _KeyState:
    index:          int                    # 1-based, for logging only
    client:         Groq                   # groq SDK client for this key
    cooldown_until: float = 0.0            # epoch time when cooldown expires
    total_calls:    int   = 0
    total_errors:   int   = 0
    remaining_rpm:  Optional[int] = None   # from response headers
    remaining_tpm:  Optional[int] = None

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    @property
    def seconds_until_ready(self) -> float:
        return max(0.0, self.cooldown_until - time.time())


# ---------------------------------------------------------------------------
# Rotating client
# ---------------------------------------------------------------------------

class RotatingGroqClient:
    """
    Groq client that rotates across multiple API keys on rate-limit errors.

    Parameters
    ----------
    cooldown_seconds : seconds to cool down a key after a 429
    max_retries      : total attempts before raising
    retry_base_delay : base for exponential backoff when all keys exhausted
    """

    def __init__(
        self,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        max_retries: int        = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    ):
        from config.groq_keys import get_all_groq_keys
        
        self._lock            = threading.Lock()
        self._cooldown        = cooldown_seconds
        self._max_retries     = max_retries
        self._retry_base      = retry_base_delay
        self._current_idx     = 0            # round-robin pointer
        self._total_requests  = 0
        self._total_429s      = 0
        self._total_rotations = 0

        raw_keys   = get_all_groq_keys()
        self._keys = [
            _KeyState(index=i + 1, client=Groq(api_key=k))
            for i, k in enumerate(raw_keys)
        ]

        logger.info(
            "RotatingGroqClient ready — %d key(s) loaded.", len(self._keys)
        )
        self._log_status()

    # ── Public API ──────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        model:       str   = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens:  int   = 500,
        **kwargs: Any,
    ):
        """
        Call the Groq chat completions endpoint with automatic key rotation.

        Parameters
        ----------
        messages    : list of {"role": ..., "content": ...} dicts
        model       : Groq model name
        temperature : sampling temperature (0 = deterministic)
        max_tokens  : maximum output tokens

        Returns
        -------
        groq.types.chat.ChatCompletion
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            key_state = self._pick_key()

            if key_state is None:
                # All keys cooling down — wait for the soonest recovery
                wait = self._wait_for_any_key()
                logger.warning(
                    "All %d keys cooling down. Waiting %.1fs for next available key.",
                    len(self._keys), wait,
                )
                time.sleep(wait + 0.5)
                key_state = self._pick_key()

            if key_state is None:
                # Still None after waiting — should not happen, but guard anyway
                delay = self._retry_base * (2 ** (attempt - 1))
                logger.error("No key available after wait. Sleeping %.1fs.", delay)
                time.sleep(delay)
                continue

            try:
                response = key_state.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                self._on_success(key_state, response)
                return response

            except RateLimitError as exc:
                self._on_rate_limit(key_state, exc, attempt)
                last_exc = exc
                # Do NOT sleep — just rotate to next key immediately

            except APIStatusError as exc:
                if exc.status_code == 429:
                    self._on_rate_limit(key_state, exc, attempt)
                    last_exc = exc
                else:
                    key_state.total_errors += 1
                    logger.error(
                        "[Key %d] API error %d on attempt %d: %s",
                        key_state.index, exc.status_code, attempt, exc.message,
                    )
                    raise

            except Exception as exc:
                key_state.total_errors += 1
                logger.error(
                    "[Key %d] Unexpected error on attempt %d: %s",
                    key_state.index, attempt, exc,
                )
                raise

        raise RuntimeError(
            f"All {self._max_retries} attempts failed. Last error: {last_exc}"
        )

    @property
    def stats(self) -> dict[str, Any]:
        """Return aggregate usage statistics across all keys."""
        return {
            "total_requests":  self._total_requests,
            "total_429s":      self._total_429s,
            "total_rotations": self._total_rotations,
            "n_keys":          len(self._keys),
            "keys": [
                {
                    "index":         k.index,
                    "total_calls":   k.total_calls,
                    "total_errors":  k.total_errors,
                    "available":     k.is_available,
                    "remaining_rpm": k.remaining_rpm,
                    "remaining_tpm": k.remaining_tpm,
                }
                for k in self._keys
            ],
        }

    def log_stats(self) -> None:
        """Print a formatted stats summary to the logger."""
        s = self.stats
        logger.info(
            "RotatingGroqClient stats — total_requests=%d  429s=%d  rotations=%d",
            s["total_requests"], s["total_429s"], s["total_rotations"],
        )
        for k in s["keys"]:
            avail = "✅" if k["available"] else "⏳"
            logger.info(
                "  Key %d %s  calls=%d  errors=%d  rpm_remaining=%s  tpm_remaining=%s",
                k["index"], avail,
                k["total_calls"], k["total_errors"],
                k["remaining_rpm"], k["remaining_tpm"],
            )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _pick_key(self) -> Optional[_KeyState]:
        """
        Return the next available key in round-robin order.
        Returns None if all keys are currently cooling down.
        """
        with self._lock:
            n = len(self._keys)
            for _ in range(n):
                ks = self._keys[self._current_idx % n]
                self._current_idx = (self._current_idx + 1) % n

                if not ks.is_available:
                    continue

                # Proactive rotation: if remaining RPM is very low, skip this key
                if (
                    ks.remaining_rpm is not None
                    and ks.remaining_rpm < LOW_REQUESTS_THRESHOLD
                ):
                    logger.debug(
                        "[Key %d] Only %d RPM remaining — skipping proactively.",
                        ks.index, ks.remaining_rpm,
                    )
                    continue

                return ks

            return None   # all keys unavailable

    def _wait_for_any_key(self) -> float:
        """Return the minimum seconds until any key recovers."""
        now = time.time()
        waits = [
            max(0.0, ks.cooldown_until - now)
            for ks in self._keys
        ]
        return min(waits) if waits else self._cooldown

    def _on_success(self, ks: _KeyState, response: Any) -> None:
        """Update state after a successful call."""
        ks.total_calls += 1
        self._total_requests += 1

        # Parse quota headers if available
        try:
            headers = response.headers if hasattr(response, "headers") else {}
            if hasattr(response, "_raw_response"):
                headers = response._raw_response.headers

            rpm = headers.get("x-ratelimit-remaining-requests")
            tpm = headers.get("x-ratelimit-remaining-tokens")

            if rpm is not None:
                ks.remaining_rpm = int(rpm)
            if tpm is not None:
                ks.remaining_tpm = int(tpm)

        except Exception:
            pass   # headers not always available; non-fatal

        if self._total_requests % 20 == 0:
            self.log_stats()

    def _on_rate_limit(self, ks: _KeyState, exc: Exception, attempt: int) -> None:
        """Mark key as cooling down and rotate."""
        ks.cooldown_until = time.time() + self._cooldown
        ks.total_errors  += 1
        self._total_429s += 1
        self._total_rotations += 1

        logger.warning(
            "[Key %d] 429 rate-limited on attempt %d — "
            "cooling for %.0fs, rotating to next key.",
            ks.index, attempt, self._cooldown,
        )

    def _log_status(self) -> None:
        logger.info(
            "Key pool: %d available, %d total. "
            "Cooldown=%.0fs  MaxRetries=%d",
            sum(1 for k in self._keys if k.is_available),
            len(self._keys),
            self._cooldown,
            self._max_retries,
        )
