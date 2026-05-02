"""
groq_key_checker.py
====================
Check Groq API key TPD status in a clean tabular format.

IMPORTANT — Groq header behaviour:
  - 200 response : key is alive, TPD headers NOT returned (Groq design)
  - 429 response : TPD/TPM exhausted, full headers + error body returned
  - 401 response : key invalid or expired

This means exact TPD remaining is only knowable when a key hits 429.
For 200 responses we show ">0 (fresh)" since the key is confirmed working.

Usage:
    from groq_key_checker import check_all_keys
    check_all_keys()
"""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

_repo = Path(__file__).resolve().parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import httpx
from config.groq_keys import get_all_groq_keys

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
TEST_MODEL    = "llama-3.3-70b-versatile"
IST_OFFSET    = timedelta(hours=5, minutes=30)
TPD_LIMIT     = 100_000

_MINIMAL_PAYLOAD = {
    "model":      TEST_MODEL,
    "max_tokens": 1,
    "messages":   [{"role": "user", "content": "1"}],
}


# ── Probe one key ─────────────────────────────────────────────────────────────

def _parse_tpd_from_error(msg: str) -> tuple[int, int]:
    """Extract (used, remaining) from a Groq 429 TPD error message body."""
    m = re.search(r"Used\s+([\d,]+),\s*Requested\s+([\d,]+)", msg)
    if m:
        used = int(m.group(1).replace(",", ""))
        return used, max(0, TPD_LIMIT - used)
    return TPD_LIMIT, 0


def _parse_retry(msg: str) -> str:
    """Extract 'X min Y sec' retry string from a 429 error message."""
    m = re.search(r"in\s+(?:(\d+)h\s*)?(?:(\d+)m\s*)?([\d.]+)s", msg)
    if not m:
        return "soon"
    h   = int(m.group(1) or 0)
    mn  = int(m.group(2) or 0)
    sec = float(m.group(3) or 0)
    total_min = h * 60 + mn + sec / 60
    if total_min >= 60:
        return f"{h}h {mn}m"
    if total_min >= 1:
        return f"{int(total_min)}m {int(sec % 60)}s"
    return f"{int(sec)}s"


def _probe(key: str, idx: int) -> dict:
    r = {
        "idx":       idx,
        "preview":   f"{key[:8]}...{key[-4:]}",
        "status":    "unknown",
        "tpd_used":  None,
        "tpd_rem":   None,
        "retry_in":  None,
    }
    try:
        resp = httpx.post(
            GROQ_CHAT_URL,
            json=_MINIMAL_PAYLOAD,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=15.0,
        )
        if resp.status_code == 200:
            r["status"]   = "ok"
            r["tpd_used"] = None   # unknown — Groq never returns this on 200
            r["tpd_rem"]  = None   # unknown but confirmed > 0

        elif resp.status_code == 429:
            try:
                body    = resp.json()
                err_msg = body.get("error", {}).get("message", "")
            except Exception:
                err_msg = resp.text

            if "tokens per day" in err_msg.lower():
                used, rem      = _parse_tpd_from_error(err_msg)
                r["status"]    = "tpd_exhausted"
                r["tpd_used"]  = used
                r["tpd_rem"]   = rem
                r["retry_in"]  = _parse_retry(err_msg)
            elif "tokens per minute" in err_msg.lower():
                r["status"]   = "tpm_limit"
                r["tpd_rem"]  = None   # TPD may still be available
                r["retry_in"] = _parse_retry(err_msg)
            else:
                r["status"] = "rate_limited"

        elif resp.status_code == 401:
            r["status"] = "invalid"
        else:
            r["status"] = f"http_{resp.status_code}"

    except httpx.TimeoutException:
        r["status"] = "timeout"
    except Exception as e:
        r["status"] = f"error"

    return r


# ── Table renderer ────────────────────────────────────────────────────────────

_STATUS_LABELS = {
    "ok":            ("🟢", "Available"),
    "tpd_exhausted": ("🔴", "TPD Exhausted"),
    "tpm_limit":     ("🟡", "TPM Limit (temp)"),
    "rate_limited":  ("🟡", "Rate Limited"),
    "invalid":       ("❌", "Invalid / Expired"),
    "timeout":       ("⚠️ ", "Timeout"),
}


def _fmt_tokens(val: int | None, fallback: str = "—") -> str:
    if val is None:
        return fallback
    return f"{val:,}"


def _render_table(results: list[dict], next_reset_ist: str) -> None:
    # ── Column widths ─────────────────────────────────────────────────────────
    C1, C2, C3, C4, C5 = 26, 15, 18, 14, 20

    def row(c1, c2, c3, c4, c5):
        print(f"  {c1:<{C1}}  {c2:<{C2}}  {c3:<{C3}}  {c4:<{C4}}  {c5:<{C5}}")

    sep = "  " + "─" * (C1 + C2 + C3 + C4 + C5 + 8)

    # ── Header ────────────────────────────────────────────────────────────────
    print(sep)
    row("Key ID", "Used Tokens", "Remaining Tokens", "Reset (IST)", "Status")
    print(sep)

    # ── Rows ──────────────────────────────────────────────────────────────────
    for r in results:
        icon, label = _STATUS_LABELS.get(r["status"], ("⚠️ ", r["status"]))

        used_str = _fmt_tokens(r["tpd_used"])
        if r["status"] == "ok":
            used_str = "< 100,000"       # confirmed < limit but exact unknown
            rem_str  = "> 0 (fresh)"     # confirmed working, exact unknown
        elif r["status"] == "tpd_exhausted":
            used_str = _fmt_tokens(r["tpd_used"], "~100,000")
            rem_str  = "0"
        elif r["status"] == "tpm_limit":
            used_str = "< 100,000"
            rem_str  = "> 0 (TPM pause)"
        elif r["status"] == "invalid":
            used_str = "—"
            rem_str  = "—"
        else:
            used_str = "—"
            rem_str  = "—"

        retry = f"  retry in {r['retry_in']}" if r.get("retry_in") else ""
        status_str = f"{icon} {label}{retry}"

        row(r["preview"], used_str, rem_str, next_reset_ist, status_str)

    print(sep)


# ── Summary stats ─────────────────────────────────────────────────────────────

def _render_summary(results: list[dict]) -> None:
    n_ok       = sum(1 for r in results if r["status"] == "ok")
    n_tpd      = sum(1 for r in results if r["status"] == "tpd_exhausted")
    n_tpm      = sum(1 for r in results if r["status"] == "tpm_limit")
    n_invalid  = sum(1 for r in results if r["status"] == "invalid")
    n_error    = sum(1 for r in results if r["status"] in ("timeout", "error") or r["status"].startswith("http_"))

    confirmed_used = sum(r["tpd_used"] for r in results if r.get("tpd_used") is not None)

    print(f"  🟢 Available          : {n_ok} keys")
    print(f"  🔴 TPD Exhausted      : {n_tpd} keys")
    if n_tpm:
        print(f"  🟡 TPM Limit (temp)   : {n_tpm} keys  (TPD may still be available)")
    if n_invalid:
        print(f"  ❌ Invalid / Expired  : {n_invalid} keys")
    if n_error:
        print(f"  ⚠️  Errors / Timeouts  : {n_error} keys")

    print()

    # Token budget estimate
    if n_tpd > 0:
        confirmed_rem = sum(r["tpd_rem"] for r in results if r.get("tpd_rem") is not None and r["tpd_rem"] >= 0)
        est_rem_from_ok = n_ok * TPD_LIMIT  # upper bound for available keys
        print(f"  Confirmed exhausted   : {confirmed_used:>10,} tokens  ({n_tpd} key(s) at limit)")
        print(f"  Max est. remaining    : {est_rem_from_ok:>10,} tokens  ({n_ok} available key(s) × 100k)")
    else:
        est_max = n_ok * TPD_LIMIT
        print(f"  All {n_ok} keys available — max possible remaining: ~{est_max:,} tokens")
        print(f"  (Exact TPD per key unknown — Groq only reveals this on 429 responses)")

    print()

    # Feasibility guide
    tasks = [
        ("No-RAG variants — generation (all 5 × 50 queries)", 150_000),
        ("No-RAG variants — RAGAS scoring (all 5, K=0)",      250_000),
        ("RAG generation  — one experiment, one K value",     125_000),
        ("RAGAS scoring   — one experiment, one K value",     600_000),
    ]
    print("  CAN YOU RUN THIS NOW?")
    print("  " + "─" * 58)
    avail_budget = n_ok * TPD_LIMIT
    for task, cost in tasks:
        keys_needed = -(-cost // TPD_LIMIT)
        if n_ok == 0:
            verdict = "❌  No available keys"
        elif avail_budget >= cost:
            verdict = f"✅  Yes  (~{cost:,} tokens, needs ~{keys_needed} key(s))"
        else:
            verdict = f"⚠️   Maybe — cost {cost:,} tokens, have ~{avail_budget:,}"
        print(f"  {task}")
        print(f"       {verdict}")
    print()


# ── Public entry point ────────────────────────────────────────────────────────

def check_all_keys() -> list[dict]:
    """Probe all configured Groq keys and print a clean status table."""
    try:
        keys = get_all_groq_keys()
    except EnvironmentError as e:
        print(f"❌  Could not load keys: {e}")
        return []

    now_utc        = datetime.now(timezone.utc)
    now_ist        = now_utc + IST_OFFSET
    next_reset_utc = (now_utc + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    next_reset_ist  = next_reset_utc + IST_OFFSET
    hours_to_reset  = (next_reset_utc - now_utc).total_seconds() / 3600
    reset_str       = next_reset_ist.strftime("%H:%M IST")

    print()
    print("=" * 72)
    print("  GROQ KEY STATUS  —  " + now_ist.strftime("%d %b %Y  %H:%M:%S IST"))
    print(f"  TPD resets at {reset_str}  ({hours_to_reset:.1f} hrs from now)  │  {len(keys)} keys loaded")
    print("=" * 72)
    print()

    results = []
    for i, key in enumerate(keys, start=1):
        print(f"  Probing {i}/{len(keys)}...", end="\r", flush=True)
        results.append(_probe(key, i))
        time.sleep(0.3)
    print(" " * 30, end="\r")

    _render_table(results, reset_str)
    print()
    _render_summary(results)
    print("=" * 72)
    print()

    return results


def key_status_dataframe():
    """Return a styled pandas DataFrame (for notebook display)."""
    import pandas as pd
    results = check_all_keys.__wrapped__(check_all_keys) if hasattr(check_all_keys, "__wrapped__") else None
    # Re-probe silently
    try:
        keys = get_all_groq_keys()
    except EnvironmentError:
        return pd.DataFrame()

    now_utc       = datetime.now(timezone.utc)
    next_reset_utc = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    next_reset_ist = (next_reset_utc + IST_OFFSET).strftime("%H:%M IST")

    rows = []
    for i, key in enumerate(keys, start=1):
        r = _probe(key, i)
        _, label = _STATUS_LABELS.get(r["status"], ("", r["status"]))
        if r["status"] == "ok":
            used_str = "< 100,000"
            rem_str  = "> 0 (fresh)"
        elif r["status"] == "tpd_exhausted":
            used_str = f"{r['tpd_used']:,}" if r["tpd_used"] else "~100,000"
            rem_str  = "0"
        else:
            used_str = "—"
            rem_str  = "—"
        rows.append({
            "Key ID":            r["preview"],
            "Used Tokens":       used_str,
            "Remaining Tokens":  rem_str,
            "Reset (IST)":       next_reset_ist,
            "Status":            label,
        })

    df = pd.DataFrame(rows)

    def _colour(val):
        v = str(val)
        if "Exhausted" in v or "Invalid" in v:
            return "background-color:#fee2e2;color:#991b1b"
        if "TPM" in v or "Limited" in v:
            return "background-color:#fef9c3;color:#854d0e"
        if "Available" in v or "fresh" in v:
            return "background-color:#dcfce7;color:#166534"
        return ""

    return df.style.map(_colour, subset=["Status"])


if __name__ == "__main__":
    check_all_keys()
