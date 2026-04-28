"""Pipeline-level constants.

Defines tunable parameters that control pipeline behaviour. Changing
these values affects multiple stages, so they are centralised here
rather than scattered across modules.
"""

from typing import Dict


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base Generation Limits
# ─────────────────────────────────────────────────────────────────────────────
# Maximum number of summary rows generated per type. Set to None for a full
# production run. The pilot scale of 50 mirrors the golden dataset size for
# methodology consistency.
#
# Expected outputs at MAX_SUMMARIES_PER_TYPE = 50:
#   GEFCom    daily, weekly, monthly, seasonal, system_level → 50 each
#   Household daily, weekly, monthly, appliance              → 50 each
#                                                  yearly    →  5 (only 5 years)
#   Total: ~480 summaries
MAX_SUMMARIES_PER_TYPE: int = 50


# ─────────────────────────────────────────────────────────────────────────────
# API Rate Limiting
# ─────────────────────────────────────────────────────────────────────────────
# Delay between consecutive API calls to respect free-tier rate limits.
# Gemini free tier allows 15 requests/minute. 4.5s gives ~13/min — safely under.
REQUEST_DELAY_SECONDS: float = 4.5

# Maximum number of retry attempts when a transient API error occurs (e.g. 503).
MAX_RETRIES: int = 3

# Base backoff duration in seconds. Wait time doubles on each retry:
# attempt 1 fail → wait 30s, attempt 2 fail → wait 60s.
RETRY_BACKOFF_SECONDS: float = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# Season Mapping
# ─────────────────────────────────────────────────────────────────────────────
# Meteorological season assignment by calendar month.
# Used by both GEFCom seasonal aggregation and household yearly peak detection.
SEASON_MAP: Dict[int, str] = {
    12: "Winter", 1: "Winter",  2: "Winter",
    3:  "Spring", 4: "Spring",  5: "Spring",
    6:  "Summer", 7: "Summer",  8: "Summer",
    9:  "Autumn", 10: "Autumn", 11: "Autumn",
}
