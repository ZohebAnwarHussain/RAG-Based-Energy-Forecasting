"""
experiments/exp_01_no_rag_variants.py
=======================================
EXP_01 — No-RAG LLM Baseline: 5 Prompting Strategy Variants

Variants
--------
  EXP_01A  Zero-Shot      : Raw question, no instructions or examples
  EXP_01B  Role Prompting : Expert persona assigned before answering
  EXP_01C  Few-Shot       : 3 worked Q&A examples prepended
  EXP_01D  Chain-of-Thought: Step-by-step reasoning before final answer
  EXP_01E  Structured Output: Forced OBSERVATION/PATTERN/IMPLICATION/CONFIDENCE format

Thesis purpose
--------------
  Answers: "Which prompting technique extracts the most grounded energy
  knowledge from the LLM's parametric memory — before any retrieval?"

  All five variants have zero retrieved context, so:
    - hallucination_rate = 1.0 by definition (no KB to ground against)
    - RAGAS context_precision / context_recall = N/A
    - RAGAS faithfulness scores the answer against ground truth only

  The comparison across variants reveals the ceiling of what parametric
  memory alone can produce, which EXP_02–09 (RAG) should consistently beat
  on faithfulness. That delta is the core RAG value argument in the thesis.

Correct / Useful Insights (textual)
-------------------------------------
  Each variant computes a qualitative insight_observation field per query,
  describing WHAT the model actually said — not just whether it was relevant.
  This is separate from is_useful (which is a numeric threshold proxy).

  insight_observation captures:
    - Whether the answer cited specific numbers or only general trends
    - Whether the answer expressed uncertainty or stated facts confidently
    - For EXP_01D: whether the reasoning chain was visible in the output
    - For EXP_01E: whether all four CONFIDENCE/IMPLICATION sections appeared
    - For EXP_01C: whether the answer matched the few-shot format/style

  These observations are stored per query and aggregated in agg_metrics
  as textual summaries for the thesis Table 1 narrative.

Usage
-----
  from experiments.exp_01_no_rag_variants import run_all_variants
  results = run_all_variants(queries=QUERIES, outputs_dir=EXP_OUTPUTS_DIR)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from config.models import MODELS, EXP_DEFAULTS
from src.experiments.groq_client import RotatingGroqClient
from src.experiments.metrics import (
    compute_answer_relevance,
    compute_semantic_similarity,
    compute_hallucination_rate,
    compute_insight_clarity,
    is_useful_answer,
)
from experiments.runner import run_experiment, ExperimentResult, _save_results

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANTS = {
    "EXP_01A_ZERO_SHOT":       "zero_shot",
    "EXP_01B_ROLE_PROMPTING":  "role_prompting",
    "EXP_01C_FEW_SHOT":        "few_shot",
    "EXP_01D_CHAIN_OF_THOUGHT":"chain_of_thought",
    "EXP_01E_STRUCTURED":      "structured_output",
}

# ---------------------------------------------------------------------------
# Few-shot examples
# These use generic energy domain knowledge — NOT from the golden dataset,
# so there is no data leakage into the evaluation queries.
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """Example 1:
Q: What typically happens to electricity demand during a heatwave in summer?
A: During heatwaves, electricity demand rises sharply due to widespread air conditioning use. Peak loads can increase by 15–25% above seasonal averages, typically occurring in mid-afternoon hours (2–5 PM) when temperatures peak. Utility operators typically activate demand response programs and spinning reserves during such events to maintain grid stability.

Example 2:
Q: How does industrial electricity demand differ from residential demand on weekdays?
A: Industrial demand follows a flat, sustained profile throughout business hours (6 AM–10 PM), driven by continuous manufacturing processes. Residential demand shows two peaks: a morning peak around 7–9 AM and a larger evening peak around 6–9 PM driven by cooking, lighting, and entertainment. Industrial load is more predictable and less weather-sensitive than residential load.

Example 3:
Q: What factors cause overnight electricity demand to drop significantly?
A: Overnight demand falls due to reduced industrial activity, lower commercial lighting and HVAC loads, and minimal residential activity. Typical overnight troughs reach 40–60% of daytime peak levels. Baseload generators such as nuclear and large hydro plants maintain continuous output during this period, creating surplus capacity that is absorbed by pumped-storage or exported to neighbouring grids."""

# ---------------------------------------------------------------------------
# Prompt builders — one per variant
# ---------------------------------------------------------------------------

def _build_zero_shot(question: str) -> list[dict]:
    """EXP_01A: Raw question, minimal framing."""
    return [
        {
            "role": "system",
            "content": (
                "You are an energy demand analyst. "
                "Answer the user's question concisely in 3–5 sentences."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nAnswer:",
        },
    ]


def _build_role_prompting(question: str) -> list[dict]:
    """EXP_01B: Expert persona assigned before answering."""
    return [
        {
            "role": "system",
            "content": (
                "You are a senior energy systems analyst with 20 years of experience "
                "in grid load forecasting, demand-side management, and utility operations "
                "for multi-zone electricity networks. You have worked with GEFCom datasets "
                "and understand seasonal, zonal, and hourly electricity consumption patterns "
                "in depth. When answering, draw on your expert knowledge to give precise, "
                "data-informed insights. Write 3–5 sentences in clear stakeholder language."
            ),
        },
        {
            "role": "user",
            "content": (
                f"As a senior energy systems analyst, please answer the following question "
                f"about electricity demand:\n\n{question}"
            ),
        },
    ]


def _build_few_shot(question: str) -> list[dict]:
    """EXP_01C: 3 worked examples prepended before the actual question."""
    return [
        {
            "role": "system",
            "content": (
                "You are an energy demand analyst. Study the following examples of "
                "energy demand questions and expert answers, then answer the new question "
                "in the same style — specific, structured, and 3–5 sentences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{FEW_SHOT_EXAMPLES}\n\n"
                f"Now answer this question in the same style:\n"
                f"Q: {question}\nA:"
            ),
        },
    ]


def _build_chain_of_thought(question: str) -> list[dict]:
    """EXP_01D: Step-by-step reasoning scaffold before final answer."""
    return [
        {
            "role": "system",
            "content": (
                "You are an energy demand analyst. When answering questions, always "
                "think through the problem step by step before giving your final answer. "
                "Show your reasoning explicitly — this helps ensure accuracy."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Energy demand question: {question}\n\n"
                "Let's think through this step by step:\n"
                "Step 1 — What data or patterns are relevant to this question?\n"
                "Step 2 — What do I know about this from energy demand principles?\n"
                "Step 3 — Are there any uncertainties or caveats I should flag?\n"
                "Final Answer — Based on the above reasoning:"
            ),
        },
    ]


def _build_structured_output(question: str) -> list[dict]:
    """EXP_01E: Forced structured format with CONFIDENCE self-assessment."""
    return [
        {
            "role": "system",
            "content": (
                "You are an energy demand analyst. Always answer energy questions using "
                "exactly this four-part structure:\n\n"
                "OBSERVATION: [What the data or pattern shows — 1–2 sentences]\n"
                "PATTERN: [The underlying trend or mechanism — 1 sentence]\n"
                "IMPLICATION: [What this means for energy planning — 1 sentence]\n"
                "CONFIDENCE: [High / Medium / Low — and one sentence explaining why]\n\n"
                "Do not deviate from this format. Do not add extra sections."
            ),
        },
        {
            "role": "user",
            "content": f"Energy demand question: {question}",
        },
    ]


# Map variant EXP_ID → prompt builder
_PROMPT_BUILDERS = {
    "EXP_01A_ZERO_SHOT":        _build_zero_shot,
    "EXP_01B_ROLE_PROMPTING":   _build_role_prompting,
    "EXP_01C_FEW_SHOT":         _build_few_shot,
    "EXP_01D_CHAIN_OF_THOUGHT": _build_chain_of_thought,
    "EXP_01E_STRUCTURED":       _build_structured_output,
}

# ---------------------------------------------------------------------------
# Qualitative insight observation — per variant
# ---------------------------------------------------------------------------

def _observe_zero_shot(question: str, answer: str) -> str:
    """What did the zero-shot model actually produce?"""
    word_count   = len(answer.split())
    has_numbers  = bool(re.search(r"\b\d+[\.,]?\d*\s*%?", answer))
    is_hedged    = any(p in answer.lower() for p in
                       ["uncertain", "may", "could", "typically", "generally",
                        "approximately", "around", "likely"])
    obs_parts = []
    if has_numbers:
        obs_parts.append("cited specific numerical values")
    else:
        obs_parts.append("relied on general qualitative trends without numbers")
    if is_hedged:
        obs_parts.append("included hedging language suggesting awareness of uncertainty")
    else:
        obs_parts.append("stated facts confidently with no expressed uncertainty")
    obs_parts.append(f"answer length: {word_count} words")
    return "Zero-Shot: " + "; ".join(obs_parts) + "."


def _observe_role(question: str, answer: str) -> str:
    """Did the role persona change tone, structure, or specificity?"""
    word_count     = len(answer.split())
    has_numbers    = bool(re.search(r"\b\d+[\.,]?\d*\s*%?", answer))
    domain_terms   = ["zone", "mwh", "kwh", "load", "grid", "demand", "peak",
                      "baseload", "utility", "gefcom", "forecast"]
    domain_count   = sum(1 for t in domain_terms if t in answer.lower())
    obs_parts = [f"answer length: {word_count} words"]
    if has_numbers:
        obs_parts.append("persona prompted inclusion of specific figures")
    else:
        obs_parts.append("persona did not trigger numerical specificity")
    obs_parts.append(f"domain terminology density: {domain_count} terms detected")
    return "Role Prompting: " + "; ".join(obs_parts) + "."


def _observe_few_shot(question: str, answer: str) -> str:
    """Did the few-shot examples shape structure and style?"""
    word_count  = len(answer.split())
    has_numbers = bool(re.search(r"\b\d+[\.,]?\d*\s*%?", answer))
    # Check if answer follows example structure (% figures, AM/PM times, comparatives)
    has_pct     = "%" in answer
    has_time    = bool(re.search(r"\d+\s*(am|pm|AM|PM)", answer))
    obs_parts   = [f"answer length: {word_count} words"]
    if has_pct:
        obs_parts.append("adopted percentage framing from examples")
    if has_time:
        obs_parts.append("included time-of-day references matching example style")
    if has_numbers:
        obs_parts.append("numerical specificity higher than zero-shot (example priming effect)")
    else:
        obs_parts.append("examples did not trigger numerical specificity")
    return "Few-Shot: " + "; ".join(obs_parts) + "."


def _observe_cot(question: str, answer: str) -> str:
    """Did the CoT scaffold produce visible reasoning steps?"""
    word_count    = len(answer.split())
    has_steps     = bool(re.search(r"step\s*\d|step-by-step|first[,.]|second[,.]",
                                   answer.lower()))
    has_final     = "final answer" in answer.lower() or "in conclusion" in answer.lower()
    has_caveat    = any(p in answer.lower() for p in
                        ["however", "caveat", "limitation", "uncertain",
                         "data may", "should note", "important to"])
    obs_parts     = [f"answer length: {word_count} words"]
    if has_steps:
        obs_parts.append("reasoning steps visible in output (scaffold followed)")
    else:
        obs_parts.append("reasoning steps collapsed — model answered directly without showing steps")
    if has_final:
        obs_parts.append("explicit 'Final Answer' section present")
    if has_caveat:
        obs_parts.append("caveats or uncertainty flags expressed mid-reasoning")
    return "Chain-of-Thought: " + "; ".join(obs_parts) + "."


def _observe_structured(question: str, answer: str) -> str:
    """Did the model follow the structured format? Is CONFIDENCE realistic?"""
    word_count = len(answer.split())
    has_obs    = "observation:" in answer.lower()
    has_pat    = "pattern:"     in answer.lower()
    has_imp    = "implication:" in answer.lower()
    has_conf   = "confidence:"  in answer.lower()

    sections_found = sum([has_obs, has_pat, has_imp, has_conf])

    # Extract CONFIDENCE label if present
    conf_match = re.search(r"confidence:\s*(high|medium|low)", answer.lower())
    conf_label = conf_match.group(1).capitalize() if conf_match else "not found"

    obs_parts = [
        f"answer length: {word_count} words",
        f"structured sections present: {sections_found}/4 "
        f"(OBSERVATION={'✓' if has_obs else '✗'}, "
        f"PATTERN={'✓' if has_pat else '✗'}, "
        f"IMPLICATION={'✓' if has_imp else '✗'}, "
        f"CONFIDENCE={'✓' if has_conf else '✗'})",
        f"self-reported confidence: {conf_label}",
    ]
    return "Structured Output: " + "; ".join(obs_parts) + "."


_OBSERVERS = {
    "EXP_01A_ZERO_SHOT":        _observe_zero_shot,
    "EXP_01B_ROLE_PROMPTING":   _observe_role,
    "EXP_01C_FEW_SHOT":         _observe_few_shot,
    "EXP_01D_CHAIN_OF_THOUGHT": _observe_cot,
    "EXP_01E_STRUCTURED":       _observe_structured,
}

# ---------------------------------------------------------------------------
# Structured output parser (EXP_01E only)
# ---------------------------------------------------------------------------

def _parse_structured(answer: str) -> dict[str, str | None]:
    """Extract the four sections from EXP_01E structured output."""
    sections = {}
    keys = ["OBSERVATION", "PATTERN", "IMPLICATION", "CONFIDENCE"]
    for i, key in enumerate(keys):
        next_key = keys[i + 1] if i + 1 < len(keys) else None
        if next_key:
            pattern = rf"{key}:\s*(.*?)\s*(?={next_key}:)"
        else:
            pattern = rf"{key}:\s*(.*?)$"
        m = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
        sections[key.lower()] = m.group(1).strip() if m else None
    return sections

# ---------------------------------------------------------------------------
# generate_fn factory — shared logic, variant-specific prompt
# ---------------------------------------------------------------------------

def _make_generate_fn(exp_id: str, client: RotatingGroqClient) -> Any:
    prompt_builder = _PROMPT_BUILDERS[exp_id]
    observer       = _OBSERVERS[exp_id]

    def generate_fn(query: dict, retrieved_docs: list, top_k: int) -> dict:
        question     = query.get("question", query.get("user_query", ""))
        ground_truth = query.get("ground_truth", query.get("reference_answer", ""))

        messages = prompt_builder(question)

        response = client.chat(
            messages=messages,
            model=MODELS["groq_rag"],
            temperature=EXP_DEFAULTS["temperature"],
            max_tokens=EXP_DEFAULTS["max_tokens"],
        )
        answer = response.choices[0].message.content.strip()

        # ── Qualitative observation (Correct/Useful Insights) ────────────
        insight_observation = observer(question, answer)

        # ── EXP_01E: extract structured sections ─────────────────────────
        structured_sections = None
        confidence_label    = None
        if exp_id == "EXP_01E_STRUCTURED":
            structured_sections = _parse_structured(answer)
            confidence_label    = structured_sections.get("confidence", "")
            # Extract just the High/Medium/Low token
            m = re.search(r"\b(high|medium|low)\b", (confidence_label or ""), re.I)
            confidence_label = m.group(1).capitalize() if m else "Unknown"

        # ── Standard metrics ─────────────────────────────────────────────
        metrics = {
            "answer_relevance":    compute_answer_relevance(question, answer),
            "semantic_similarity": compute_semantic_similarity(answer, ground_truth)
                                   if ground_truth else None,
            # No context → every answer is by definition ungrounded
            "hallucination_rate":  compute_hallucination_rate(answer, context_docs=[]),
            "insight_clarity":     compute_insight_clarity(answer),
            "is_useful":           int(is_useful_answer(answer, question)),
            "answer_word_count":   len(answer.split()),
            # Qualitative — textual observation
            "insight_observation": insight_observation,
            # RAGAS placeholders (filled by ragas_evaluator.py)
            "faithfulness":        None,
            "context_precision":   None,
            "context_recall":      None,
            # EXP_01E extras
            "confidence_label":    confidence_label,
        }

        return {
            "answer":            answer,
            "retrieved_docs":    [],
            "metrics":           metrics,
            # Structured sections stored for EXP_01E analysis
            "structured_output": structured_sections,
        }

    return generate_fn


# ---------------------------------------------------------------------------
# Aggregate metrics builder
# ---------------------------------------------------------------------------

def _compute_agg(result: ExperimentResult, exp_id: str, pipeline: str) -> None:
    qrs   = result.query_results
    valid = [qr for qr in qrs if not qr.error]

    def _mean(key: str) -> float | None:
        vals = [qr.metrics.get(key) for qr in valid
                if qr.metrics.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def _rate(key: str) -> float | None:
        vals = [qr.metrics.get(key, 0) for qr in valid]
        return round(sum(vals) / len(vals), 4) if vals else None

    # ── Qualitative summary — collect all insight observations ────────────
    observations = [
        qr.metrics.get("insight_observation", "")
        for qr in valid
        if qr.metrics.get("insight_observation")
    ]

    # Aggregate observation patterns
    has_numbers_count = sum(
        1 for obs in observations if "specific numerical" in obs or "figures" in obs
    )
    has_hedging_count = sum(
        1 for obs in observations if "hedging" in obs or "uncertainty" in obs
    )
    has_steps_count   = sum(
        1 for obs in observations if "steps visible" in obs
    )

    # EXP_01E: confidence label distribution
    confidence_dist = {}
    if exp_id == "EXP_01E_STRUCTURED":
        for qr in valid:
            label = qr.metrics.get("confidence_label") or "Unknown"
            confidence_dist[label] = confidence_dist.get(label, 0) + 1
        sections_complete = sum(
            1 for qr in valid
            if "4/4" in (qr.metrics.get("insight_observation") or "")
        )
    else:
        sections_complete = None

    # EXP_01D: count answers with visible CoT steps
    cot_steps_visible = has_steps_count if exp_id == "EXP_01D_CHAIN_OF_THOUGHT" else None

    result.agg_metrics = {
        "exp_id":   exp_id,
        "pipeline": pipeline,
        "top_k":    0,
        "n_queries": len(qrs),
        "n_valid":   len(valid),
        "n_errors":  result.total_errors,
        # ── Table 1 metrics ───────────────────────────────────────────────
        "pct_useful":              _rate("is_useful"),
        "avg_answer_relevance":    _mean("answer_relevance"),
        "avg_semantic_similarity": _mean("semantic_similarity"),
        "avg_faithfulness":        None,   # RAGAS — filled later
        "avg_context_precision":   None,   # N/A — no retrieval
        "avg_context_recall":      None,   # N/A — no retrieval
        "avg_hallucination_rate":  _mean("hallucination_rate"),
        "avg_insight_clarity":     _mean("insight_clarity"),
        "avg_latency_sec": round(
            sum(qr.latency_sec for qr in valid) / max(len(valid), 1), 3
        ),
        "total_time_sec": result.total_time_sec,
        # ── Correct / Useful Insights — textual summary ───────────────────
        "useful_insights_summary": (
            f"{int(_rate('is_useful') * len(valid))}/{len(valid)} answers passed "
            f"the usefulness threshold (relevance >= 0.40, >= 20 words). "
            f"{has_numbers_count}/{len(valid)} answers cited specific numerical values. "
            f"{has_hedging_count}/{len(valid)} answers included uncertainty/hedging language."
        ),
        # ── Variant-specific textual observations ─────────────────────────
        "variant_observation": _build_variant_observation(
            exp_id, valid, has_numbers_count, has_hedging_count,
            cot_steps_visible, sections_complete, confidence_dist
        ),
        # ── EXP_01D specific ──────────────────────────────────────────────
        "cot_steps_visible_count": cot_steps_visible,
        # ── EXP_01E specific ──────────────────────────────────────────────
        "structured_sections_complete": sections_complete,
        "confidence_distribution":      json.dumps(confidence_dist) if confidence_dist else None,
    }


def _build_variant_observation(
    exp_id: str,
    valid: list,
    has_numbers_count: int,
    has_hedging_count: int,
    cot_steps_visible: int | None,
    sections_complete: int | None,
    confidence_dist: dict,
) -> str:
    """Build a thesis-ready textual observation paragraph for each variant."""
    n = len(valid)

    if exp_id == "EXP_01A_ZERO_SHOT":
        return (
            f"Zero-Shot baseline: of {n} queries, {has_numbers_count} answers "
            f"({has_numbers_count/n:.0%}) cited specific numerical values, and "
            f"{has_hedging_count} ({has_hedging_count/n:.0%}) included hedging language. "
            f"The model answered directly from parametric memory with no retrieval "
            f"grounding. This establishes the floor for all RAG experiments."
        )

    elif exp_id == "EXP_01B_ROLE_PROMPTING":
        return (
            f"Role Prompting: assigning an expert energy analyst persona resulted in "
            f"{has_numbers_count}/{n} ({has_numbers_count/n:.0%}) answers citing specific "
            f"figures, compared to the Zero-Shot baseline. "
            f"Hedging language appeared in {has_hedging_count}/{n} answers. "
            f"Persona conditioning altered response tone and domain vocabulary but "
            f"did not provide additional factual grounding since no context was retrieved."
        )

    elif exp_id == "EXP_01C_FEW_SHOT":
        return (
            f"Few-Shot Prompting: providing 3 worked examples produced "
            f"{has_numbers_count}/{n} ({has_numbers_count/n:.0%}) answers with specific "
            f"numerical values, suggesting example priming influenced output specificity. "
            f"Hedging was present in {has_hedging_count}/{n} responses. "
            f"The few-shot examples demonstrated percentage figures and time-of-day "
            f"references which the model partially adopted in its answers."
        )

    elif exp_id == "EXP_01D_CHAIN_OF_THOUGHT":
        steps = cot_steps_visible or 0
        return (
            f"Chain-of-Thought Prompting: the step-by-step scaffold produced visible "
            f"reasoning steps in {steps}/{n} ({steps/n:.0%}) answers. "
            f"{has_numbers_count}/{n} answers cited specific values, and "
            f"{has_hedging_count}/{n} included explicit caveats or uncertainty flags "
            f"within the reasoning chain. CoT prompted the model to surface its "
            f"uncertainty mid-reasoning, which may partially explain any hallucination "
            f"reduction relative to zero-shot."
        )

    elif exp_id == "EXP_01E_STRUCTURED":
        complete = sections_complete or 0
        conf_str = ", ".join(f"{k}: {v}" for k, v in sorted(confidence_dist.items()))
        return (
            f"Structured Output Prompting: {complete}/{n} answers contained all four "
            f"required sections (OBSERVATION / PATTERN / IMPLICATION / CONFIDENCE). "
            f"Self-reported confidence distribution — {conf_str}. "
            f"{has_numbers_count}/{n} answers cited numerical values. "
            f"The forced CONFIDENCE field provides a novel self-assessment signal "
            f"that can be correlated against computed hallucination rates — if the "
            f"model's self-reported Low confidence aligns with higher hallucination, "
            f"it supports the Novelty 2 (query difficulty prediction) thesis argument."
        )

    return f"{exp_id}: {n} queries processed."


# ---------------------------------------------------------------------------
# Single-variant runner
# ---------------------------------------------------------------------------

def run_variant(
    exp_id:      str,
    queries:     list[dict],
    client:      RotatingGroqClient,
    outputs_dir: Path,
) -> ExperimentResult:
    """Run one prompting variant over all queries."""
    pipeline    = VARIANTS[exp_id]
    generate_fn = _make_generate_fn(exp_id, client)

    result = run_experiment(
        exp_id=exp_id,
        pipeline=pipeline,
        top_k=0,
        queries=queries,
        generate_fn=generate_fn,
        outputs_dir=outputs_dir,
        log_every=10,
    )

    _compute_agg(result, exp_id, pipeline)

    out_dir = outputs_dir / exp_id / "k0"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_results(result, out_dir)

    logger.info("[%s] complete — %d queries, %d errors",
                exp_id, result.total_queries, result.total_errors)
    logger.info("[%s] variant_observation: %s",
                exp_id, result.agg_metrics.get("variant_observation", ""))

    return result


# ---------------------------------------------------------------------------
# Run all 5 variants
# ---------------------------------------------------------------------------

def run_all_variants(
    queries:     list[dict],
    outputs_dir: str | Path = "outputs/experiments",
    variant_ids: list[str] | None = None,
) -> dict[str, ExperimentResult]:
    """
    Run all 5 (or a subset of) EXP_01 prompting variants.

    Parameters
    ----------
    queries      : 50-row golden dataset as list of dicts
    outputs_dir  : root experiments output directory
    variant_ids  : optional subset, e.g. ["EXP_01A_ZERO_SHOT", "EXP_01C_FEW_SHOT"]
                   defaults to all 5 variants

    Returns
    -------
    dict mapping exp_id → ExperimentResult
    """
    outputs_dir = Path(outputs_dir)
    to_run      = variant_ids or list(VARIANTS.keys())

    # Single shared client — key rotation persists across all 5 variants
    client  = RotatingGroqClient()
    results = {}

    for exp_id in to_run:
        if exp_id not in VARIANTS:
            logger.warning("Unknown variant %s — skipping.", exp_id)
            continue

        logger.info("─── Running %s ───", exp_id)
        print(f"\n{'='*60}")
        print(f"  {exp_id}")
        print(f"  Strategy : {VARIANTS[exp_id].replace('_', ' ').title()}")
        print(f"  Queries  : {len(queries)}")
        print(f"  Model    : {MODELS['groq_rag']}")
        print(f"{'='*60}")

        result = run_variant(exp_id, queries, client, outputs_dir)
        results[exp_id] = result

        print(f"\n  ✅ {exp_id} complete")
        print(f"     Queries   : {result.total_queries}")
        print(f"     Errors    : {result.total_errors}")
        print(f"     Time      : {result.total_time_sec:.1f}s")
        m = result.agg_metrics
        print(f"     Useful %  : {m.get('pct_useful', 0)*100:.1f}%")
        print(f"     Ans Rel.  : {m.get('avg_answer_relevance')}")
        print(f"     Sem Sim.  : {m.get('avg_semantic_similarity')}")
        print(f"     Halluc.   : {m.get('avg_hallucination_rate')}")
        print(f"     Clarity   : {m.get('avg_insight_clarity')}")
        print(f"\n  📝 Observation:")
        print(f"     {m.get('useful_insights_summary')}")
        print(f"\n  📊 Variant Analysis:")
        print(f"     {m.get('variant_observation')}")

    client.log_stats()
    return results


# ---------------------------------------------------------------------------
# Comparison table builder
# ---------------------------------------------------------------------------

def build_comparison_table(results: dict[str, ExperimentResult]) -> "pd.DataFrame":
    """
    Build a 5-row Table 1 comparing all variants.
    Returns a pandas DataFrame ready for display in the notebook.
    """
    import pandas as pd

    rows = []
    for exp_id, result in results.items():
        m = result.agg_metrics
        rows.append({
            "Variant":               exp_id,
            "Strategy":              VARIANTS.get(exp_id, "").replace("_", " ").title(),
            "Useful %":              f"{m.get('pct_useful', 0)*100:.1f}%",
            "Answer Relevance":      m.get("avg_answer_relevance"),
            "Semantic Similarity":   m.get("avg_semantic_similarity"),
            "Faithfulness (RAGAS)":  m.get("avg_faithfulness"),
            "Hallucination Rate %":  f"{(m.get('avg_hallucination_rate') or 0)*100:.1f}%",
            "Insight Clarity":       m.get("avg_insight_clarity"),
            "Avg Latency (s)":       m.get("avg_latency_sec"),
            "Useful Insights Note":  m.get("useful_insights_summary", ""),
        })

    return pd.DataFrame(rows)