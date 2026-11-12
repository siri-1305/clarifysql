"""Ambiguity scoring: ask a follow-up only when the score is high."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .db import Database
from .llm import BaseLLM, LLMError, parse_json

# Vague evaluative words and the metric words that make them precise ("best rated", "top by revenue").
VAGUE_WORDS = {"best", "top", "popular", "recent", "recently", "good", "great", "expensive", "cheap", "low",
               "active", "inactive", "performs", "performing", "biggest", "major", "trending", "important",
               "worst", "bad", "high", "latest"}
METRIC_WORDS = {"rating", "rated", "ratings", "sales", "sold", "revenue", "profit", "price", "priced", "spend",
                "spent", "units", "quantity", "margin", "cost", "stock", "orders", "count", "number"}

SYSTEM = """You are the query-understanding module of a Text-to-SQL system for a non-technical user.
You decide whether a question can be answered by ONE reasonable SQL query without guessing a key business definition.
Reply with a single JSON object and nothing else."""

PROMPT = """DATABASE (tables and columns):
{schema}

USER QUESTION: {question}

Return JSON with exactly these keys:
{{
  "answerable": true or false,        // false only if the database has NO data that could answer it
  "unanswerable_reason": "short reason, or empty string",
  "ambiguity": a number from 0.0 to 1.0,
  "reason": "one short sentence",
  "clarifying_question": "one short question to the user, or empty string",
  "options": ["2 to 4 short answer choices the user can pick, or empty list"]
}}

Scoring guide for "ambiguity":
- 0.0-0.2: fully specified (explicit filter, count, name, year or metric).
- 0.3-0.5: minor wording choices that would not change the numbers much.
- 0.6-1.0: a vague term such as best, top, popular, recent, active, low, good, expensive, "performs well" with NO metric or
  threshold, or several plausible readings that would give clearly different results.
Do NOT count as ambiguity: synonyms of stored values (e.g. "completed" for delivered, "Bangalore" for Bengaluru),
words like "users" or "items" that map obviously to a table, or questions that give an explicit number/metric.
Write the clarifying question and options in plain language for a shop manager (no SQL words)."""


@dataclass
class AmbiguityResult:
    score: float
    llm_score: float
    lexical: float
    answerable: bool = True
    unanswerable_reason: str = ""
    reason: str = ""
    clarifying_question: str = ""
    options: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    source: str = "llm"


def lexical_vagueness(question: str) -> float:
    """1.0 when the question has a vague evaluative word but no metric, number or explicit threshold."""
    toks = set(re.findall(r"[a-z0-9]+", question.lower()))
    if not toks & VAGUE_WORDS:
        return 0.0
    if toks & METRIC_WORDS or any(t.isdigit() for t in toks):
        return 0.0
    return 1.0


def assess(llm: BaseLLM, db: Database, question: str, threshold: float) -> AmbiguityResult:
    lex = lexical_vagueness(question)
    data = None
    try:
        raw = llm.generate(PROMPT.format(schema=db.schema_summary(), question=question),
                           system=SYSTEM, json_mode=True)
        data = parse_json(raw)
    except LLMError:
        raise
    if data is None:  # unreadable reply: fall back to the lexical cue alone
        score = lex
        return AmbiguityResult(score=score, llm_score=lex, lexical=lex, source="lexical-fallback",
                               needs_clarification=score >= threshold,
                               clarifying_question="Could you tell me more precisely what you mean?" if score >= threshold else "")
    try:
        llm_score = float(data.get("ambiguity", 0.0))
    except (TypeError, ValueError):
        llm_score = 0.0
    llm_score = min(1.0, max(0.0, llm_score))
    score = round(0.8 * llm_score + 0.2 * lex, 3)
    opts = [str(o).strip() for o in (data.get("options") or []) if str(o).strip()][:4]
    cq = str(data.get("clarifying_question") or "").strip()
    answerable = bool(data.get("answerable", True))
    needs = answerable and score >= threshold
    if needs and not cq:
        cq = "Could you be more specific about what you mean?"
    return AmbiguityResult(score=score, llm_score=llm_score, lexical=lex, answerable=answerable,
                           unanswerable_reason=str(data.get("unanswerable_reason") or "").strip(),
                           reason=str(data.get("reason") or "").strip(), clarifying_question=cq,
                           options=opts, needs_clarification=needs)
