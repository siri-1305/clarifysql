"""Plain-language explanation of how the system understood the question."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from .db import Database
from .llm import BaseLLM, LLMError, parse_json
from .validator import HAS_SQLGLOT

try:
    import sqlglot
    from sqlglot import exp
except ImportError:  # pragma: no cover
    sqlglot = exp = None

log = logging.getLogger("clarifysql.explain")

SYSTEM = "You explain database queries to non-technical shop managers. Reply with one JSON object only."
PROMPT = """QUESTION: {question}

SQL THAT WAS RUN:
{sql}

Explain in plain English how the question was understood. Return JSON:
{{
  "summary": "2-3 short sentences: what data was looked at, which filters were applied, and how the answer was calculated. No SQL words, no table or column names in code style.",
  "assumptions": ["each guess made about a vague word or missing detail, e.g. 'Counted only delivered orders'. Empty list if none."]
}}"""


@dataclass
class Explanation:
    summary: str
    tables: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)


def structure_of(sql: str, db: Database):
    """Tables and filter conditions read straight from the SQL (no LLM involved)."""
    tables, filters = [], []
    if not HAS_SQLGLOT:
        return [t for t in db.tables if t.lower() in sql.lower()], filters
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        for t in tree.find_all(exp.Table):
            real = db.table_lookup(t.name) if t.name else None
            if real and real not in tables:
                tables.append(real)
        for w in tree.find_all(exp.Where):
            text = w.this.sql(dialect="sqlite")
            if text and text not in filters:
                filters.append(text[:200])
    except Exception as e:  # noqa: BLE001
        log.warning("could not read SQL structure: %s", e)
    return tables, filters


def explain(llm: BaseLLM, db: Database, question: str, sql: str) -> Explanation:
    tables, filters = structure_of(sql, db)
    summary, assumptions = "", []
    try:
        data = parse_json(llm.generate(PROMPT.format(question=question, sql=sql), system=SYSTEM, json_mode=True))
        if data:
            summary = str(data.get("summary") or "").strip()
            assumptions = [str(a).strip() for a in (data.get("assumptions") or []) if str(a).strip()]
    except LLMError as e:
        log.warning("explanation LLM call failed: %s", e)
    if not summary:
        summary = ("I looked at the " + ", ".join(tables) + " data" if tables else "I queried the database") + \
                  (f" and applied these conditions: {'; '.join(filters)}." if filters else ".")
    return Explanation(summary=summary, tables=tables, filters=filters, assumptions=assumptions)
