"""SQL generation, repair and the plain 'LLM-to-SQL' baseline."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .llm import BaseLLM

SYSTEM_SQL = """You are an expert SQLite analyst who translates business questions into SQL.
Rules:
1. Output exactly ONE read-only SQLite SELECT statement (a WITH ... SELECT is fine). Never modify data.
2. Use ONLY the tables and columns listed in the schema. Never invent columns.
3. Text comparisons are case-sensitive in SQLite: use the exact stored values shown in the schema, in single quotes.
4. Dates are TEXT in ISO format YYYY-MM-DD. Compare with >= and < on dates. "This year"/"recent"/"last N days" are relative to the
   latest date in the data, not today's date.
5. If a number must be computed (revenue, order total, margin) derive it from the columns that exist (e.g. quantity * unit_price).
6. If the database cannot answer the question because the data does not exist, output exactly: UNANSWERABLE: <short reason>
7. Output only the SQL (or the UNANSWERABLE line). No explanations, no markdown."""

GEN_PROMPT = """SCHEMA:
{schema}

QUESTION: {question}

SQL:"""

REPAIR_PROMPT = """SCHEMA:
{schema}

QUESTION: {question}

The SQL below was produced for this question but has problems.

SQL:
{sql}

PROBLEMS FOUND BY THE CHECKER:
{diagnostics}

Write a corrected SQL statement that fixes every problem and still answers the question.
If the data truly cannot answer the question, output exactly: UNANSWERABLE: <short reason>
Output only the SQL."""


@dataclass
class Draft:
    sql: Optional[str]
    unanswerable_reason: Optional[str] = None


def extract_sql(text: str) -> Draft:
    t = (text or "").strip()
    fence = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", t, re.S | re.I)
    if fence:
        t = fence.group(1).strip()
    m = re.match(r"\s*UNANSWERABLE\s*:?\s*(.*)", t, re.I | re.S)
    if m:
        return Draft(None, (m.group(1).strip().splitlines() or ["The data needed is not in the database."])[0])
    start = re.search(r"\b(SELECT|WITH)\b", t, re.I)
    if start:
        t = t[start.start():]
    t = t.strip().rstrip(";").strip()
    if not t:
        return Draft(None, "The model returned no SQL.")
    return Draft(t + ";")


def generate_sql(llm: BaseLLM, question: str, schema_text: str, sample: int = 0) -> Draft:
    raw = llm.generate(GEN_PROMPT.format(schema=schema_text, question=question), system=SYSTEM_SQL, sample=sample)
    return extract_sql(raw)


def repair_sql(llm: BaseLLM, question: str, schema_text: str, sql: str, diagnostics: str, attempt: int = 1) -> Draft:
    raw = llm.generate(REPAIR_PROMPT.format(schema=schema_text, question=question, sql=sql, diagnostics=diagnostics),
                       system=SYSTEM_SQL, sample=100 + attempt)
    return extract_sql(raw)


def baseline_sql(llm: BaseLLM, question: str, full_schema_text: str) -> Draft:
    """Plain LLM question-to-SQL: whole schema, one shot, no checks (the comparison baseline)."""
    raw = llm.generate(GEN_PROMPT.format(schema=full_schema_text, question=question), system=SYSTEM_SQL, sample=0)
    return extract_sql(raw)
