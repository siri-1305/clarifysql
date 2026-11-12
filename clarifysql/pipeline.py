"""End-to-end pipeline:
question -> ambiguity check -> schema retrieval -> SQL generation -> validate & repair -> explanation + trust badge.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .ambiguity import AmbiguityResult, assess
from .config import PipelineOptions, Settings
from .db import Database, SQLExecError, canonical_sql
from .explain import Explanation, explain
from .generator import baseline_sql, generate_sql, repair_sql
from .llm import BaseLLM, GroqLLM, LLMError
from .metrics import results_match
from .retrieval import RetrievalResult, SchemaIndex
from .trust import TrustResult, TrustSignals, compute_trust
from .validator import validate

log = logging.getLogger("clarifysql.pipeline")


@dataclass
class Attempt:
    sql: str
    errors: List[str]
    suspicious: Optional[str] = None


@dataclass
class PipelineResult:
    status: str = "error"                  # answer | clarify | unanswerable | error
    question: str = ""
    resolved_question: str = ""
    message: str = ""                      # explanation for unanswerable / error
    ambiguity: Optional[AmbiguityResult] = None
    clarification_question: str = ""
    options: List[str] = field(default_factory=list)
    retrieval: Optional[RetrievalResult] = None
    retrieval_expanded: bool = False
    sql: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    rows: List[tuple] = field(default_factory=list)
    truncated: bool = False
    validated: bool = False
    repairs: int = 0
    attempts: List[Attempt] = field(default_factory=list)
    agreement: Optional[float] = None
    n_candidates: int = 0
    explanation: Optional[Explanation] = None
    trust: Optional[TrustResult] = None
    llm_calls: int = 0
    seconds: float = 0.0

    def to_log(self) -> Dict[str, Any]:
        """Compact JSON-friendly view used by evaluate.py."""
        return {
            "status": self.status, "sql": self.sql, "message": self.message, "repairs": self.repairs,
            "validated": self.validated, "agreement": self.agreement, "n_rows": len(self.rows),
            "rows_preview": [list(map(str, r)) for r in self.rows[:5]],
            "ambiguity": self.ambiguity.score if self.ambiguity else None,
            "retrieval_tables": self.retrieval.tables if self.retrieval else None,
            "retrieval_confidence": round(self.retrieval.confidence, 3) if self.retrieval else None,
            "retrieval_expanded": self.retrieval_expanded,
            "trust_level": self.trust.level if self.trust else None,
            "trust_score": self.trust.score if self.trust else None,
            "llm_calls": self.llm_calls, "seconds": round(self.seconds, 2),
        }


class ClarifySQL:
    def __init__(self, db: Database, llm: BaseLLM, index: Optional[SchemaIndex] = None,
                 options: Optional[PipelineOptions] = None):
        self.db, self.llm, self.index = db, llm, index
        self.o = options or PipelineOptions()

    @classmethod
    def from_settings(cls, settings: Settings, options: Optional[PipelineOptions] = None) -> "ClarifySQL":
        db = Database(settings.db_path)
        llm = GroqLLM(settings)
        index = SchemaIndex(db, llm, settings.cache_dir)
        return cls(db, llm, index, options)

    # ------------------------------------------------------------------ public API
    def ask(self, question: str, clarification_answer: Optional[str] = None,
            clarification_question: Optional[str] = None) -> PipelineResult:
        res = PipelineResult(question=question, resolved_question=question)
        t0, c0 = time.time(), getattr(self.llm, "calls", 0)
        try:
            self._run(res, question, clarification_answer, clarification_question)
        except LLMError as e:
            res.status, res.message = "error", str(e)
        except Exception as e:  # noqa: BLE001 - the UI must never crash
            log.exception("pipeline failure")
            res.status, res.message = "error", f"Unexpected error ({type(e).__name__}): {e}"
        res.seconds = time.time() - t0
        res.llm_calls = getattr(self.llm, "calls", 0) - c0
        return res

    def baseline(self, question: str) -> PipelineResult:
        """Plain LLM question-to-SQL with the whole schema and no checks (comparison baseline)."""
        res = PipelineResult(question=question, resolved_question=question)
        t0, c0 = time.time(), getattr(self.llm, "calls", 0)
        try:
            draft = baseline_sql(self.llm, question, self.db.schema_text())
            if draft.sql is None:
                res.status, res.message = "unanswerable", draft.unanswerable_reason or ""
            else:
                res.sql = draft.sql
                try:
                    r = self.db.execute(draft.sql, self.o.max_rows, self.o.query_timeout_s)
                    res.status, res.columns, res.rows, res.truncated = "answer", r.columns, r.rows, r.truncated
                except SQLExecError as e:
                    res.status, res.message = "error", f"SQLite error: {e}"
        except LLMError as e:
            res.status, res.message = "error", str(e)
        res.seconds = time.time() - t0
        res.llm_calls = getattr(self.llm, "calls", 0) - c0
        return res

    # ------------------------------------------------------------------ internals
    def _run(self, res: PipelineResult, question: str, clar_answer: Optional[str], clar_question: Optional[str]):
        o, db, llm = self.o, self.db, self.llm
        resolved = question

        # 1) clarification-aware step ---------------------------------------------------
        if clar_answer:
            resolved = f"{question}\nClarification: {clar_question or 'Please clarify'} -> {clar_answer}"
            res.resolved_question = resolved
        elif o.use_clarification:
            amb = assess(llm, db, question, o.ambiguity_threshold)
            res.ambiguity = amb
            if not amb.answerable:
                res.status = "unanswerable"
                res.message = amb.unanswerable_reason or "The database does not contain the data needed for this question."
                return
            if amb.needs_clarification:
                res.status = "clarify"
                res.clarification_question, res.options = amb.clarifying_question, amb.options
                return

        # 2) schema retrieval (RAG) ------------------------------------------------------
        full_schema = db.schema_text()
        ret: Optional[RetrievalResult] = None
        if o.use_retrieval and self.index is not None:
            ret = self.index.retrieve(resolved, o.top_k_tables)
            schema_text = db.schema_text(ret.tables)
        else:
            schema_text = full_schema
        res.retrieval = ret
        expanded = False

        # 3) SQL generation --------------------------------------------------------------
        draft = generate_sql(llm, resolved, schema_text, sample=0)
        if draft.sql is None:
            res.status, res.message = "unanswerable", draft.unanswerable_reason or "The data needed is not in the database."
            return

        # 4) validate + bounded repair loop ------------------------------------------------
        sql, repairs, best_ok = draft.sql, 0, None
        while True:
            rep = validate(sql, db, o.max_rows, o.query_timeout_s)
            res.attempts.append(Attempt(sql, [i.message for i in rep.errors], rep.suspicious))
            if rep.ok:
                best_ok = (sql, rep)
            if not rep.needs_repair or not o.use_repair or repairs >= o.max_repairs:
                break
            if ret is not None and not expanded and any(i.code in ("unknown_table", "unknown_column") for i in rep.errors):
                schema_text, expanded = full_schema, True      # retrieval missed something: use everything
            repairs += 1
            new = repair_sql(llm, resolved, schema_text, sql, rep.diagnostics(), attempt=repairs)
            if new.sql is None:
                if best_ok is None:
                    res.status, res.message = "unanswerable", new.unanswerable_reason or "The data needed is not in the database."
                    return
                break
            if canonical_sql(new.sql) == canonical_sql(sql):
                break
            sql = new.sql

        if rep.ok:
            final_sql, final = sql, rep
        elif best_ok is not None:
            final_sql, final = best_ok
        else:
            final_sql, final = sql, rep
        res.sql, res.repairs, res.validated = final_sql, repairs, final.ok
        if ret is not None and final.tables_used and not set(final.tables_used) <= set(ret.tables):
            expanded = True
        res.retrieval_expanded = expanded

        if final.result is None:
            res.status = "error"
            res.message = "I could not build a working query for this question. Last problems found: " + \
                          (final.diagnostics().replace("\n", " ") or "unknown")
            return
        res.columns, res.rows, res.truncated = final.result.columns, final.result.rows, final.result.truncated

        # 5) agreement between independently generated queries ---------------------------
        if o.agreement_samples > 0 and final.ok:
            matches, total = 1, 1
            for i in range(1, o.agreement_samples + 1):
                try:
                    d = generate_sql(llm, resolved, schema_text, sample=i)
                except LLMError as e:
                    log.warning("agreement sample %d skipped: %s", i, e)
                    continue
                total += 1
                if d.sql is None:
                    continue
                try:
                    r = db.execute(d.sql, o.max_rows, o.query_timeout_s)
                    if results_match(r.rows, final.result.rows):
                        matches += 1
                except Exception:  # noqa: BLE001 - a failing candidate simply disagrees
                    continue
            if total > 1:
                res.agreement, res.n_candidates = matches / total, total

        # 6) plain-language explanation + trust badge -----------------------------------
        res.explanation = explain(llm, db, resolved, final_sql)
        sig = TrustSignals(repairs=repairs, validated=final.ok, agreement=res.agreement, n_candidates=res.n_candidates,
                           suspicious=final.suspicious,
                           retrieval_confidence=ret.confidence if ret else None, retrieval_expanded=expanded)
        res.trust = compute_trust(sig, o.trust_high, o.trust_medium)
        res.status = "answer"
