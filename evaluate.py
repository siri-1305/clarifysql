"""Evaluation + ablation study for ClarifySQL.

Examples
    python evaluate.py --limit 3                      # quick trial: 3 questions per type
    python evaluate.py                                # everything (baseline, full system, 3 ablations)
    python evaluate.py --modes baseline,full          # only some modes

Modes
    baseline      plain LLM question-to-SQL (whole schema, no checks)
    full          ClarifySQL with every module
    no_clarify    ClarifySQL without the clarification module
    no_retrieval  ClarifySQL without schema retrieval (whole schema is given to the LLM)
    no_repair     ClarifySQL without the validate-and-repair loop

Vague questions are tested like this: a modes that can clarify gets the question, and if it asks a follow-up we
answer it once with EACH valid interpretation (simulated user) and check the SQL against that interpretation.
Modes that cannot clarify get credit if their answer matches ANY valid interpretation (generous to them).
Correctness = same result rows as the reference SQL (column order and row order ignored; extra columns allowed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List

from clarifysql.config import PipelineOptions, Settings
from clarifysql.db import Database, SQLExecError
from clarifysql.llm import GeminiLLM, LLMError
from clarifysql.metrics import auroc, results_match
from clarifysql.pipeline import ClarifySQL
from clarifysql.retrieval import SchemaIndex

ALL_MODES = ["baseline", "full", "no_clarify", "no_retrieval", "no_repair"]


def gold_rows(db: Database, sql: str):
    return db.execute(sql, max_rows=100000, timeout=30).rows


def make_system(mode: str, db, llm, index) -> ClarifySQL:
    base = PipelineOptions()
    opts = {"baseline": base, "full": base,
            "no_clarify": replace(base, use_clarification=False),
            "no_retrieval": replace(base, use_retrieval=False),
            "no_repair": replace(base, use_repair=False)}[mode]
    return ClarifySQL(db, llm, index, opts)


def run_unit(mode: str, sysm: ClarifySQL, db: Database, q: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Returns one or more evaluation records for one test question."""
    recs: List[Dict[str, Any]] = []
    qtype = q["type"]
    answerable = q.get("answerable", True)

    def call(question, **kw):
        return sysm.baseline(question) if mode == "baseline" else sysm.ask(question, **kw)

    def rec(res, correct, **extra):
        d = {"id": q["id"], "type": qtype, "question": q["question"], "mode": mode, "correct": bool(correct),
             **res.to_log()}
        d.update(extra)
        recs.append(d)

    def matches(res, gold_sql) -> bool:
        if res.status != "answer":
            return False
        try:
            return results_match(res.rows, gold_rows(db, gold_sql))
        except SQLExecError:
            return False

    if qtype == "vague":
        interps = q["interpretations"]
        first = call(q["question"])
        if first.status == "clarify":
            for it in interps:
                res = call(q["question"], clarification_answer=it["label"],
                           clarification_question=q["clarifying_question"])
                rec(res, matches(res, it["expected_sql"]), asked=True, interpretation=it["label"])
        else:
            ok = any(matches(first, it["expected_sql"]) for it in interps)
            rec(first, ok, asked=False, interpretation="(any)")
    elif not answerable:
        res = call(q["question"])
        rec(res, res.status == "unanswerable", asked=(res.status == "clarify"))
    else:
        res = call(q["question"])
        rec(res, matches(res, q["expected_sql"]), asked=(res.status == "clarify"))
        if res.status == "clarify":              # unnecessary clarification: give up on this one
            recs[-1]["correct"] = False
    return recs


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_mode[r["mode"]].append(r)
    out: Dict[str, Any] = {}
    for mode, rs in by_mode.items():
        s: Dict[str, Any] = {"cases": len(rs), "accuracy": sum(r["correct"] for r in rs) / len(rs)}
        for t in ("clear", "tricky", "vague"):
            sub = [r for r in rs if r["type"] == t]
            if sub:
                s[f"acc_{t}"] = sum(r["correct"] for r in sub) / len(sub)
        unans = [r for r in rs if r["type"] == "tricky" and r["status"] in ("unanswerable",) or
                 (r["type"] == "tricky" and r["sql"] is None and r["status"] != "answer")]
        vague = [r for r in rs if r["type"] == "vague"]
        if mode != "baseline" and mode != "no_clarify":
            if vague:
                s["clarification_recall"] = sum(1 for r in vague if r.get("asked")) / len(vague)
            nonvague = [r for r in rs if r["type"] in ("clear", "tricky")]
            if nonvague:
                s["unnecessary_clarification_rate"] = sum(1 for r in nonvague if r.get("asked") or r["status"] == "clarify") / len(nonvague)
        s["avg_llm_calls"] = sum(r["llm_calls"] for r in rs) / len(rs)
        s["avg_seconds"] = sum(r["seconds"] for r in rs) / len(rs)
        answered = [r for r in rs if r["status"] == "answer" and r.get("trust_level")]
        if answered:
            lv = {}
            for level in ("High", "Medium", "Low"):
                sub = [r for r in answered if r["trust_level"] == level]
                lv[level] = {"n": len(sub), "accuracy": (sum(r["correct"] for r in sub) / len(sub)) if sub else None}
            s["accuracy_by_trust_badge"] = lv
            s["trust_auroc"] = auroc([r["trust_score"] for r in answered], [r["correct"] for r in answered])
            good = [r["trust_score"] for r in answered if r["correct"]]
            bad = [r["trust_score"] for r in answered if not r["correct"]]
            s["mean_trust_correct"] = sum(good) / len(good) if good else None
            s["mean_trust_incorrect"] = sum(bad) / len(bad) if bad else None
        out[mode] = s
    return out


def fmt(x, pct=True):
    if x is None:
        return "-"
    return f"{x * 100:.1f}%" if pct else f"{x:.2f}"


def markdown(summary: Dict[str, Any]) -> str:
    lines = ["| Mode | Cases | Exec. accuracy | Clear | Tricky | Vague | Clarif. recall | Unneeded clarif. | Avg LLM calls |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m, s in summary.items():
        lines.append(f"| {m} | {s['cases']} | {fmt(s['accuracy'])} | {fmt(s.get('acc_clear'))} | {fmt(s.get('acc_tricky'))} | "
                     f"{fmt(s.get('acc_vague'))} | {fmt(s.get('clarification_recall'))} | "
                     f"{fmt(s.get('unnecessary_clarification_rate'))} | {s['avg_llm_calls']:.1f} |")
    for m, s in summary.items():
        if "accuracy_by_trust_badge" in s:
            lines += ["", f"**Trust badge ({m})** - does High mean more accurate than Low?", "",
                      "| Badge | Answers | Accuracy |", "|---|---|---|"]
            for level, v in s["accuracy_by_trust_badge"].items():
                lines.append(f"| {level} | {v['n']} | {fmt(v['accuracy'])} |")
            lines.append(f"\nAUROC of trust score for correct-vs-wrong: {fmt(s.get('trust_auroc'), pct=False)} "
                         f"(0.5 = useless, 1.0 = perfect). Mean score: correct {fmt(s.get('mean_trust_correct'), pct=False)}, "
                         f"wrong {fmt(s.get('mean_trust_incorrect'), pct=False)}.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="ClarifySQL evaluation")
    ap.add_argument("--modes", default=",".join(ALL_MODES), help="comma separated subset of: " + ",".join(ALL_MODES))
    ap.add_argument("--limit", type=int, default=0, help="max questions per type (0 = all)")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ALL_MODES:
            sys.exit(f"Unknown mode '{m}'. Choose from {ALL_MODES}")

    settings = Settings()
    try:
        db = Database(settings.db_path)
        llm = GeminiLLM(settings)
        index = SchemaIndex(db, llm, settings.cache_dir)
    except (LLMError, FileNotFoundError) as e:
        sys.exit(f"Cannot start: {e}")
    questions = json.loads(settings.questions_path.read_text(encoding="utf-8"))
    if args.limit:
        per, keep = defaultdict(int), []
        for q in questions:
            if per[q["type"]] < args.limit:
                per[q["type"]] += 1
                keep.append(q)
        questions = keep
    print(f"{len(questions)} questions x {len(modes)} modes. Answers are cached, so you can stop (Ctrl+C) and rerun.\n")

    records: List[Dict[str, Any]] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        for mode in modes:
            sysm = make_system(mode, db, llm, index)
            t0 = time.time()
            for i, q in enumerate(questions, 1):
                try:
                    recs = run_unit(mode, sysm, db, q)
                except LLMError as e:
                    print(f"  ! {mode} {q['id']}: LLM problem: {e}")
                    recs = [{"id": q["id"], "type": q["type"], "question": q["question"], "mode": mode, "correct": False,
                             "status": "error", "sql": None, "message": str(e), "llm_calls": 0, "seconds": 0.0}]
                records.extend(recs)
                mark = "".join("+" if r["correct"] else "-" for r in recs)
                print(f"  [{mode}] {i}/{len(questions)} {q['id']} {mark}", end="\r", flush=True)
            print(f"  [{mode}] finished in {time.time() - t0:.0f}s" + " " * 20)
    except KeyboardInterrupt:
        print("\nInterrupted - saving what has been finished so far.")

    if not records:
        sys.exit("No results to save.")
    summary = summarise(records)
    (settings.results_dir / f"records_{stamp}.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (settings.results_dir / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = markdown(summary)
    (settings.results_dir / f"summary_{stamp}.md").write_text(md, encoding="utf-8")
    print("\n" + md + f"\n\nSaved to {settings.results_dir}/ (records_{stamp}.json, summary_{stamp}.md/.json)")


if __name__ == "__main__":
    main()
