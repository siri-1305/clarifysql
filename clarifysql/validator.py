"""SQL validation: syntax, schema consistency, filter values, then a real (read-only) execution.

Everything the checker finds is returned as plain-text diagnostics that are fed back to the LLM
by the bounded repair loop.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .db import Database, QueryResult, SQLExecError

try:
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import SqlglotError
    HAS_SQLGLOT = True
except ImportError:  # the app still runs, with reduced static checks
    sqlglot = exp = None
    SqlglotError = Exception
    HAS_SQLGLOT = False

log = logging.getLogger("clarifysql.validator")

FATAL = {"empty", "syntax", "not_select", "multi_statement", "unknown_table", "unknown_column", "forbidden"}
FORBIDDEN_RE = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|REPLACE|VACUUM|REINDEX)\b", re.I)


@dataclass
class Issue:
    code: str
    message: str
    severity: str = "error"      # "error" or "warning"


@dataclass
class ValidationReport:
    sql: str
    issues: List[Issue] = field(default_factory=list)
    result: Optional[QueryResult] = None
    tables_used: Set[str] = field(default_factory=set)
    suspicious: Optional[str] = None      # "empty" | "all_null" | "zero" | "truncated" | None

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        """Valid SQL that executed (value problems and suspicious results are reported separately)."""
        return not self.errors and self.result is not None

    @property
    def needs_repair(self) -> bool:
        return bool(self.errors) or self.suspicious in ("empty", "all_null")

    def diagnostics(self) -> str:
        lines = [f"- {i.message}" for i in self.issues if i.severity == "error"]
        if self.suspicious == "empty":
            lines.append("- The query ran but returned NO rows. Check filter values against the schema's stored values, join "
                         "conditions and date ranges. (If the data genuinely has no matching rows, keep the query as it is.)")
        elif self.suspicious == "all_null":
            lines.append("- The query ran but every returned value is NULL. Check the joins and the columns used in the aggregate.")
        return "\n".join(lines) if lines else "- (no problems)"


def check_result(res: QueryResult) -> Optional[str]:
    if not res.rows:
        return "empty"
    if all(v is None for r in res.rows for v in r):
        return "all_null"
    if len(res.rows) == 1 and len(res.rows[0]) == 1 and isinstance(res.rows[0][0], (int, float)) and res.rows[0][0] == 0:
        return "zero"
    if res.truncated:
        return "truncated"
    return None


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---------------------------------------------------------------------- static checks (sqlglot)
def _static_checks(sql: str, db: Database) -> Tuple[List[Issue], Set[str], Optional[object]]:
    issues: List[Issue] = []
    tables_used: Set[str] = set()
    try:
        trees = sqlglot.parse(sql, read="sqlite")
    except SqlglotError as e:
        first = _strip_ansi(str(e)).strip().splitlines()[0][:300]
        return [Issue("syntax", f"SQL syntax error: {first}")], tables_used, None
    trees = [t for t in trees if t is not None]
    if not trees:
        return [Issue("empty", "The statement is empty.")], tables_used, None
    if len(trees) > 1:
        issues.append(Issue("multi_statement", "Only ONE statement is allowed (found several separated by ';')."))
    tree = trees[0]
    select_types = tuple(t for t in (getattr(exp, "Select", None), getattr(exp, "Union", None),
                                     getattr(exp, "SetOperation", None)) if t is not None)
    if not isinstance(tree, select_types):
        return issues + [Issue("not_select", "Only read-only SELECT statements are allowed.")], tables_used, tree

    cte_names = {c.alias.lower() for c in tree.find_all(exp.CTE) if c.alias}
    scopes: Dict[int, Dict[str, object]] = {}
    for sel in tree.find_all(exp.Select):
        scopes[id(sel)] = {"real": {}, "virtual": set()}

    all_table_names = ", ".join(sorted(db.tables))
    for t in tree.find_all(exp.Table):
        name = t.name
        if not name:
            continue
        sel = t.find_ancestor(exp.Select)
        if sel is None or id(sel) not in scopes:
            continue
        alias = (t.alias or name).lower()
        if name.lower() in cte_names:
            scopes[id(sel)]["virtual"].add(alias)
            continue
        real = db.table_lookup(name)
        if real is None:
            close = difflib.get_close_matches(name.lower(), [x.lower() for x in db.tables], n=1, cutoff=0.6)
            hint = f" Did you mean '{db.table_lookup(close[0])}'?" if close else ""
            issues.append(Issue("unknown_table", f"Table '{name}' does not exist.{hint} Existing tables: {all_table_names}."))
        else:
            scopes[id(sel)]["real"][alias] = real
            tables_used.add(real)
    for sq in tree.find_all(exp.Subquery):
        if sq.alias:
            sel = sq.find_ancestor(exp.Select)
            if sel is not None and id(sel) in scopes:
                scopes[id(sel)]["virtual"].add(sq.alias.lower())

    out_aliases = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
    resolved: List[Tuple[object, Optional[str]]] = []   # (column node, real table or None)

    for col in tree.find_all(exp.Column):
        if isinstance(col.this, exp.Star):
            continue
        name, qual = col.name, (col.table or "")
        chain = []
        s = col.find_ancestor(exp.Select)
        while s is not None:
            chain.append(scopes.get(id(s), {"real": {}, "virtual": set()}))
            s = s.find_ancestor(exp.Select)
        if not chain:
            continue                                   # e.g. ORDER BY after a UNION
        if qual:
            found, virtual = None, False
            for sc in chain:
                if qual.lower() in sc["real"]:
                    found = sc["real"][qual.lower()]
                    break
                if qual.lower() in sc["virtual"]:
                    virtual = True
                    break
            if found:
                if db.has_column(found, name):
                    resolved.append((col, found))
                else:
                    issues.append(Issue("unknown_column",
                                        f"Column '{qual}.{name}' does not exist: table {found} has columns {', '.join(db.column_names(found))}."))
            elif not virtual:
                issues.append(Issue("unknown_column", f"Unknown table alias '{qual}' (used in '{qual}.{name}')."))
        else:
            if any(sc["virtual"] for sc in chain) or name.lower() in out_aliases:
                continue
            cands = [tb for sc in chain for tb in sc["real"].values()]
            owners = [tb for tb in dict.fromkeys(cands) if db.has_column(tb, name)]
            if owners:
                resolved.append((col, owners[0]))
            else:
                where = ", ".join(dict.fromkeys(cands)) or "the query"
                issues.append(Issue("unknown_column",
                                    f"Column '{name}' not found in {where}. (Text values must be written in single quotes, not double quotes.)"))

    issues.extend(_value_checks(tree, db, resolved))
    seen, uniq = set(), []
    for i in issues:
        if i.message not in seen:
            seen.add(i.message)
            uniq.append(i)
    return uniq, tables_used, tree


def _value_checks(tree, db: Database, resolved) -> List[Issue]:
    """Flag string literals compared with a categorical column that never occur in that column."""
    table_of = {id(c): t for c, t in resolved}
    issues: List[Issue] = []
    for node in tree.find_all(exp.EQ, exp.NEQ, exp.In):
        pairs = []
        if isinstance(node, exp.In):
            pairs.append((node.this, list(node.expressions or [])))
        else:
            l, r = node.this, node.expression
            if isinstance(l, exp.Column):
                pairs.append((l, [r]))
            if isinstance(r, exp.Column):
                pairs.append((r, [l]))
        for col, lits in pairs:
            if not isinstance(col, exp.Column) or id(col) not in table_of:
                continue
            table = table_of[id(col)]
            valid = db.categorical_values(table, col.name)
            if valid is None:
                continue
            for lit in lits:
                if isinstance(lit, exp.Literal) and lit.is_string:
                    v = lit.name
                    if v in valid:
                        continue
                    issues.append(Issue("bad_value", _value_message(v, table, col.name, valid)))
    return issues


def _value_message(v: str, table: str, column: str, valid: List[str]) -> str:
    msg = f"Filter value '{v}' does not exist in {table}.{column}."
    low = {x.lower(): x for x in valid}
    if v.lower() in low:
        return msg + f" Values are case-sensitive; use '{low[v.lower()]}'."
    sub = [x for x in valid if v.lower() in x.lower() or x.lower() in v.lower()]
    close = difflib.get_close_matches(v, valid, n=3, cutoff=0.5)
    sugg = list(dict.fromkeys(sub + close))[:3]
    if sugg:
        msg += " Did you mean " + " or ".join(f"'{s}'" for s in sugg) + "?"
    if len(valid) <= 15:
        msg += " All stored values: " + ", ".join(f"'{x}'" for x in valid) + "."
    return msg


# ---------------------------------------------------------------------- public entry point
def validate(sql: str, db: Database, max_rows: int = 500, timeout: float = 5.0) -> ValidationReport:
    rep = ValidationReport(sql=sql)
    if not sql or not sql.strip():
        rep.issues.append(Issue("empty", "The statement is empty."))
        return rep

    if HAS_SQLGLOT:
        try:
            issues, tables, _ = _static_checks(sql, db)
            rep.issues.extend(issues)
            rep.tables_used = tables
        except Exception as e:  # noqa: BLE001 - never let a checker bug crash the app
            log.warning("static validation crashed: %s", e, exc_info=True)
            rep.issues.append(Issue("validator_internal", f"static checks skipped ({type(e).__name__})", "warning"))
    else:
        stripped = sql.strip().rstrip(";")
        if ";" in stripped:
            rep.issues.append(Issue("multi_statement", "Only ONE statement is allowed."))
        if not re.match(r"\s*(SELECT|WITH)\b", stripped, re.I) or FORBIDDEN_RE.search(stripped):
            rep.issues.append(Issue("not_select", "Only read-only SELECT statements are allowed."))
        rep.issues.append(Issue("sqlglot_missing", "sqlglot is not installed: schema/value checks are reduced.", "warning"))
        rep.tables_used = {t for t in db.tables if re.search(r"\b" + re.escape(t) + r"\b", sql, re.I)}

    if any(i.code in FATAL for i in rep.errors):
        return rep
    try:
        rep.result = db.execute(sql, max_rows=max_rows, timeout=timeout)
        rep.suspicious = check_result(rep.result)
    except SQLExecError as e:
        rep.issues.append(Issue("execution", f"SQLite rejected the query: {e}"))
    return rep
