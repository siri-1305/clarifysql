"""SQLite access: read-only execution, schema introspection and value profiling."""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

CATEGORICAL_MAX = 60      # text columns with <= this many distinct values are treated as categorical
INLINE_VALUES_MAX = 15    # list all values in the prompt only when there are at most this many


class SQLExecError(Exception):
    """A SQL statement failed inside SQLite (message is the SQLite error text)."""


@dataclass
class QueryResult:
    columns: List[str]
    rows: List[tuple]
    truncated: bool = False


@dataclass
class ColumnInfo:
    name: str
    type: str
    pk: bool = False
    fk: Optional[Tuple[str, str]] = None          # (table, column)
    values: Optional[List[str]] = None            # every distinct value, if categorical
    samples: List[str] = field(default_factory=list)
    vmin: Optional[str] = None
    vmax: Optional[str] = None


@dataclass
class TableInfo:
    name: str
    columns: List[ColumnInfo]
    row_count: int

    def column(self, name: str) -> Optional[ColumnInfo]:
        for c in self.columns:
            if c.name.lower() == name.lower():
                return c
        return None


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class Database:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Database not found: {self.path}. Run 'python make_dataset.py' inside the data folder.")
        self.tables: Dict[str, TableInfo] = {}
        self._introspect()

    # ------------------------------------------------------------------ connections
    def _connect(self) -> sqlite3.Connection:
        uri = self.path.as_uri() + "?mode=ro"          # works on Windows and Linux/macOS
        con = sqlite3.connect(uri, uri=True, check_same_thread=False)
        con.execute("PRAGMA query_only = ON")
        return con

    def execute(self, sql: str, max_rows: int = 500, timeout: float = 5.0) -> QueryResult:
        con = self._connect()
        try:
            deadline = time.monotonic() + timeout
            con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            return QueryResult(cols, [tuple(r) for r in rows[:max_rows]], truncated)
        except sqlite3.Error as e:
            msg = str(e)
            if "interrupted" in msg.lower():
                msg = f"query took longer than {timeout:.0f}s and was stopped"
            raise SQLExecError(msg) from e
        finally:
            con.close()

    # ------------------------------------------------------------------ introspection
    def _introspect(self):
        con = self._connect()
        try:
            names = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            for t in names:
                fks = {r[3]: (r[2], r[4]) for r in con.execute(f"PRAGMA foreign_key_list({_q(t)})")}
                cols: List[ColumnInfo] = []
                for _cid, name, ctype, _nn, _d, pk in con.execute(f"PRAGMA table_info({_q(t)})"):
                    ci = ColumnInfo(name=name, type=(ctype or "").upper() or "TEXT", pk=bool(pk), fk=fks.get(name))
                    is_text = any(k in ci.type for k in ("CHAR", "TEXT", "CLOB")) or ci.type == ""
                    if is_text:
                        vals = [r[0] for r in con.execute(
                            f"SELECT DISTINCT {_q(name)} FROM {_q(t)} WHERE {_q(name)} IS NOT NULL "
                            f"LIMIT {CATEGORICAL_MAX + 1}")]
                        if len(vals) <= CATEGORICAL_MAX:
                            ci.values = sorted(str(v) for v in vals)
                    ci.samples = [str(r[0]) for r in con.execute(
                        f"SELECT {_q(name)} FROM {_q(t)} WHERE {_q(name)} IS NOT NULL LIMIT 3")]
                    if "date" in name.lower():
                        lo, hi = con.execute(f"SELECT MIN({_q(name)}), MAX({_q(name)}) FROM {_q(t)}").fetchone()
                        ci.vmin, ci.vmax = (str(lo) if lo is not None else None), (str(hi) if hi is not None else None)
                    cols.append(ci)
                n = con.execute(f"SELECT COUNT(*) FROM {_q(t)}").fetchone()[0]
                self.tables[t] = TableInfo(t, cols, n)
        finally:
            con.close()

    # ------------------------------------------------------------------ lookups
    def table_lookup(self, name: str) -> Optional[str]:
        for t in self.tables:
            if t.lower() == name.lower():
                return t
        return None

    def column_names(self, table: str) -> List[str]:
        return [c.name for c in self.tables[table].columns]

    def has_column(self, table: str, column: str) -> bool:
        return self.tables[table].column(column) is not None

    def categorical_values(self, table: str, column: str) -> Optional[List[str]]:
        t = self.tables.get(table)
        c = t.column(column) if t else None
        return c.values if c else None

    def fk_edges(self) -> List[Tuple[str, str, str, str]]:
        out = []
        for t in self.tables.values():
            for c in t.columns:
                if c.fk:
                    out.append((t.name, c.name, c.fk[0], c.fk[1]))
        return out

    # ------------------------------------------------------------------ text for prompts
    def table_text(self, name: str) -> str:
        t = self.tables[name]
        lines = [f"Table {t.name} ({t.row_count} rows)"]
        for c in t.columns:
            line = f"  - {c.name} {c.type}"
            if c.pk:
                line += " PRIMARY KEY"
            if c.fk:
                line += f" -> {c.fk[0]}.{c.fk[1]}"
            if c.values is not None and len(c.values) <= INLINE_VALUES_MAX:
                line += "   values: " + " | ".join(f"'{v}'" for v in c.values)
            elif c.vmin is not None:
                line += f"   range: {c.vmin} .. {c.vmax}"
            elif c.samples:
                line += "   e.g. " + ", ".join(f"'{s}'" if "CHAR" in c.type or "TEXT" in c.type else s for s in c.samples)
            lines.append(line)
        return "\n".join(lines)

    def schema_text(self, tables: Optional[Sequence[str]] = None) -> str:
        names = list(tables) if tables else list(self.tables)
        return "\n\n".join(self.table_text(n) for n in names if n in self.tables)

    def schema_summary(self) -> str:
        """Compact overview (names only) used by the ambiguity checker."""
        lines = []
        for t in self.tables.values():
            lines.append(f"{t.name}({', '.join(c.name for c in t.columns)})")
        dates = [f"{t.name}.{c.name} covers {c.vmin} to {c.vmax}" for t in self.tables.values()
                 for c in t.columns if c.vmin]
        text = "\n".join(lines)
        if dates:
            text += "\nDate ranges: " + "; ".join(dates)
        return text


def canonical_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()
