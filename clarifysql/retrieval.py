"""Schema retrieval (the 'R' in RAG): embed tables/columns, retrieve only what the question needs.

Vectors come from the LLM provider's embedding API when it has one (Groq does not), otherwise from a local
hashed bag-of-words embedder, which is always available as a fallback.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import zlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from .db import Database
from .llm import BaseLLM, LLMError

log = logging.getLogger("clarifysql.retrieval")

_STOP = set("""a an the of is are was were be to in on for by and or with from at as it its this that these those
show me list give tell what which who whom how many much our my your all any each every do does did there their
total number find get""".split())


def _stem(tok: str) -> str:
    for suf in ("ing", "es", "ed", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


class LocalHashEmbedder:
    """Tiny dependency-free embedder (hashed bag of stemmed words). Used only as a fallback."""
    dim = 1024

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = [_stem(w) for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in _STOP]
            for w in toks:
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
        out = np.log1p(out)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


@dataclass
class Doc:
    kind: str            # "table" or "column"
    table: str
    column: Optional[str]
    text: str


@dataclass
class RetrievalResult:
    tables: List[str]
    scores: Dict[str, float]
    confidence: float
    method: str
    columns: List[str] = field(default_factory=list)
    expanded: bool = False       # set by the pipeline if it had to fall back to the full schema


class SchemaIndex:
    # (floor, ceiling) used to turn the best similarity into a 0..1 confidence; heuristic - tune with evaluate.py
    CONF_RANGE = {"api": (0.40, 0.75), "local": (0.05, 0.40)}
    MARGIN = {"api": 0.10, "local": 0.12}
    VALUE_BOOST = 0.15

    def __init__(self, db: Database, llm: Optional[BaseLLM], cache_dir: Path):
        self.db, self.llm, self.cache_dir = db, llm, Path(cache_dir)
        self.docs: List[Doc] = self._make_docs()
        self.value_index = self._make_value_index()
        self.local = LocalHashEmbedder()
        self.local_vecs = self.local.embed([d.text for d in self.docs])
        self.api_vecs: Optional[np.ndarray] = None
        self.api_model = ""
        self._load_or_build_api_vecs()

    # ------------------------------------------------------------------ documents
    def _make_docs(self) -> List[Doc]:
        docs: List[Doc] = []
        for t in self.db.tables.values():
            cols = ", ".join(c.name for c in t.columns)
            links = [f"{c.name} links to {c.fk[0]}" for c in t.columns if c.fk]
            text = f"Table {t.name.replace('_', ' ')} with columns: {cols.replace('_', ' ')}."
            if links:
                text += " " + "; ".join(links).replace("_", " ") + "."
            docs.append(Doc("table", t.name, None, text))
            for c in t.columns:
                extra = ""
                if c.values:
                    extra = " Values such as " + ", ".join(c.values[:8]) + "."
                elif c.samples:
                    extra = " Example values " + ", ".join(c.samples[:3]) + "."
                docs.append(Doc("column", t.name, c.name,
                                f"Column {c.name.replace('_', ' ')} of table {t.name.replace('_', ' ')} ({c.type}).{extra}"))
        return docs

    def _make_value_index(self):
        """Stored text values (cities, statuses, product names ...) used to ground question words to tables."""
        out = []
        for t in self.db.tables.values():
            for c in t.columns:
                for v in (c.values or []):
                    if len(v) >= 3:
                        out.append((re.compile(r"\b" + re.escape(v.lower()) + r"\b"), t.name))
        return out

    def _fingerprint(self) -> str:
        raw = json.dumps([d.text for d in self.docs])
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _load_or_build_api_vecs(self):
        if self.llm is None or not getattr(self.llm, "supports_embeddings", True):
            return                      # e.g. Groq: no embedding API, use the local embedder only
        fp = self._fingerprint()
        vec_file = self.cache_dir / f"schema_vecs_{fp}.npy"
        meta_file = self.cache_dir / f"schema_vecs_{fp}.json"
        try:
            if vec_file.exists() and meta_file.exists():
                self.api_vecs = np.load(vec_file)
                self.api_model = json.loads(meta_file.read_text())["model"]
                if self.api_vecs.shape[0] == len(self.docs):
                    return
            vecs = self.llm.embed([d.text for d in self.docs])
            self.api_vecs, self.api_model = vecs, getattr(self.llm, "embed_model_name", "api")
            np.save(vec_file, vecs)
            meta_file.write_text(json.dumps({"model": self.api_model}))
        except Exception as e:  # noqa: BLE001
            log.warning("API embeddings unavailable (%s) - using the local embedder.", e)
            self.api_vecs = None

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, question: str, top_k: int = 4) -> RetrievalResult:
        method, vecs, q = "local", self.local_vecs, None
        if self.api_vecs is not None and self.llm is not None:
            try:
                q = self.llm.embed([question])[0]
                if q.shape[0] == self.api_vecs.shape[1]:
                    method, vecs = "api", self.api_vecs
                else:
                    q = None
            except Exception as e:  # noqa: BLE001
                log.warning("Query embedding failed (%s) - using local embedder.", e)
                q = None
        if q is None:
            q = self.local.embed([question])[0]
        sims = vecs @ q

        t_score: Dict[str, float] = {}
        best_col: Dict[str, List[tuple]] = {}
        for d, s in zip(self.docs, sims):
            s = float(s)
            t_score[d.table] = max(t_score.get(d.table, -1.0), s)
            if d.kind == "column":
                best_col.setdefault(d.table, []).append((s, d.column))
        ql = question.lower()
        grounded = {t for rx, t in self.value_index if rx.search(ql)}
        for t in grounded:                                  # value grounding: 'Hyderabad' -> customers
            t_score[t] = t_score[t] + self.VALUE_BOOST
        ranked = sorted(t_score.items(), key=lambda kv: kv[1], reverse=True)
        best = ranked[0][1]
        k = max(1, top_k)
        within = [t for t, s in ranked if s >= best - self.MARGIN[method]]
        keep = list(dict.fromkeys([t for t, _ in ranked[:min(3, k)]] + within))[:k]
        keep = self._bridge(keep)
        lo, hi = self.CONF_RANGE[method]
        conf = float(min(1.0, max(0.0, (best - lo) / (hi - lo))))
        cols = []
        for t in keep:
            for s, c in sorted(best_col.get(t, []), reverse=True)[:3]:
                cols.append(f"{t}.{c}")
        label = self.api_model if method == "api" else "local-hash"
        return RetrievalResult(tables=keep, scores=t_score, confidence=conf, method=label, columns=cols)

    def _bridge(self, selected: List[str]) -> List[str]:
        """Add the tables needed to join the selected ones (shortest foreign-key paths)."""
        graph: Dict[str, Set[str]] = {t: set() for t in self.db.tables}
        for a, _, b, _ in self.db.fk_edges():
            graph[a].add(b)
            graph[b].add(a)
        result = list(selected)
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                for t in self._path(graph, selected[i], selected[j]):
                    if t not in result:
                        result.append(t)
        return result

    @staticmethod
    def _path(graph, src, dst) -> List[str]:
        prev = {src: None}
        dq = deque([src])
        while dq:
            cur = dq.popleft()
            if cur == dst:
                break
            for nb in graph.get(cur, ()):
                if nb not in prev:
                    prev[nb] = cur
                    dq.append(nb)
        if dst not in prev:
            return []
        path, node = [], prev[dst]
        while node is not None and node != src:
            path.append(node)
            node = prev[node]
        return path
