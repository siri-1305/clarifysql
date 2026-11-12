"""Result-set comparison (execution accuracy) and evaluation statistics."""
from __future__ import annotations

from itertools import permutations
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple


def _norm(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    if isinstance(v, str):
        return v.strip()
    return v


def _rows(rows) -> List[tuple]:
    return [tuple(_norm(v) for v in r) for r in rows]


def _key(rows, ordered: bool):
    return rows if ordered else sorted(rows, key=repr)


def results_match(pred_rows, gold_rows, ordered: bool = False) -> bool:
    """Execution-accuracy comparison.

    Two result sets match when they contain the same rows (order ignored unless `ordered`).
    Column order is ignored, and the prediction may contain extra columns
    (e.g. it also returns the score that was used for ranking).
    """
    p, g = _rows(pred_rows), _rows(gold_rows)
    if len(p) != len(g):
        return False
    if not p:
        return True
    pw, gw = len(p[0]), len(g[0])
    gk = _key(g, ordered)
    if pw == gw and _key(p, ordered) == gk:
        return True
    if pw < gw:
        return False
    if comb(pw, gw) * 1 > 5000 or pw > 8:
        # too many column permutations: only try the first gw columns
        return _key([r[:gw] for r in p], ordered) == gk
    for idx in permutations(range(pw), gw):
        if _key([tuple(r[i] for i in idx) for r in p], ordered) == gk:
            return True
    return False


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Probability that a random correct answer gets a higher trust score than a random wrong one."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / (len(pos) * len(neg))
