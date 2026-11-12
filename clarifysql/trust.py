"""Trust layer: turn the signals the pipeline already produces into a High / Medium / Low badge."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Weights are design hypotheses; evaluate.py measures whether the badge actually separates right from wrong.
WEIGHTS = {"agreement": 0.35, "repairs": 0.25, "result": 0.20, "retrieval": 0.20}


@dataclass
class TrustSignals:
    repairs: int = 0
    validated: bool = True                  # final SQL passed validation and executed
    agreement: Optional[float] = None       # share of independently generated queries giving the same result
    n_candidates: int = 0
    suspicious: Optional[str] = None        # "empty" | "all_null" | "zero" | "truncated" | None
    retrieval_confidence: Optional[float] = None
    retrieval_expanded: bool = False        # had to fall back to the full schema


@dataclass
class TrustResult:
    level: str                              # "High" | "Medium" | "Low"
    score: float                            # 0..1
    reasons: List[str] = field(default_factory=list)
    parts: Dict[str, float] = field(default_factory=dict)


def compute_trust(sig: TrustSignals, high: float = 0.80, medium: float = 0.55) -> TrustResult:
    parts: Dict[str, float] = {}
    reasons: List[str] = []

    # 1) repairs needed
    if not sig.validated:
        parts["repairs"] = 0.0
        reasons.append("The query could not be fully validated, even after repair attempts.")
    else:
        parts["repairs"] = {0: 1.0, 1: 0.6, 2: 0.3}.get(sig.repairs, 0.1)
        reasons.append("The query passed all checks on the first try." if sig.repairs == 0 else
                       f"The query needed {sig.repairs} automatic fix{'es' if sig.repairs > 1 else ''} before it passed the checks.")

    # 2) agreement between independently generated queries
    if sig.agreement is not None:
        parts["agreement"] = sig.agreement
        k = round(sig.agreement * (sig.n_candidates or 1))
        reasons.append(f"{k} of {sig.n_candidates} independently generated queries returned the same answer.")

    # 3) suspicious results
    sus = {None: (1.0, "The result looks normal (not empty, no unexpected NULLs)."),
           "zero": (0.5, "The answer is a single 0, which is sometimes correct but worth a look."),
           "truncated": (0.7, "The result was very long and was cut to the first rows."),
           "empty": (0.0, "The query returned no rows, which often means a wrong filter."),
           "all_null": (0.0, "Every returned value is empty (NULL), which usually means a wrong join.")}
    parts["result"], msg = sus.get(sig.suspicious, sus[None])
    reasons.append(msg)

    # 4) retrieval confidence
    if sig.retrieval_confidence is not None:
        r = sig.retrieval_confidence
        if sig.retrieval_expanded:
            r = min(r, 0.2)
            reasons.append("The first schema selection was not enough, so the whole schema had to be used.")
        else:
            reasons.append("The relevant tables were found with " + ("high" if r >= 0.7 else "moderate" if r >= 0.4 else "low") + " confidence.")
        parts["retrieval"] = r

    total_w = sum(WEIGHTS[k] for k in parts)
    score = sum(WEIGHTS[k] * v for k, v in parts.items()) / total_w if total_w else 0.0
    level = "High" if score >= high else "Medium" if score >= medium else "Low"

    # hard caps: a single strong red flag should never hide behind good averages
    if not sig.validated:
        level = "Low"
    if level == "High" and ((sig.agreement is not None and sig.agreement < 0.5) or sig.suspicious in ("empty", "all_null") or sig.repairs >= 2):
        level = "Medium"
    return TrustResult(level=level, score=round(score, 3), reasons=reasons, parts={k: round(v, 3) for k, v in parts.items()})
