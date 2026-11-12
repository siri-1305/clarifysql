"""Central settings. Everything can be overridden from the .env file."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at import time
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _csv(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _opt_float(name: str) -> Optional[float]:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else None


@dataclass
class Settings:
    db_path: Path = ROOT / "data" / "ecommerce.db"
    questions_path: Path = ROOT / "data" / "questions.json"
    cache_dir: Path = ROOT / ".cache"
    results_dir: Path = ROOT / "results"

    api_key: str = field(default_factory=lambda: (os.getenv("GROQ_API_KEY") or "").strip())
    # Tried in order. If one is retired or hits its daily limit, the next is used (limits are per model).
    chat_models: List[str] = field(default_factory=lambda: _csv("GROQ_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b"))
    reasoning_effort: str = field(default_factory=lambda: os.getenv("GROQ_REASONING_EFFORT", "").strip().lower())
    temperature: Optional[float] = field(default_factory=lambda: _opt_float("LLM_TEMPERATURE"))
    max_output_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048")))
    # Free-tier pacing (30 requests/min and ~8K tokens/min on the models above). Set LLM_TPM=0 to disable token pacing.
    min_interval: float = field(default_factory=lambda: float(os.getenv("LLM_MIN_INTERVAL", "2.2")))
    tokens_per_minute: int = field(default_factory=lambda: int(os.getenv("LLM_TPM", "7000")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "6")))
    use_cache: bool = field(default_factory=lambda: os.getenv("LLM_CACHE", "1") != "0")

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineOptions:
    """Switches used by the app and by the ablation study in evaluate.py."""
    use_clarification: bool = True
    use_retrieval: bool = True
    use_repair: bool = True
    agreement_samples: int = 2        # extra independently generated queries used for the agreement signal
    ambiguity_threshold: float = 0.5  # ask a follow-up only when ambiguity >= this
    max_repairs: int = 2              # bounded repair loop
    top_k_tables: int = 4
    max_rows: int = 500
    query_timeout_s: float = 5.0
    trust_high: float = 0.80
    trust_medium: float = 0.55
