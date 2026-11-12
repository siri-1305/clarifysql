"""LLM access layer (Groq) with retries, model fallback, token-aware rate limiting and an on-disk cache.

Every other module talks to the `BaseLLM` interface only, so tests can plug in a fake LLM.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import Settings

log = logging.getLogger("clarifysql.llm")


class LLMError(Exception):
    """Raised when the LLM cannot be reached or returns nothing usable."""


class BaseLLM:
    """Interface used by the whole pipeline."""
    calls: int = 0

    def generate(self, prompt: str, *, system: Optional[str] = None, json_mode: bool = False,
                 sample: int = 0) -> str:
        raise NotImplementedError

    def embed(self, texts: List[str]) -> np.ndarray:
        raise LLMError("embeddings not available")

    embed_model_name: str = "none"
    supports_embeddings: bool = True   # GroqLLM sets this to False (Groq has no embedding models)


def _clean_err(e: Exception) -> str:
    return re.sub(r"\s+", " ", str(e))[:400]


class GroqLLM(BaseLLM):
    """Chat access through Groq's OpenAI-compatible API.

    Groq serves chat models only, so `embed()` is unavailable and schema retrieval uses the local
    embedder (see retrieval.py). Free-tier limits are tight (requests AND tokens per minute/day),
    so every call is throttled by both, and a daily-limit error switches to the next model in
    GROQ_MODELS (limits are counted per model).
    """
    supports_embeddings = False

    def __init__(self, settings: Settings, client=None):
        if client is None:
            if not settings.api_key or settings.api_key.startswith("PASTE_YOUR"):
                raise LLMError("No Groq API key found. Put your key after GROQ_API_KEY= in the .env file (README step 5).")
            try:
                from groq import Groq
            except ImportError as e:  # pragma: no cover
                raise LLMError("The 'groq' package is missing. Run: pip install -r requirements.txt") from e
            client = Groq(api_key=settings.api_key, max_retries=0, timeout=90.0)   # we do our own retries
        self.client = client
        self.s = settings
        self.chat_models = list(settings.chat_models)
        self.model_idx = 0
        self.json_ok = True              # turned off if the model rejects response_format
        self.reasoning_ok = True         # turned off if the model rejects reasoning_effort
        self.calls = 0
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._window = deque()           # (timestamp, tokens) of recent calls, for the per-minute token limit
        self._cache_file = settings.cache_dir / "llm_cache.jsonl"
        self._cache = {}
        if settings.use_cache and self._cache_file.exists():
            for line in self._cache_file.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self._cache[row["k"]] = row["v"]
                except Exception:
                    continue

    # ------------------------------------------------------------------ helpers
    @property
    def model(self) -> str:
        return self.chat_models[min(self.model_idx, len(self.chat_models) - 1)]

    def _effort(self) -> Optional[str]:
        """Reasoning effort for models that support it. gpt-oss defaults to 'low' to save tokens."""
        if not self.reasoning_ok:
            return None
        if self.s.reasoning_effort:
            return self.s.reasoning_effort
        return "low" if self.model.startswith("openai/gpt-oss") else None

    def _key(self, prompt, system, json_mode, sample) -> str:
        raw = json.dumps([self.chat_models[0], system, prompt, json_mode, sample,
                          self.s.temperature, self.s.reasoning_effort], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _throttle(self, est_tokens: int):
        """Wait so that we respect both requests-per-minute and tokens-per-minute."""
        with self._lock:
            wait = self.s.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            tpm = self.s.tokens_per_minute
            if tpm > 0:
                while True:
                    now = time.monotonic()
                    while self._window and now - self._window[0][0] >= 60.0:
                        self._window.popleft()
                    used = sum(t for _, t in self._window)
                    if not self._window or used + est_tokens <= tpm:
                        break
                    time.sleep(max(0.2, 60.0 - (now - self._window[0][0]) + 0.1))
            self._last_call = time.monotonic()

    def _record(self, tokens: int):
        with self._lock:
            self._window.append((time.monotonic(), tokens))

    @staticmethod
    def _retry_delay(msg: str, attempt: int, headers=None) -> float:
        """Seconds to wait. Groq says e.g. 'Please try again in 7.5s', '1m2.4s' or '345ms'."""
        try:
            ra = headers.get("retry-after") if headers else None
            if ra:
                return min(float(ra) + 0.5, 120.0)
        except Exception:
            pass
        m = re.search(r"try again in\s+(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)(m?s))?", msg, re.I)
        if m and any(m.groups()[:3]):
            h, mi, sec, unit = m.groups()
            total = int(h or 0) * 3600 + int(mi or 0) * 60
            if sec:
                total += float(sec) / (1000.0 if unit and unit.lower() == "ms" else 1.0)
            return min(total + 1.0, 120.0)
        return min(2.0 * (2 ** attempt), 60.0)

    @staticmethod
    def _status(e: Exception) -> int:
        for attr in ("status_code", "code"):
            code = getattr(e, attr, None)
            if isinstance(code, int):
                return code
        m = re.search(r"\b(4\d\d|5\d\d)\b", str(e))
        return int(m.group(1)) if m else 0

    def _next_model(self, why: str) -> bool:
        if self.model_idx + 1 < len(self.chat_models):
            log.warning("Model %s %s, switching to %s", self.model, why, self.chat_models[self.model_idx + 1])
            self.model_idx += 1
            return True
        return False

    # ------------------------------------------------------------------ generation
    def generate(self, prompt: str, *, system: Optional[str] = None, json_mode: bool = False,
                 sample: int = 0) -> str:
        key = self._key(prompt, system, json_mode, sample)
        if self.s.use_cache and key in self._cache:
            return self._cache[key]

        last_err = ""
        for attempt in range(self.s.max_retries):
            sys_text = system or ""
            if json_mode:
                sys_text = (sys_text + "\n\n" if sys_text else "") + "Respond with a single valid JSON object only, with no other text."
            messages = ([{"role": "system", "content": sys_text}] if sys_text else []) + [{"role": "user", "content": prompt}]
            kwargs = {"model": self.model, "messages": messages, "max_completion_tokens": self.s.max_output_tokens}
            if json_mode and self.json_ok:
                kwargs["response_format"] = {"type": "json_object"}
            if self.s.temperature is not None:
                kwargs["temperature"] = self.s.temperature
            effort = self._effort()
            if effort:
                kwargs["reasoning_effort"] = effort
            try:
                self._throttle((len(prompt) + len(sys_text)) // 3 + 600)
                self.calls += 1
                resp = self.client.chat.completions.create(**kwargs)
                usage = getattr(resp, "usage", None)
                self._record(int(getattr(usage, "total_tokens", 0) or (len(prompt) // 3 + 600)))
                text = (resp.choices[0].message.content or "") if resp.choices else ""
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # some models return their reasoning inline
                if not text:
                    raise LLMError("empty response from model")
                if self.s.use_cache:
                    self._cache[key] = text
                    with self._lock, open(self._cache_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"k": key, "v": text}, ensure_ascii=False) + "\n")
                return text
            except Exception as e:  # noqa: BLE001 - the SDK raises many different types
                msg = _clean_err(e)
                last_err = msg
                status = self._status(e)
                low = msg.lower()
                headers = getattr(getattr(e, "response", None), "headers", None)
                log.warning("LLM call failed (attempt %d, status %s): %s", attempt + 1, status, msg)
                if status in (401, 403) or "invalid api key" in low or "invalid_api_key" in low:
                    raise LLMError("The Groq API key was rejected. Check GROQ_API_KEY in your .env file.") from e
                if status == 400 and "reasoning" in low and effort:
                    self.reasoning_ok = False                  # this model does not accept reasoning_effort
                    continue
                if status == 400 and json_mode and self.json_ok and ("json" in low or "response_format" in low):
                    self.json_ok = False                       # fall back to plain text; parse_json() is lenient
                    continue
                if status == 404 or any(w in low for w in ("model_not_found", "decommissioned", "does not exist", "no longer supported")):
                    if self._next_model("is unavailable"):
                        continue
                    raise LLMError(f"Model '{self.model}' is not available. Set GROQ_MODELS in .env to a current model "
                                   f"(see https://console.groq.com/docs/models). Details: {msg}") from e
                if status == 413 or "request too large" in low:
                    raise LLMError("The request is larger than your Groq tokens-per-minute limit allows. Lower LLM_TPM or "
                                   f"use a model/tier with a higher limit. Details: {msg}") from e
                if status == 429 and any(w in low for w in ("per day", "tokens per day", "requests per day", "(tpd)", "(rpd)")):
                    if self._next_model("hit its daily limit"):
                        continue
                    raise LLMError("Groq's daily limit for this model is used up. Wait for it to reset, add another model to "
                                   "GROQ_MODELS in .env (limits are per model), or upgrade the Groq tier. "
                                   "Cached answers are reused, so re-running costs nothing for questions already answered. "
                                   f"Details: {msg}") from e
                if status in (429, 500, 502, 503, 504) or any(w in low for w in ("rate limit", "overloaded", "unavailable", "timeout", "timed out", "connection", "empty response")):
                    time.sleep(self._retry_delay(msg, attempt, headers))
                    continue
                if status == 400:
                    raise LLMError(f"The request was rejected by the API: {msg}") from e
                time.sleep(self._retry_delay(msg, attempt, headers))
        raise LLMError(f"LLM call failed after {self.s.max_retries} attempts: {last_err}")

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: List[str]) -> np.ndarray:
        raise LLMError("Groq has no embedding models; the local embedder is used for schema retrieval.")


def parse_json(text: str) -> Optional[dict]:
    """Leniently parse a JSON object out of an LLM reply (handles code fences and extra prose)."""
    if not text:
        return None
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S | re.I)
    if fence:
        t = fence.group(1).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None
