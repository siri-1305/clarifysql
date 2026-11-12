"""LLM access layer (Gemini) with retries, model fallback, rate limiting and an on-disk cache.

Every other module talks to the `BaseLLM` interface only, so tests can plug in a fake LLM.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
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


def _clean_err(e: Exception) -> str:
    return re.sub(r"\s+", " ", str(e))[:400]


class GeminiLLM(BaseLLM):
    def __init__(self, settings: Settings):
        if not settings.api_key or settings.api_key.startswith("PASTE_YOUR"):
            raise LLMError("No API key found. Put your key after GEMINI_API_KEY= in the .env file (README step 5).")
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise LLMError("The 'google-genai' package is missing. Run: pip install -r requirements.txt") from e
        self._types = types
        self.client = genai.Client(api_key=settings.api_key)
        self.s = settings
        self.chat_models = list(settings.chat_models)
        self.embed_models = list(settings.embed_models)
        self.model_idx = 0
        self.embed_idx = 0
        self.embed_model_name = self.embed_models[0] if self.embed_models else "none"
        self.use_thinking = bool(settings.thinking_level)
        self.calls = 0
        self._lock = threading.Lock()
        self._last_call = 0.0
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

    def _key(self, prompt, system, json_mode, sample) -> str:
        raw = json.dumps([self.chat_models[0], system, prompt, json_mode, sample,
                          self.s.temperature, self.s.thinking_level], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _throttle(self):
        with self._lock:
            wait = self.s.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    @staticmethod
    def _retry_delay(msg: str, attempt: int) -> float:
        m = re.search(r"retry in ([\d.]+)\s*s", msg, re.I) or re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", msg)
        if m:
            return min(float(m.group(1)) + 1.0, 90.0)
        return min(2.0 * (2 ** attempt), 60.0)

    @staticmethod
    def _status(e: Exception) -> int:
        code = getattr(e, "code", None)
        if isinstance(code, int):
            return code
        m = re.search(r"\b(4\d\d|5\d\d)\b", str(e))
        return int(m.group(1)) if m else 0

    # ------------------------------------------------------------------ generation
    def generate(self, prompt: str, *, system: Optional[str] = None, json_mode: bool = False,
                 sample: int = 0) -> str:
        key = self._key(prompt, system, json_mode, sample)
        if self.s.use_cache and key in self._cache:
            return self._cache[key]

        last_err = ""
        for attempt in range(self.s.max_retries):
            cfg = {}
            if system:
                cfg["system_instruction"] = system
            if json_mode:
                cfg["response_mime_type"] = "application/json"
            if self.s.temperature is not None:
                cfg["temperature"] = self.s.temperature
            if self.use_thinking and self.s.thinking_level:
                cfg["thinking_config"] = self._types.ThinkingConfig(thinking_level=self.s.thinking_level)
            try:
                self._throttle()
                self.calls += 1
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt,
                    config=self._types.GenerateContentConfig(**cfg))
                text = (resp.text or "").strip()
                if not text:
                    raise LLMError("empty response from model")
                if self.s.use_cache:
                    self._cache[key] = text
                    with self._lock, open(self._cache_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"k": key, "v": text}, ensure_ascii=False) + "\n")
                return text
            except Exception as e:  # noqa: BLE001 - SDK raises many different types
                msg = _clean_err(e)
                last_err = msg
                status = self._status(e)
                low = msg.lower()
                log.warning("LLM call failed (attempt %d, status %s): %s", attempt + 1, status, msg)
                if "thinking" in low and self.use_thinking:
                    self.use_thinking = False           # model does not accept thinking_level
                    continue
                if status in (401, 403) or "api key" in low and "not valid" in low:
                    raise LLMError("The API key was rejected. Check GEMINI_API_KEY in your .env file.") from e
                if status == 404 or "not found" in low or "no longer available" in low or "is not supported" in low:
                    if self.model_idx + 1 < len(self.chat_models):
                        log.warning("Model %s unavailable, switching to %s", self.model, self.chat_models[self.model_idx + 1])
                        self.model_idx += 1
                        continue
                    raise LLMError(f"Model '{self.model}' is not available. Set GEMINI_MODELS in .env to a current model "
                                   f"(see https://ai.google.dev/gemini-api/docs/models). Details: {msg}") from e
                if status in (429, 500, 502, 503, 504) or "overloaded" in low or "unavailable" in low or "exhausted" in low or "empty response" in low:
                    time.sleep(self._retry_delay(msg, attempt))
                    continue
                if status == 400:
                    raise LLMError(f"The request was rejected by the API: {msg}") from e
                time.sleep(self._retry_delay(msg, attempt))
        raise LLMError(f"LLM call failed after {self.s.max_retries} attempts: {last_err}")

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (len(texts), dim). Tries each embedding model in turn."""
        last = ""
        while self.embed_idx < len(self.embed_models):
            model = self.embed_models[self.embed_idx]
            try:
                vecs = self._embed_with(model, texts)
                self.embed_model_name = model
                return vecs
            except Exception as e:  # noqa: BLE001
                last = _clean_err(e)
                log.warning("Embedding model %s failed: %s", model, last)
                status = self._status(e)
                if status == 429 or status >= 500:
                    time.sleep(self._retry_delay(last, 0))
                    try:
                        vecs = self._embed_with(model, texts)
                        self.embed_model_name = model
                        return vecs
                    except Exception as e2:  # noqa: BLE001
                        last = _clean_err(e2)
                self.embed_idx += 1
        raise LLMError(f"No embedding model worked: {last}")

    def _embed_with(self, model: str, texts: List[str]) -> np.ndarray:
        out: List[List[float]] = []
        for i in range(0, len(texts), 50):
            chunk = texts[i:i + 50]
            self._throttle()
            self.calls += 1
            resp = self.client.models.embed_content(model=model, contents=chunk)
            embs = resp.embeddings or []
            if len(embs) == len(chunk):
                out.extend([list(e.values) for e in embs])
            else:  # some models merge a list into one vector -> embed one by one
                for t in chunk:
                    self._throttle()
                    self.calls += 1
                    r = self.client.models.embed_content(model=model, contents=t)
                    out.append(list(r.embeddings[0].values))
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


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
