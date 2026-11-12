"""A scripted fake LLM so the whole pipeline can be tested without an API key or internet."""
from __future__ import annotations

import json
from typing import Callable, List, Optional

from clarifysql.llm import BaseLLM, LLMError


class FakeLLM(BaseLLM):
    def __init__(self, handler: Callable[[str, Optional[str], bool, int], str]):
        self.handler = handler
        self.calls = 0
        self.log: List[tuple] = []

    def generate(self, prompt, *, system=None, json_mode=False, sample=0):
        self.calls += 1
        out = self.handler(prompt, system, json_mode, sample)
        self.log.append((prompt, sample, out))
        return out

    def embed(self, texts):
        raise LLMError("no embeddings in fake LLM")


def js(**kw) -> str:
    return json.dumps(kw)
