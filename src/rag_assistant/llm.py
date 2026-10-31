"""LLM backends behind one small interface.

* ``OpenAICompatibleLLM`` works with any server that speaks the OpenAI Chat
  Completions API: NVIDIA NIM, Azure OpenAI / AI Foundry, vLLM, TGI, Ollama.
* ``ExtractiveLLM`` is an offline, deterministic stand-in that answers by
  selecting the most relevant source sentences. It keeps tests, CI and demos
  free of API keys and makes the evaluation reproducible.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .retrieval.text import tokenize


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM:
    supports_tools = False

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        raise NotImplementedError


class OpenAICompatibleLLM(LLM):
    supports_tools = True

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        azure_api_version: str | None = None,
        temperature: float = 0.0,
    ):
        if azure_api_version:
            from openai import AzureOpenAI

            self.client = AzureOpenAI(
                azure_endpoint=base_url, api_key=api_key, api_version=azure_api_version
            )
        else:
            from openai import OpenAI

            self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self.model, self.temperature = model, temperature

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            tools=tools or None,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(c.id, c.function.name, json.loads(c.function.arguments or "{}"))
            for c in (msg.tool_calls or [])
        ]
        return LLMResponse(msg.content, calls)


_SOURCE = re.compile(r"^\[(\d+)\] [^\n]*\n(.*?)(?=^\[\d+\] |\Z)", re.S | re.M)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class ExtractiveLLM(LLM):
    """Answers with the 1-2 source sentences that best match the question."""

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        prompt = messages[-1]["content"]
        question = prompt.split("QUESTION:", 1)[-1].strip()
        context = prompt.split("SOURCES:", 1)[-1].split("QUESTION:", 1)[0]
        candidates = []
        for m in _SOURCE.finditer(context):
            n, body = int(m.group(1)), m.group(2)
            body = body.split("\n", 1)[-1]  # drop the "Title. Section." line
            for line in body.splitlines():  # bullets are separate statements
                for sent in _SENTENCE.split(line.strip().lstrip("-* ")):
                    if sent.strip():
                        candidates.append((n, sent.strip()))
        if not candidates:
            return LLMResponse("I don't know based on the available documents.")
        q = set(tokenize(question))
        df = Counter(t for _, s in candidates for t in set(tokenize(s)))
        idf = {t: math.log(1 + len(candidates) / df[t]) for t in df}
        scored = []
        for i, (n, s) in enumerate(candidates):
            toks = set(tokenize(s))
            # lexical overlap, discounted by the retriever's rank of the source
            score = sum(idf[t] for t in q & toks) / (1.0 + 0.25 * (n - 1))
            scored.append((score - 0.001 * i, n, s))  # tie-break by source order
        best = [x for x in sorted(scored, reverse=True)[: self.max_sentences] if x[0] > 0]
        if not best:
            return LLMResponse("I don't know based on the available documents.")
        return LLMResponse(" ".join(f"{s} [{n}]" for _, n, s in best))


def build_llm(settings) -> LLM:
    if settings.llm_backend == "extractive":
        return ExtractiveLLM()
    if settings.llm_backend == "openai":
        return OpenAICompatibleLLM(
            settings.llm_model,
            settings.llm_base_url,
            settings.llm_api_key,
            settings.azure_api_version,
        )
    raise ValueError(f"unknown LLM backend {settings.llm_backend!r}")
