"""Settings from environment variables (12-factor; maps directly to a Helm ConfigMap/Secret)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@dataclass
class Settings:
    corpus_dir: str = field(default_factory=lambda: _env("RAG_CORPUS_DIR", "corpus/northwind"))
    retriever: str = field(default_factory=lambda: _env("RAG_RETRIEVER", "hybrid"))
    top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", "4")))
    min_relevance: float = field(default_factory=lambda: float(_env("RAG_MIN_RELEVANCE", "0.12")))
    max_tool_steps: int = field(default_factory=lambda: int(_env("RAG_MAX_TOOL_STEPS", "4")))

    # LLM backend: "extractive" (offline, deterministic) or "openai" (any OpenAI-compatible API)
    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "extractive"))
    llm_base_url: str | None = field(default_factory=lambda: _env("LLM_BASE_URL"))
    llm_api_key: str | None = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "meta/llama-3.1-8b-instruct"))
    azure_api_version: str | None = field(default_factory=lambda: _env("AZURE_OPENAI_API_VERSION"))

    # Optional dense retrieval through an OpenAI-compatible /embeddings endpoint
    embedding_base_url: str | None = field(default_factory=lambda: _env("EMBEDDING_BASE_URL"))
    embedding_api_key: str | None = field(default_factory=lambda: _env("EMBEDDING_API_KEY"))
    embedding_model: str | None = field(default_factory=lambda: _env("EMBEDDING_MODEL"))
