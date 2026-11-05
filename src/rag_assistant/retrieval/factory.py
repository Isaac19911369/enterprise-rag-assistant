from __future__ import annotations

from ..config import Settings
from .base import Retriever
from .hybrid import HybridRetriever
from .lexical import BM25Retriever, TfidfRetriever

RETRIEVERS = ("bm25", "tfidf_word", "tfidf_char", "hybrid", "dense", "hybrid_dense")


def _dense(settings: Settings) -> Retriever:
    from openai import OpenAI

    from .dense import EmbeddingRetriever

    if not settings.embedding_model:
        raise ValueError("dense retrieval needs EMBEDDING_MODEL (and EMBEDDING_BASE_URL)")
    client = OpenAI(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        max_retries=6,
        timeout=120,
    )
    return EmbeddingRetriever(client, settings.embedding_model)


def build_retriever(name: str, settings: Settings | None = None) -> Retriever:
    settings = settings or Settings()
    if name == "bm25":
        return BM25Retriever()
    if name == "tfidf_word":
        return TfidfRetriever("word")
    if name == "tfidf_char":
        return TfidfRetriever("char_wb")
    if name == "hybrid":
        return HybridRetriever([BM25Retriever(), TfidfRetriever("char_wb")])
    if name == "dense":
        return _dense(settings)
    if name == "hybrid_dense":
        return HybridRetriever([BM25Retriever(), _dense(settings)], name="hybrid_dense")
    raise ValueError(f"unknown retriever {name!r}; choose from {RETRIEVERS}")
