"""Reciprocal-rank fusion of several retrievers."""

from __future__ import annotations

import numpy as np

from .base import Retriever


class HybridRetriever(Retriever):
    """RRF: ``score(d) = sum_r 1 / (k + rank_r(d))``.

    Rank-based fusion needs no score calibration between retrievers, which is why
    it is the default way to combine lexical and dense retrieval.
    """

    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], k: int = 60, name: str = "hybrid"):
        self.retrievers, self.k, self.name = retrievers, k, name

    def fit(self, chunks):
        self.chunks = chunks
        for r in self.retrievers:
            r.fit(chunks)
        return self

    def _fit(self, texts):  # pragma: no cover - fit() is overridden
        raise NotImplementedError

    def scores(self, query: str) -> np.ndarray:
        fused = np.zeros(len(self.chunks))
        for r in self.retrievers:
            order = np.argsort(-r.scores(query), kind="stable")
            ranks = np.empty(len(order))
            ranks[order] = np.arange(1, len(order) + 1)
            fused += 1.0 / (self.k + ranks)
        return fused
