"""Retriever interface with access-control filtering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class Retriever(ABC):
    name = "base"

    def fit(self, chunks: list[Chunk]) -> Retriever:
        self.chunks = chunks
        self._fit([c.text for c in chunks])
        return self

    @abstractmethod
    def _fit(self, texts: list[str]) -> None: ...

    @abstractmethod
    def scores(self, query: str) -> np.ndarray:
        """Relevance of every chunk for the query (higher is better)."""

    def search(self, query: str, k: int = 4, groups: set[str] | None = None) -> list[Hit]:
        """Top-k chunks the caller is allowed to see.

        Access control is enforced *inside* retrieval: chunks of documents the user
        has no group for are never ranked, so they cannot leak into the prompt.
        """
        s = self.scores(query).astype(float)
        if groups is not None:
            allowed = np.array([bool(set(c.groups) & groups) for c in self.chunks])
            s = np.where(allowed, s, -np.inf)
        order = np.argsort(-s, kind="stable")[:k]
        return [
            Hit(self.chunks[i], float(s[i]), r + 1)
            for r, i in enumerate(order)
            if np.isfinite(s[i])
        ]
