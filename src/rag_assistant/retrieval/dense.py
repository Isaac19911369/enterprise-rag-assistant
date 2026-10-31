"""Dense retrieval through any OpenAI-compatible embeddings endpoint.

Works with NVIDIA NIM (e.g. ``nvidia/nv-embedqa-e5-v5``), Azure OpenAI
(``text-embedding-3-small``), vLLM or Ollama.
"""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from .base import Retriever


class EmbeddingRetriever(Retriever):
    name = "dense"

    def __init__(self, client: OpenAI, model: str, batch_size: int = 64):
        self.client, self.model, self.batch_size = client, model, batch_size

    def _embed(self, texts: list[str], input_type: str) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(
                model=self.model,
                input=texts[i : i + self.batch_size],
                # asymmetric models (NIM retrieval embedders) distinguish queries and passages
                extra_body={"input_type": input_type} if "nv-embed" in self.model else None,
            )
            vectors += [d.embedding for d in resp.data]
        v = np.asarray(vectors, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def _fit(self, texts: list[str]) -> None:
        self.matrix = self._embed(texts, "passage")

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self._embed([query], "query")[0]
