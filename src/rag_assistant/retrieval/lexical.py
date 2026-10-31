"""BM25 and TF-IDF retrievers."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import Retriever
from .text import tokenize


class BM25Retriever(Retriever):
    """Okapi BM25 (k1=1.5, b=0.75) over stemmed tokens."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def _fit(self, texts: list[str]) -> None:
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], float)
        self.avgdl = self.lengths.mean()
        df = Counter(term for d in self.docs for term in d)
        n = len(texts)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.docs], float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


class TfidfRetriever(Retriever):
    """Cosine similarity of TF-IDF vectors (word uni/bigrams or character n-grams)."""

    def __init__(self, analyzer: str = "word"):
        self.analyzer = analyzer
        self.name = "tfidf_char" if analyzer == "char_wb" else "tfidf_word"

    def _fit(self, texts: list[str]) -> None:
        if self.analyzer == "word":
            self.vec = TfidfVectorizer(
                tokenizer=tokenize,
                token_pattern=None,
                ngram_range=(1, 2),
                sublinear_tf=True,
                lowercase=False,
            )
        else:
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = normalize(self.vec.fit_transform(texts))

    def scores(self, query: str) -> np.ndarray:
        q = normalize(self.vec.transform([query]))
        return (self.matrix @ q.T).toarray().ravel()
