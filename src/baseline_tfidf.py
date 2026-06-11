"""TF-IDF baseline retriever for comparison against the transformer pipeline."""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import pickle

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfidfBaseline:
    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: sparse.csr_matrix | None = None

    def fit(self, documents: List[str]) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int) -> Tuple[List[int], List[float]]:
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("TfidfBaseline is not fitted")
        q_vec = self.vectorizer.transform([query])
        scores = linear_kernel(q_vec, self.matrix).ravel()
        if top_k >= len(scores):
            order = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, top_k)[:top_k]
            order = top_idx[np.argsort(-scores[top_idx])]
        return order.tolist(), scores[order].tolist()

    def save(self, vectorizer_path: Path, matrix_path: Path) -> None:
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        sparse.save_npz(matrix_path, self.matrix)

    @classmethod
    def load(cls, vectorizer_path: Path, matrix_path: Path) -> "TfidfBaseline":
        obj = cls()
        with open(vectorizer_path, "rb") as f:
            obj.vectorizer = pickle.load(f)
        obj.matrix = sparse.load_npz(matrix_path)
        return obj
