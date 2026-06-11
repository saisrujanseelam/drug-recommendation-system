"""FAISS index build / save / load / search.

Uses IndexFlatIP because embeddings are L2-normalized, so inner product equals
cosine similarity. Flat is fine at KB scales of <10k drugs.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss


def build_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def search(index: faiss.Index, query_vec: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    scores, ids = index.search(query_vec.astype(np.float32), top_k)
    return ids[0].tolist(), scores[0].tolist()
