"""Orchestrates the embedder, FAISS index, and KB metadata into a recommender."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np

from . import faiss_index
from .embeddings import Embedder
from .preprocessing import normalize_text


@dataclass
class Recommendation:
    drug_id: str
    name: str
    drug_class: str
    indications: str
    description: str
    score: float

    def to_dict(self) -> Dict:
        return {
            "drug_id": self.drug_id,
            "name": self.name,
            "drug_class": self.drug_class,
            "indications": self.indications,
            "description": self.description,
            "score": round(float(self.score), 4),
        }


class Recommender:
    def __init__(self, embedder: Embedder, kb_meta: List[Dict], index):
        self.embedder = embedder
        self.kb_meta = kb_meta
        self.index = index

    @classmethod
    def from_artifacts(
        cls,
        embedder: Embedder,
        index_path: Path,
        kb_meta_path: Path,
    ) -> "Recommender":
        index = faiss_index.load_index(index_path)
        with open(kb_meta_path) as f:
            kb_meta = json.load(f)
        return cls(embedder=embedder, kb_meta=kb_meta, index=index)

    def recommend(self, query: str, top_k: int = 5) -> List[Recommendation]:
        query = normalize_text(query)
        if not query:
            return []
        vec = self.embedder.encode([query])
        ids, scores = faiss_index.search(self.index, vec, top_k)
        recs: List[Recommendation] = []
        for idx, score in zip(ids, scores):
            if idx < 0 or idx >= len(self.kb_meta):
                continue
            row = self.kb_meta[idx]
            recs.append(
                Recommendation(
                    drug_id=row["drug_id"],
                    name=row["name"],
                    drug_class=row.get("drug_class", ""),
                    indications=row.get("indications", ""),
                    description=row.get("description", ""),
                    score=score,
                )
            )
        return recs
