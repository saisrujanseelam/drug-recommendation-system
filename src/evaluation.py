"""Retrieval evaluation: recall@k and MRR for both retrievers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Sequence


@dataclass
class EvalResult:
    name: str
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    per_query: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "recall@1": round(self.recall_at_1, 4),
            "recall@3": round(self.recall_at_3, 4),
            "recall@5": round(self.recall_at_5, 4),
            "mrr": round(self.mrr, 4),
        }


def _recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    hit = len(top & set(relevant))
    return hit / min(len(relevant), k)


def _reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    relevant_set = set(relevant)
    for rank, drug_id in enumerate(retrieved, start=1):
        if drug_id in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(
    name: str,
    queries: List[Dict],
    retrieve_fn: Callable[[str, int], List[str]],
    k_max: int = 5,
) -> EvalResult:
    """Run a retriever over labeled queries and return aggregated metrics.

    retrieve_fn(query, k) must return an ordered list of drug_ids.
    """
    result = EvalResult(name=name)
    r1 = r3 = r5 = mrr = 0.0
    n = len(queries)
    for q in queries:
        query_text = q["query"]
        relevant = q["relevant_ids"]
        retrieved = retrieve_fn(query_text, k_max)
        q_r1 = _recall_at_k(retrieved, relevant, 1)
        q_r3 = _recall_at_k(retrieved, relevant, 3)
        q_r5 = _recall_at_k(retrieved, relevant, 5)
        q_rr = _reciprocal_rank(retrieved, relevant)
        r1 += q_r1
        r3 += q_r3
        r5 += q_r5
        mrr += q_rr
        result.per_query.append(
            {
                "query": query_text,
                "retrieved": retrieved,
                "relevant": relevant,
                "recall@5": round(q_r5, 4),
                "reciprocal_rank": round(q_rr, 4),
            }
        )
    if n:
        result.recall_at_1 = r1 / n
        result.recall_at_3 = r3 / n
        result.recall_at_5 = r5 / n
        result.mrr = mrr / n
    return result


def relative_improvement(challenger: float, baseline: float) -> float:
    if baseline == 0:
        return float("inf") if challenger > 0 else 0.0
    return (challenger - baseline) / baseline
