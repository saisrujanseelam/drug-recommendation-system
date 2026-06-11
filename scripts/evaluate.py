"""Compare transformer + FAISS retriever against TF-IDF baseline."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.embeddings import Embedder  # noqa: E402
from src.recommender import Recommender  # noqa: E402
from src.baseline_tfidf import TfidfBaseline  # noqa: E402
from src.evaluation import evaluate_retriever, relative_improvement  # noqa: E402


def _format_header() -> str:
    return f"{'Retriever':<20} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'MRR':>8}"


def _format_row(label: str, recall1: float, recall3: float, recall5: float, mrr: float) -> str:
    return f"{label:<20} {recall1:>8.4f} {recall3:>8.4f} {recall5:>8.4f} {mrr:>8.4f}"


def main() -> None:
    if not config.EVAL_QUERIES.exists():
        raise SystemExit(f"Missing eval queries at {config.EVAL_QUERIES}")
    if not config.FAISS_INDEX_PATH.exists():
        raise SystemExit("Artifacts missing. Run `python -m scripts.build_index` first.")

    with open(config.EVAL_QUERIES) as f:
        eval_set = json.load(f)
    queries = eval_set["queries"]
    print(f"Loaded {len(queries)} eval queries")

    embedder = Embedder(config.EMBEDDING_MODEL)
    recommender = Recommender.from_artifacts(
        embedder=embedder,
        index_path=config.FAISS_INDEX_PATH,
        kb_meta_path=config.KB_META_PATH,
    )
    tfidf = TfidfBaseline.load(config.TFIDF_VECTORIZER_PATH, config.TFIDF_MATRIX_PATH)
    with open(config.KB_META_PATH) as f:
        kb_meta = json.load(f)

    def transformer_retrieve(query: str, k: int):
        return [r.drug_id for r in recommender.recommend(query, top_k=k)]

    def tfidf_retrieve(query: str, k: int):
        ids, _ = tfidf.search(query, top_k=k)
        return [kb_meta[i]["drug_id"] for i in ids]

    transformer_result = evaluate_retriever("transformer+faiss", queries, transformer_retrieve)
    tfidf_result = evaluate_retriever("tfidf", queries, tfidf_retrieve)

    header = _format_header()
    print()
    print(header)
    print("-" * len(header))
    print(
        _format_row(
            tfidf_result.name,
            tfidf_result.recall_at_1,
            tfidf_result.recall_at_3,
            tfidf_result.recall_at_5,
            tfidf_result.mrr,
        )
    )
    print(
        _format_row(
            transformer_result.name,
            transformer_result.recall_at_1,
            transformer_result.recall_at_3,
            transformer_result.recall_at_5,
            transformer_result.mrr,
        )
    )
    print()
    improvement = relative_improvement(
        transformer_result.recall_at_5, tfidf_result.recall_at_5
    )
    print(
        f"Transformer recall@5 vs TF-IDF baseline: "
        f"{transformer_result.recall_at_5:.4f} vs {tfidf_result.recall_at_5:.4f} "
        f"({improvement*100:+.1f}%)"
    )

    out_path = config.PROCESSED_DIR / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "transformer": transformer_result.to_dict(),
                "tfidf": tfidf_result.to_dict(),
                "recall@5_relative_improvement": improvement,
                "per_query_transformer": transformer_result.per_query,
                "per_query_tfidf": tfidf_result.per_query,
            },
            f,
            indent=2,
        )
    print(f"Full report -> {out_path}")


if __name__ == "__main__":
    main()
