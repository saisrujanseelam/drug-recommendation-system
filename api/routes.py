"""REST endpoints for the recommender API."""
from __future__ import annotations
from flask import Blueprint, jsonify, request, current_app

import config

bp = Blueprint("api", __name__)


def _parse_top_k(payload: dict) -> int:
    top_k = int(payload.get("top_k", config.DEFAULT_TOP_K))
    return max(1, min(top_k, config.MAX_TOP_K))


@bp.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": config.EMBEDDING_MODEL,
            "kb_size": len(current_app.kb_meta),
        }
    )


@bp.post("/recommend")
def recommend():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Field 'query' is required"}), 400
    top_k = _parse_top_k(data)
    recs = current_app.recommender.recommend(query, top_k=top_k)
    return jsonify(
        {
            "query": query,
            "retriever": "transformer+faiss",
            "top_k": top_k,
            "results": [r.to_dict() for r in recs],
        }
    )


@bp.post("/baseline")
def baseline():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Field 'query' is required"}), 400
    top_k = _parse_top_k(data)
    ids, scores = current_app.tfidf.search(query, top_k=top_k)
    results = []
    for idx, score in zip(ids, scores):
        row = current_app.kb_meta[idx]
        results.append(
            {
                "drug_id": row["drug_id"],
                "name": row["name"],
                "drug_class": row.get("drug_class", ""),
                "indications": row.get("indications", ""),
                "score": round(float(score), 4),
            }
        )
    return jsonify(
        {
            "query": query,
            "retriever": "tfidf",
            "top_k": top_k,
            "results": results,
        }
    )


@bp.post("/compare")
def compare():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Field 'query' is required"}), 400
    top_k = _parse_top_k(data)
    transformer_results = [
        r.to_dict() for r in current_app.recommender.recommend(query, top_k=top_k)
    ]
    ids, scores = current_app.tfidf.search(query, top_k=top_k)
    tfidf_results = []
    for idx, score in zip(ids, scores):
        row = current_app.kb_meta[idx]
        tfidf_results.append(
            {
                "drug_id": row["drug_id"],
                "name": row["name"],
                "score": round(float(score), 4),
            }
        )
    return jsonify(
        {
            "query": query,
            "top_k": top_k,
            "transformer": transformer_results,
            "tfidf": tfidf_results,
        }
    )
