"""Smoke test for the Flask API.

Skipped when artifacts have not been built yet (e.g., in a fresh clone).
"""
import json
import pytest

import config


@pytest.fixture(scope="module")
def client():
    if not config.FAISS_INDEX_PATH.exists() or not config.KB_META_PATH.exists():
        pytest.skip("Artifacts missing — run `python -m scripts.build_index` first")
    from api.app import create_app

    app = create_app()
    app.testing = True
    return app.test_client()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["kb_size"] > 0


def test_recommend_returns_results(client):
    resp = client.post(
        "/recommend",
        data=json.dumps({"query": "high blood pressure", "top_k": 3}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["retriever"] == "transformer+faiss"
    assert len(body["results"]) == 3
    assert all("drug_id" in r for r in body["results"])


def test_recommend_requires_query(client):
    resp = client.post(
        "/recommend",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_compare_returns_both(client):
    resp = client.post(
        "/compare",
        data=json.dumps({"query": "type 2 diabetes", "top_k": 3}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert "transformer" in body and "tfidf" in body
    assert len(body["transformer"]) == 3
    assert len(body["tfidf"]) == 3
