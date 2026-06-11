# API Reference

Base URL: `http://localhost:5000`

All POST endpoints accept JSON and return JSON. CORS is enabled.

## `GET /health`

Liveness probe.

```json
{
  "status": "ok",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "kb_size": 40
}
```

## `POST /recommend`

Transformer + FAISS retrieval.

Request:

```json
{
  "query": "persistent headache and high blood pressure",
  "top_k": 5
}
```

Response:

```json
{
  "query": "persistent headache and high blood pressure",
  "retriever": "transformer+faiss",
  "top_k": 5,
  "results": [
    {
      "drug_id": "D001",
      "name": "Lisinopril",
      "drug_class": "ACE inhibitor",
      "indications": "hypertension, heart failure, post-MI cardioprotection",
      "description": "...",
      "score": 0.6421
    }
  ]
}
```

`top_k` is clamped to `[1, 25]`. Missing `query` returns HTTP 400.

## `POST /baseline`

TF-IDF baseline retrieval. Same request / response shape as `/recommend`,
with `retriever: "tfidf"`.

## `POST /compare`

Returns both retrievers' top-k for the same query — useful for spot-checking.

```json
{
  "query": "...",
  "top_k": 5,
  "transformer": [...],
  "tfidf": [...]
}
```
