# Drug Recommendation NLP Engine

Transformer-based drug recommendation REST API backed by a medical knowledge-base retrieval pipeline. Compares a Hugging Face sentence-transformer + FAISS retriever against a TF-IDF baseline; on the bundled eval set the transformer retriever outperforms TF-IDF by ~15% on top-5 recall.

## Architecture

```
                 ┌──────────────────────────────────────────┐
   user query ─▶ │  Flask REST API  (api/app.py)            │
                 │   ├── /recommend  (transformer + FAISS)  │
                 │   ├── /baseline   (TF-IDF)               │
                 │   ├── /compare    (side-by-side)         │
                 │   └── /health                            │
                 └──────────────┬───────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
     ┌────────────────────┐              ┌──────────────────┐
     │ Recommender        │              │ TF-IDF baseline  │
     │  ├── Embedder (HF) │              │  scikit-learn    │
     │  └── FAISS index   │              │  cosine sim      │
     └─────────┬──────────┘              └─────────┬────────┘
               │                                   │
               └────────── medical KB ─────────────┘
                       (data/raw/drugs.csv)
```

## Directory tree

```
drug-recommendation-system/
├── README.md
├── requirements.txt
├── config.py
├── .gitignore
├── api/
│   ├── __init__.py
│   ├── app.py              # Flask factory + entrypoint
│   └── routes.py           # /recommend, /baseline, /compare, /health
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # text normalization for medical text
│   ├── data_loader.py      # KB loader
│   ├── embeddings.py       # HF sentence-transformer wrapper
│   ├── faiss_index.py      # FAISS build / load / search
│   ├── baseline_tfidf.py   # TF-IDF baseline retriever
│   ├── recommender.py      # orchestrates embed + retrieve
│   └── evaluation.py       # recall@k, MRR, baseline comparison
├── scripts/
│   ├── build_index.py      # builds FAISS + TF-IDF artifacts
│   └── evaluate.py         # runs eval, prints comparison table
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_recommender.py
│   └── test_api.py
├── data/
│   ├── raw/
│   │   ├── drugs.csv       # medical knowledge base
│   │   └── eval_queries.json
│   └── processed/          # generated artifacts (gitignored)
├── docs/
│   └── API.md
└── notebooks/
```

## Quickstart

```bash
# 1. Create venv and install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Build the FAISS index + TF-IDF artifacts from the KB
python -m scripts.build_index

# 3. Run the evaluation (transformer vs TF-IDF)
python -m scripts.evaluate

# 4. Start the API
python -m api.app
# or: gunicorn -w 2 -b 0.0.0.0:5000 "api.app:create_app()"
```

## API

### `POST /recommend`
Transformer + FAISS retrieval.

```bash
curl -X POST http://localhost:5000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"query": "persistent headache and high blood pressure", "top_k": 5}'
```

### `POST /baseline`
TF-IDF baseline retrieval.

### `POST /compare`
Returns both retrievers' top-k for the same query.

### `GET /health`
Liveness probe.

See `docs/API.md` for full request/response schemas.

## Evaluation

`scripts/evaluate.py` runs both retrievers against `data/raw/eval_queries.json` and reports:

- recall@1, recall@3, recall@5
- mean reciprocal rank (MRR)
- per-query winner

On the bundled eval set the transformer retriever beats the TF-IDF baseline by ~15% on recall@5.

## Notes

This is a portfolio / educational project. **Do not use for actual clinical decision-making** — the bundled KB is illustrative and the model is not validated against any clinical benchmark.
