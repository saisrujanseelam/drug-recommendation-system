"""Flask app factory and entrypoint."""
from __future__ import annotations
import json
from flask import Flask
from flask_cors import CORS

import config
from src.embeddings import Embedder
from src.recommender import Recommender
from src.baseline_tfidf import TfidfBaseline
from .routes import bp as api_bp


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    if not config.FAISS_INDEX_PATH.exists() or not config.KB_META_PATH.exists():
        raise RuntimeError(
            "Artifacts missing. Run `python -m scripts.build_index` first."
        )

    embedder = Embedder(config.EMBEDDING_MODEL)
    recommender = Recommender.from_artifacts(
        embedder=embedder,
        index_path=config.FAISS_INDEX_PATH,
        kb_meta_path=config.KB_META_PATH,
    )
    tfidf = TfidfBaseline.load(
        vectorizer_path=config.TFIDF_VECTORIZER_PATH,
        matrix_path=config.TFIDF_MATRIX_PATH,
    )
    with open(config.KB_META_PATH) as f:
        kb_meta = json.load(f)

    app.recommender = recommender  # type: ignore[attr-defined]
    app.tfidf = tfidf  # type: ignore[attr-defined]
    app.kb_meta = kb_meta  # type: ignore[attr-defined]

    app.register_blueprint(api_bp)
    return app


if __name__ == "__main__":
    create_app().run(host=config.API_HOST, port=config.API_PORT, debug=config.API_DEBUG)
