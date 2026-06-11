"""Build FAISS index, transformer embeddings, and TF-IDF artifacts from the KB."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.data_loader import load_kb, kb_to_records  # noqa: E402
from src.preprocessing import build_document  # noqa: E402
from src.embeddings import Embedder  # noqa: E402
from src import faiss_index  # noqa: E402
from src.baseline_tfidf import TfidfBaseline  # noqa: E402


def main() -> None:
    print(f"[1/5] Loading KB from {config.KB_CSV}")
    df = load_kb(config.KB_CSV)
    records = kb_to_records(df)
    print(f"      {len(records)} drug records loaded")

    print("[2/5] Building searchable documents")
    documents = [build_document(r) for r in tqdm(records)]

    print(f"[3/5] Encoding documents with {config.EMBEDDING_MODEL}")
    embedder = Embedder(config.EMBEDDING_MODEL)
    embeddings = embedder.encode(documents)
    np.save(config.EMBEDDINGS_PATH, embeddings)
    print(f"      embeddings: shape={embeddings.shape} -> {config.EMBEDDINGS_PATH}")

    print("[4/5] Building FAISS index")
    index = faiss_index.build_index(embeddings)
    faiss_index.save_index(index, config.FAISS_INDEX_PATH)
    print(f"      index -> {config.FAISS_INDEX_PATH}")

    print("[5/5] Fitting TF-IDF baseline")
    tfidf = TfidfBaseline()
    tfidf.fit(documents)
    tfidf.save(config.TFIDF_VECTORIZER_PATH, config.TFIDF_MATRIX_PATH)
    print(f"      tfidf -> {config.TFIDF_VECTORIZER_PATH}, {config.TFIDF_MATRIX_PATH}")

    with open(config.KB_META_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"      kb meta -> {config.KB_META_PATH}")

    print("Done. You can now run `python -m scripts.evaluate` or `python -m api.app`.")


if __name__ == "__main__":
    main()
