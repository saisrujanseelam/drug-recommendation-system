"""Project-wide configuration."""
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

KB_CSV = RAW_DIR / "drugs.csv"
EVAL_QUERIES = RAW_DIR / "eval_queries.json"

FAISS_INDEX_PATH = PROCESSED_DIR / "drugs.faiss"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
KB_META_PATH = PROCESSED_DIR / "kb_meta.json"
TFIDF_VECTORIZER_PATH = PROCESSED_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = PROCESSED_DIR / "tfidf_matrix.npz"

# Sentence-transformer model. Small + fast default; swap for a clinical model
# such as "pritamdeka/S-PubMedBert-MS-MARCO" for production use.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_TOP_K = 25

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5000"))
API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"
