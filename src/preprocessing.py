"""Text normalization for medical KB entries and user queries."""
from __future__ import annotations
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s\-/+%.]")


def normalize_text(text: str) -> str:
    """Light normalization that preserves medical tokens (mg, %, /, -)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_document(row: dict) -> str:
    """Concatenate KB fields into one searchable document.

    Indications are repeated because they carry the strongest signal for a
    user query like "I have high blood pressure".
    """
    parts = [
        row.get("name", ""),
        row.get("generic_name", ""),
        row.get("drug_class", ""),
        row.get("indications", ""),
        row.get("indications", ""),
        row.get("description", ""),
        row.get("side_effects", ""),
    ]
    return normalize_text(" ".join(str(p) for p in parts if p))
