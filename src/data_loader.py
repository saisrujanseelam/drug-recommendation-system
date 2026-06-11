"""Load the medical knowledge base CSV."""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import pandas as pd


def load_kb(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"drug_id", "name", "indications", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"KB missing required columns: {missing}")
    df = df.fillna("")
    return df


def kb_to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient="records")
