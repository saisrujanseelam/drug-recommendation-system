"""Generate result charts from the evaluation report."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import config  # noqa: E402


TFIDF_COLOR = "#94a3b8"
TX_COLOR = "#2563eb"
WIN_COLOR = "#16a34a"
LOSS_COLOR = "#dc2626"
TIE_COLOR = "#cbd5e1"


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        }
    )


def plot_aggregate(report: dict, out_path: Path) -> None:
    metrics = ["recall@1", "recall@3", "recall@5", "mrr"]
    tfidf_vals = [report["tfidf"][m] for m in metrics]
    tx_vals = [report["transformer"][m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=140)
    bars1 = ax.bar(x - width / 2, tfidf_vals, width, label="TF-IDF baseline", color=TFIDF_COLOR)
    bars2 = ax.bar(x + width / 2, tx_vals, width, label="Transformer + FAISS", color=TX_COLOR)
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.012,
                f"{b.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval metrics — Transformer + FAISS vs TF-IDF baseline\n(20-query medical KB eval set)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_query(report: dict, out_path: Path) -> None:
    tx_q = {q["query"]: q for q in report["per_query_transformer"]}
    tf_q = {q["query"]: q for q in report["per_query_tfidf"]}
    rows = []
    for q, a in tf_q.items():
        b = tx_q[q]
        rows.append((q, a["recall@5"], b["recall@5"]))
    rows.sort(key=lambda r: (r[2] - r[1]), reverse=True)

    labels = [r[0][:62] + ("…" if len(r[0]) > 62 else "") for r in rows]
    tfidf_vals = [r[1] for r in rows]
    tx_vals = [r[2] for r in rows]

    y = np.arange(len(rows))
    height = 0.4

    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=140)
    ax.barh(y - height / 2, tfidf_vals, height, label="TF-IDF baseline", color=TFIDF_COLOR)
    ax.barh(y + height / 2, tx_vals, height, label="Transformer + FAISS", color=TX_COLOR)

    for i, (_, a, b) in enumerate(rows):
        if b > a:
            ax.scatter(1.04, i, marker="^", color=WIN_COLOR, s=70, clip_on=False)
        elif b < a:
            ax.scatter(1.04, i, marker="v", color=LOSS_COLOR, s=70, clip_on=False)
        else:
            ax.scatter(1.04, i, marker="o", color=TIE_COLOR, s=40, clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Recall@5")
    ax.set_title("Per-query Recall@5 — sorted by transformer advantage\n▲ transformer win   ▼ TF-IDF win   ● tie")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_relative_improvement(report: dict, out_path: Path) -> None:
    metrics = ["recall@1", "recall@3", "recall@5", "mrr"]
    rel = []
    for m in metrics:
        baseline = report["tfidf"][m]
        challenger = report["transformer"][m]
        rel.append((challenger - baseline) / baseline * 100 if baseline else 0.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=140)
    colors = [WIN_COLOR if v >= 0 else LOSS_COLOR for v in rel]
    bars = ax.bar([m.upper() for m in metrics], rel, color=colors, width=0.55)
    for b, v in zip(bars, rel):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + (0.15 if v >= 0 else -0.4),
            f"{v:+.1f}%",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )
    ax.axhline(0, color="#0f172a", linewidth=0.8)
    ax.set_ylabel("Relative improvement (%)")
    ax.set_title("Transformer + FAISS lift over TF-IDF baseline (aggregate)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    report_path = config.PROCESSED_DIR / "eval_report.json"
    if not report_path.exists():
        raise SystemExit(
            f"Missing {report_path}. Run `python -m scripts.evaluate` first."
        )
    with open(report_path) as f:
        report = json.load(f)

    out_dir = ROOT / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()
    print(f"Generating result charts -> {out_dir}")
    plot_aggregate(report, out_dir / "metrics_comparison.png")
    plot_per_query(report, out_dir / "per_query_recall.png")
    plot_relative_improvement(report, out_dir / "relative_improvement.png")
    print("Done.")


if __name__ == "__main__":
    main()
