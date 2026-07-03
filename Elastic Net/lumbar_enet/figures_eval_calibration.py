from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _ece(*, conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return 0.0, pd.DataFrame(columns=["bin", "n", "conf", "acc"])
    n_bins = int(n_bins)
    bin_id = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    n = float(conf.size)
    for b in range(n_bins):
        m = bin_id == b
        nb = int(m.sum())
        if nb == 0:
            continue
        cb = float(conf[m].mean())
        ab = float(correct[m].mean())
        w = float(nb) / n
        ece += w * float(abs(ab - cb))
        rows.append({"bin": int(b), "n": nb, "conf": cb, "acc": ab})
    return float(ece), pd.DataFrame(rows)


def _brier_multiclass(*, probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    K = int(n_classes)
    if probs.size == 0:
        return 0.0
    if probs.shape[1] != K:
        raise ValueError(f"probs must have shape (n,{K}) (got {probs.shape})")
    y_idx = np.clip(y_true - 1, 0, K - 1)
    one_hot = np.zeros_like(probs, dtype=float)
    one_hot[np.arange(probs.shape[0]), y_idx] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def save_reliability_diagram(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    n_classes: int,
    out_path: Path,
    title: str,
    n_bins: int = 10,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    probs = np.asarray(probs, dtype=float)
    K = int(n_classes)
    if probs.ndim != 2 or probs.shape[1] != K:
        raise ValueError(f"probs must have shape (n,{K}) (got {probs.shape})")

    pred_idx = np.clip(y_pred - 1, 0, K - 1)
    conf = probs[np.arange(probs.shape[0]), pred_idx]
    correct = (y_true == y_pred).astype(float)
    ece, bins_df = _ece(conf=conf, correct=correct, n_bins=int(n_bins))
    brier_top = float(np.mean((conf - correct) ** 2)) if conf.size else 0.0
    brier_multi = _brier_multiclass(probs=probs, y_true=y_true, n_classes=K)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0, label="Ideal")
    if not bins_df.empty:
        ax.plot(bins_df["conf"], bins_df["acc"], marker="o", color="tab:blue", linewidth=1.5, label="Observed")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence (p of predicted class)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="best", frameon=True, fontsize=8)
    ax.text(
        0.02,
        0.98,
        f"ECE={ece:.4f}\nBrier(top)={brier_top:.4f}\nBrier(multi)={brier_multi:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
    )

    ax2 = axes[1]
    ax2.hist(conf, bins=int(n_bins), range=(0.0, 1.0), color="tab:gray", alpha=0.85)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence histogram")
    ax2.grid(True, axis="y", linewidth=0.3, alpha=0.5)

    fig.suptitle(title, y=1.03, fontsize=12)
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "ece": float(ece),
        "brier_top": float(brier_top),
        "brier_multiclass": float(brier_multi),
        "n_bins": int(n_bins),
        "bins": bins_df.to_dict(orient="records"),
    }


def save_reliability_diagram_by_class(
    *,
    y_true: np.ndarray,
    probs: np.ndarray,
    n_classes: int,
    out_path: Path,
    title: str,
    n_bins: int = 10,
) -> dict[str, Any]:
    """One-vs-rest calibration curves for each class probability p(y=k)."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    K = int(n_classes)
    if probs.ndim != 2 or probs.shape[1] != K:
        raise ValueError(f"probs must have shape (n,{K}) (got {probs.shape})")

    n_cols = 3
    n_rows = int(np.ceil(K / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows), squeeze=False)
    summaries: list[dict[str, Any]] = []

    for k in range(1, K + 1):
        r = (k - 1) // n_cols
        c = (k - 1) % n_cols
        ax = axes[r, c]

        conf = probs[:, k - 1]
        true = (y_true == k).astype(float)
        ece, bins_df = _ece(conf=conf, correct=true, n_bins=int(n_bins))
        brier = float(np.mean((conf - true) ** 2)) if conf.size else 0.0

        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0)
        if not bins_df.empty:
            ax.plot(bins_df["conf"], bins_df["acc"], marker="o", color="tab:blue", linewidth=1.5)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Class {k}")
        ax.set_xlabel("Pred prob")
        ax.set_ylabel("Empirical freq")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.text(
            0.02,
            0.98,
            f"ECE={ece:.4f}\nBrier={brier:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
        )
        summaries.append({"class": int(k), "ece": float(ece), "brier": float(brier), "bins": bins_df.to_dict("records")})

    # Hide unused axes.
    for idx in range(K, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")

    fig.suptitle(title, y=1.02, fontsize=12)
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"type": "by_class", "n_bins": int(n_bins), "classes": summaries}


def save_reliability_diagram_cumulative(
    *,
    y_true: np.ndarray,
    p_gt: np.ndarray,
    n_classes: int,
    out_path: Path,
    title: str,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Calibration curves for cumulative probabilities p(y>k), k=1..K-1."""
    y_true = np.asarray(y_true).astype(int)
    p_gt = np.asarray(p_gt, dtype=float)
    K = int(n_classes)
    n_bounds = int(K - 1)
    if p_gt.ndim != 2 or p_gt.shape[1] != n_bounds:
        raise ValueError(f"p_gt must have shape (n,{n_bounds}) (got {p_gt.shape})")

    n_cols = 2
    n_rows = int(np.ceil(n_bounds / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.4 * n_rows), squeeze=False)
    summaries: list[dict[str, Any]] = []

    for kk in range(1, n_bounds + 1):
        r = (kk - 1) // n_cols
        c = (kk - 1) % n_cols
        ax = axes[r, c]

        conf = p_gt[:, kk - 1]
        true = (y_true > kk).astype(float)
        ece, bins_df = _ece(conf=conf, correct=true, n_bins=int(n_bins))
        brier = float(np.mean((conf - true) ** 2)) if conf.size else 0.0

        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0)
        if not bins_df.empty:
            ax.plot(bins_df["conf"], bins_df["acc"], marker="o", color="tab:blue", linewidth=1.5)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"P(y>{kk})")
        ax.set_xlabel("Pred prob")
        ax.set_ylabel("Empirical freq")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.text(
            0.02,
            0.98,
            f"ECE={ece:.4f}\nBrier={brier:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
        )
        summaries.append(
            {"boundary": int(kk), "ece": float(ece), "brier": float(brier), "bins": bins_df.to_dict("records")}
        )

    for idx in range(n_bounds, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")

    fig.suptitle(title, y=1.02, fontsize=12)
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"type": "cumulative", "n_bins": int(n_bins), "boundaries": summaries}

