from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    out_path: Path,
    normalize: bool,
    title: str,
) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    K = int(n_classes)
    labels = list(range(1, K + 1))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        denom = cm.sum(axis=1, keepdims=True).astype(float)
        cm_plot = np.divide(cm.astype(float), denom, out=np.zeros_like(cm, dtype=float), where=denom > 0)
        fmt = ".2f"
    else:
        cm_plot = cm.astype(int)
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels([str(i) for i in labels])
    ax.set_yticklabels([str(i) for i in labels])

    thresh = float(np.max(cm_plot) / 2.0) if cm_plot.size else 0.0
    for i in range(K):
        for j in range(K):
            val = cm_plot[i, j]
            txt = format(val, fmt)
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if float(val) > thresh else "black",
                fontsize=8,
            )

    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _x_jitter_by_group(*, x: np.ndarray, width: float = 0.18, seed: int = 0) -> np.ndarray:
    """Deterministic jitter for discrete x to reduce overplotting.

    Important: jitter MUST NOT depend on y-values; otherwise points will visually form slanted/curved bands.
    """
    x = np.asarray(x).astype(float)
    out = x.astype(float).copy()
    for g in np.unique(x).tolist():
        m = x == float(g)
        idx = np.where(m)[0]
        if idx.size <= 1:
            continue
        # Evenly spaced jitter, shuffled in a deterministic way that depends only on (seed, group).
        jit = np.linspace(-float(width), float(width), num=int(idx.size), dtype=float)
        rng = np.random.RandomState(int(seed) + int(round(float(g))) * 9973)
        jit = jit[rng.permutation(int(idx.size))]
        out[idx] = float(g) + jit
    return out


def save_ycont_scatter(
    *,
    y_true: np.ndarray,
    y_cont: np.ndarray,
    out_path: Path,
    title: str,
    y_pred: np.ndarray | None = None,
) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_cont = np.asarray(y_cont).astype(float)
    x = _x_jitter_by_group(x=y_true)

    fig, ax = plt.subplots(figsize=(6, 5))
    if y_pred is None:
        ax.scatter(x, y_cont, s=18, alpha=0.7, color="tab:blue", edgecolors="none")
    else:
        y_pred_i = np.asarray(y_pred).astype(int)
        abs_err = np.abs(y_true - y_pred_i)
        m0 = abs_err == 0
        m1 = abs_err == 1
        m2 = abs_err >= 2
        if int(m2.sum()) > 0:
            ax.scatter(x[m2], y_cont[m2], s=10, alpha=0.15, color="tab:gray", edgecolors="none", label="|err|>=2")
        if int(m1.sum()) > 0:
            ax.scatter(x[m1], y_cont[m1], s=14, alpha=0.30, color="tab:orange", edgecolors="none", label="|err|=1")
        if int(m0.sum()) > 0:
            ax.scatter(x[m0], y_cont[m0], s=26, alpha=0.85, color="tab:blue", edgecolors="none", label="|err|=0")
        ax.legend(loc="best", frameon=True, fontsize=8)

    ax.set_xlabel("True grade")
    ax.set_ylabel("Continuous output (model)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _t_crit_975(df: int) -> float:
    try:
        from scipy.stats import t as _t

        return float(_t.ppf(0.975, df=int(df)))
    except Exception:
        return 2.0


def save_performance_stability_plot(
    *,
    by_fold: pd.DataFrame,
    out_path: Path,
    metrics: list[str],
    title: str,
) -> None:
    df = by_fold.copy()
    if "fold" not in df.columns:
        raise ValueError("by_fold missing 'fold' column")
    n_folds = int(df.shape[0])
    if n_folds < 2:
        raise ValueError("Need >=2 folds for stability plot")

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.8), squeeze=False)
    tcrit = _t_crit_975(n_folds - 1)
    for j, m in enumerate(metrics):
        ax = axes[0, j]
        vals = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float)
        x = np.arange(1, n_folds + 1, dtype=float)
        ax.scatter(x, vals, s=32, alpha=0.85, color="tab:blue", edgecolors="none")
        mean = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=1))
        ci = tcrit * sd / math.sqrt(float(n_folds)) if np.isfinite(sd) else 0.0
        ax.axhline(mean, color="tab:red", linewidth=1.5)
        ax.axhspan(mean - ci, mean + ci, color="tab:red", alpha=0.12, linewidth=0.0)
        ax.set_title(m)
        ax.set_xlabel("Fold")
        ax.set_xticks(x)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.text(
            0.02,
            0.98,
            f"mean={mean:.4f}\n95%CI=[{mean-ci:.4f},{mean+ci:.4f}]",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
        )

    fig.suptitle(title, y=1.03, fontsize=12)
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_abs_error_distribution(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    out_path: Path,
    title: str,
) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    abs_err = np.abs(y_true - y_pred).astype(int)
    max_d = int(max(0, int(n_classes) - 1))
    xs = np.arange(0, max_d + 1, dtype=int)
    counts = np.array([(abs_err == d).sum() for d in xs], dtype=float)
    denom = float(counts.sum()) if counts.size else 1.0
    frac = counts / denom

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(xs, frac, color="tab:blue", alpha=0.85)
    ax.set_xlabel("|y_true - y_pred|")
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    for x, f in zip(xs.tolist(), frac.tolist()):
        ax.text(x, f + 0.01, f"{f:.2f}", ha="center", va="bottom", fontsize=8)
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
