from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lumbar_xgb.figures_explain_types import FoldShapContrib


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _shape_curve_from_samples(
    *,
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute y(x) curve using unique x-values with mean aggregation; downsample if needed."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x_v = x[valid]
    y_v = y[valid]
    if x_v.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    x_unique, inv = np.unique(x_v, return_inverse=True)
    sum_y = np.bincount(inv, weights=y_v, minlength=int(x_unique.size)).astype(float)
    cnt = np.bincount(inv, minlength=int(x_unique.size)).astype(float)
    y_mean = np.divide(sum_y, cnt, out=np.zeros_like(sum_y), where=cnt > 0)

    if int(max_points) > 0 and x_unique.size > int(max_points):
        idx = np.linspace(0, int(x_unique.size) - 1, num=int(max_points), dtype=float).round().astype(int)
        idx = np.unique(np.clip(idx, 0, int(x_unique.size) - 1))
        x_unique = x_unique[idx]
        y_mean = y_mean[idx]
    return x_unique.astype(float), y_mean.astype(float)


def save_explain_stability_plots(
    *,
    folds: list[FoldShapContrib],
    X_raw: pd.DataFrame,
    importance: pd.DataFrame,
    out_dir: Path,
    top_n: int = 20,
    max_points: int = 200,
) -> list[str]:
    """Shape-like curve stability: fold curves + mean±SD band for top features."""
    _ensure_dir(out_dir)
    per_fold_index = [{n: i for i, n in enumerate(fd.feature_names)} for fd in folds]
    common = set(folds[0].feature_names)
    for fd in folds[1:]:
        common &= set(fd.feature_names)

    selected = [f for f in importance["feature"].head(int(top_n)).tolist() if f in common]
    for feat in selected:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for k, fd in enumerate(folds):
            j = int(per_fold_index[k][feat])
            x = np.asarray(fd.feature_values[:, j], dtype=float)
            x_u, y_u = _shape_curve_from_samples(x=x, y=fd.phi_centered[:, j], max_points=int(max_points))
            xs.append(x_u)
            ys.append(y_u)

        x_ref = xs[0]
        if any(arr.shape != x_ref.shape or (arr.size and not np.allclose(arr, x_ref, equal_nan=True)) for arr in xs[1:]):
            x_ref = None  # type: ignore[assignment]

        fig, ax = plt.subplots(figsize=(6, 5))
        for k, _fd in enumerate(folds):
            ax.plot(xs[k], ys[k], color="tab:gray", alpha=0.25, linewidth=1.0, label=None if k else "Fold curves")
        if x_ref is not None and ys and len(ys) >= 1:
            y_stack = np.stack(ys, axis=0)
            y_mean = y_stack.mean(axis=0)
            y_sd = y_stack.std(axis=0, ddof=1) if y_stack.shape[0] >= 2 else np.zeros_like(y_mean)
            ax.plot(x_ref, y_mean, color="tab:blue", linewidth=2.5, label="Mean")
            ax.fill_between(x_ref, y_mean - y_sd, y_mean + y_sd, color="tab:blue", alpha=0.18, label="±1 SD")
        ax.set_title(f"Shape stability: {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel("Centered contribution (phi)")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.legend(loc="best", frameon=True, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"shape_stability_{feat}.png", dpi=200)
        plt.close(fig)

    # Quick grid overview (mean curve only).
    if selected:
        n = len(selected)
        n_cols = 4
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.8 * n_rows), squeeze=False)
        for idx, feat in enumerate(selected):
            r = idx // n_cols
            c = idx % n_cols
            ax = axes[r, c]
            ys = []
            x_ref = None
            for k, fd in enumerate(folds):
                j = int(per_fold_index[k][feat])
                x = np.asarray(fd.feature_values[:, j], dtype=float)
                x_u, y_u = _shape_curve_from_samples(x=x, y=fd.phi_centered[:, j], max_points=int(max_points))
                if x_ref is None:
                    x_ref = x_u
                ys.append(y_u)
            ax.plot(x_ref, np.stack(ys, axis=0).mean(axis=0), color="tab:blue", linewidth=2.0)
            ax.set_title(feat, fontsize=9)
            ax.grid(True, linewidth=0.3, alpha=0.5)
        for idx in range(len(selected), n_rows * n_cols):
            r = idx // n_cols
            c = idx % n_cols
            axes[r, c].axis("off")
        fig.suptitle("Shape stability (mean across folds; centered SHAP contributions)", y=1.01, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"shape_stability_top{len(selected)}_grid.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    return selected

