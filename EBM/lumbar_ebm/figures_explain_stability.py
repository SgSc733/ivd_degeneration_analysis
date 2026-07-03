from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lumbar_ebm.figures_explain_types import FoldContrib


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
    folds: list[FoldContrib],
    X_raw: pd.DataFrame,
    importance: pd.DataFrame,
    out_dir: Path,
    top_n: int = 20,
    max_points: int = 200,
) -> list[str]:
    """Shape stability: fold curves + mean±SD band for top features."""
    _ensure_dir(out_dir)
    selected = list(importance["feature"].head(int(top_n)).tolist())
    for feat in selected:
        available = [
            fd
            for fd in folds
            if feat in fd.feature_names and fd.X_raw is not None and feat in fd.X_raw.columns
        ]
        if not available:
            continue
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for fd in available:
            j = fd.feature_names.index(feat)
            x = pd.to_numeric(fd.X_raw[feat], errors="coerce").to_numpy(dtype=float)
            x_u, y_u = _shape_curve_from_samples(x=x, y=fd.f_centered[:, j], max_points=int(max_points))
            if x_u.size == 0 or y_u.size == 0:
                continue
            xs.append(x_u)
            ys.append(y_u)
        if not xs:
            continue

        x_grid = np.unique(np.concatenate(xs, axis=0))
        if int(max_points) > 0 and x_grid.size > int(max_points):
            idx = np.linspace(0, int(x_grid.size) - 1, num=int(max_points)).round().astype(int)
            x_grid = x_grid[np.unique(idx)]
        y_interp = []
        for x_u, y_u in zip(xs, ys):
            interp = np.interp(x_grid, x_u, y_u, left=np.nan, right=np.nan)
            outside = (x_grid < x_u.min()) | (x_grid > x_u.max())
            interp[outside] = np.nan
            y_interp.append(interp)
        y_stack = np.vstack(y_interp)
        y_mean = np.nanmean(y_stack, axis=0)
        y_sd = np.nanstd(y_stack, axis=0, ddof=1) if y_stack.shape[0] >= 2 else np.zeros_like(y_mean)

        fig, ax = plt.subplots(figsize=(6, 5))
        for k, _fd in enumerate(available):
            ax.plot(xs[k], ys[k], color="tab:gray", alpha=0.25, linewidth=1.0, label=None if k else "Fold curves")
        ax.plot(x_grid, y_mean, color="tab:blue", linewidth=2.5, label="Mean")
        ax.fill_between(x_grid, y_mean - y_sd, y_mean + y_sd, color="tab:blue", alpha=0.18, label="±1 SD")
        ax.set_title(f"Shape stability: {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel("Centered contribution")
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
            available = [
                fd
                for fd in folds
                if feat in fd.feature_names and fd.X_raw is not None and feat in fd.X_raw.columns
            ]
            if not available:
                ax.axis("off")
                continue
            ys = []
            xs = []
            for fd in available:
                j = fd.feature_names.index(feat)
                x = pd.to_numeric(fd.X_raw[feat], errors="coerce").to_numpy(dtype=float)
                x_u, y_u = _shape_curve_from_samples(x=x, y=fd.f_centered[:, j], max_points=int(max_points))
                if x_u.size == 0 or y_u.size == 0:
                    continue
                xs.append(x_u)
                ys.append(y_u)
            if not xs:
                ax.axis("off")
                continue
            x_grid = np.unique(np.concatenate(xs, axis=0))
            if int(max_points) > 0 and x_grid.size > int(max_points):
                grid_idx = np.linspace(0, int(x_grid.size) - 1, num=int(max_points)).round().astype(int)
                x_grid = x_grid[np.unique(grid_idx)]
            interp_rows = []
            for x_u, y_u in zip(xs, ys):
                interp = np.interp(x_grid, x_u, y_u, left=np.nan, right=np.nan)
                outside = (x_grid < x_u.min()) | (x_grid > x_u.max())
                interp[outside] = np.nan
                interp_rows.append(interp)
            ax.plot(x_grid, np.nanmean(np.vstack(interp_rows), axis=0), color="tab:blue", linewidth=2.0)
            ax.set_title(feat, fontsize=9)
            ax.grid(True, linewidth=0.3, alpha=0.5)
        for idx in range(len(selected), n_rows * n_cols):
            r = idx // n_cols
            c = idx % n_cols
            axes[r, c].axis("off")
        fig.suptitle("Shape stability (mean across folds; centered contributions)", y=1.01, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"shape_stability_top{len(selected)}_grid.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    return selected
