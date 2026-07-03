from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as patches  # noqa: E402


def _density_blocks(ax, x_all: np.ndarray, *, y_min: float, y_max: float, n_blocks: int = 20) -> None:
    x_all = x_all.astype(float)
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    if x_max == x_min:
        return

    x_n_blocks = int(min(n_blocks, len(np.unique(x_all))))
    hist, _ = np.histogram(x_all, bins=x_n_blocks)
    hist = hist.astype(float)
    if hist.max() > 0:
        hist = hist / hist.max()

    length = (x_max - x_min) / x_n_blocks
    for j in range(x_n_blocks):
        x_start = x_min + length * j
        x_end = x_start + length
        alpha = float(min(1.0, 0.01 + hist[j]))
        rect = patches.Rectangle(
            (x_start, y_min),
            x_end - x_start,
            y_max - y_min,
            linewidth=0.01,
            edgecolor=[0.9, 0.5, 0.5],
            facecolor=[0.9, 0.5, 0.5],
            alpha=alpha,
        )
        ax.add_patch(rect)


def _plot_curve(
    *,
    x_unique: np.ndarray,
    y_unique: np.ndarray,
    x_all: np.ndarray,
    title: str | None,
    xlabel: str,
    out_path: Path,
    dpi: int,
) -> None:
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_unique.astype(float), y_unique.astype(float), color="blue", linewidth=3)

    y_lo = float(np.percentile(y_unique, 1))
    y_hi = float(np.percentile(y_unique, 99))
    if y_hi == y_lo:
        y_hi = y_lo + 1.0
    margin = 0.15 * (y_hi - y_lo)
    y_min = y_lo - margin
    y_max = y_hi + margin

    _density_blocks(ax, x_all, y_min=y_min, y_max=y_max)
    ax.set_xlim(float(np.min(x_all)), float(np.max(x_all)))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(xlabel, fontsize="x-large")
    ax.set_ylabel("Centered contribution (delta y_cont)", fontsize="x-large")
    if title:
        ax.set_title(title, fontsize=18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def explain_features(
    *,
    X_raw: pd.DataFrame,
    segment_col: str,
    segment_levels: list[str],
    segment_reference: str,
    feature_names_to_plot: list[str],
    fold_dir: Path,
    out_dir: Path,
    cfg: dict[str, Any],
) -> None:
    """
    Generate additive "shape function" style plots for Elastic Net:
    - Global (reference segment) curves
    - Segment-specific curves (if segment interactions are enabled in the trained model)

    The y-axis is the *centered* contribution on y_cont:
        y(x) - y(x_median)
    where only one feature is varied and other features are fixed to medians.
    """
    dpi = int(cfg.get("explain", {}).get("dpi", 150))
    n_grid = int(cfg.get("explain", {}).get("n_grid", 200))
    max_feats = (cfg.get("explain", {}) or {}).get("feature_max_plots")
    if max_feats is not None:
        feature_names_to_plot = feature_names_to_plot[: int(max_feats)]

    model_path = fold_dir / "best_model.pkl"
    feat_path = fold_dir / "feature_names.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_names.json not found: {feat_path}")

    with open(model_path, "rb") as f:
        est = pickle.load(f)

    feat_meta = json.loads(feat_path.read_text(encoding="utf-8"))
    # Backward-compatible: old version used "int_" prefix; current version uses "delta_" prefix.
    include_interactions = any(
        str(n).startswith(("int_", "delta_")) for n in feat_meta.get("design_feature_names", [])
    )

    base_df = X_raw.copy()
    if segment_col not in base_df.columns:
        raise ValueError(f"X_raw must contain segment_col '{segment_col}'.")

    base_features = [c for c in base_df.columns if c != segment_col]
    med = base_df[base_features].median(numeric_only=True)
    baseline_row = {**{c: float(med[c]) for c in med.index}, segment_col: segment_reference}
    baseline_df = pd.DataFrame([baseline_row])
    baseline_y = float(est.predict(baseline_df)[0])

    shared_dir = out_dir / "shared_reference"
    task_dir = out_dir / "by_segment"
    shared_dir.mkdir(parents=True, exist_ok=True)
    task_dir.mkdir(parents=True, exist_ok=True)

    for feat in feature_names_to_plot:
        x_all_raw = base_df[feat].to_numpy(dtype=float)
        if x_all_raw.size == 0:
            continue

        # Explanation uses raw features; handle missing/inf robustly to avoid plotting crashes.
        x_all = x_all_raw[np.isfinite(x_all_raw)]
        if x_all.size == 0:
            continue

        # Use quantiles to control plot density.
        x_unique = np.unique(x_all.astype(float))
        if x_unique.size > n_grid:
            qs = np.linspace(0.0, 1.0, n_grid)
            grid = np.quantile(x_all, qs)
            x_grid = np.unique(grid.astype(float))
        else:
            x_grid = x_unique

        # Reference segment curve.
        rows = []
        for xv in x_grid.tolist():
            row = dict(baseline_row)
            row[feat] = float(xv)
            rows.append(row)
        df_grid = pd.DataFrame(rows)
        y_grid = est.predict(df_grid).astype(float)
        y_center = y_grid - baseline_y
        order = np.argsort(x_grid.astype(float))

        _plot_curve(
            x_unique=x_grid[order],
            y_unique=y_center[order],
            x_all=x_all,
            title=None,
            xlabel=feat,
            out_path=shared_dir / f"{feat}.png",
            dpi=dpi,
        )

        if not include_interactions:
            continue

        seg_all = base_df[segment_col].astype(str).to_numpy()
        for seg in segment_levels:
            mask = seg_all == seg
            x_seg_raw = x_all_raw[mask]
            x_seg = x_seg_raw[np.isfinite(x_seg_raw)]
            if x_seg.size == 0:
                continue

            base_seg = dict(baseline_row)
            base_seg[segment_col] = seg
            baseline_seg_y = float(est.predict(pd.DataFrame([base_seg]))[0])

            rows = []
            for xv in x_grid.tolist():
                row = dict(base_seg)
                row[feat] = float(xv)
                rows.append(row)
            y_grid = est.predict(pd.DataFrame(rows)).astype(float)
            y_center = y_grid - baseline_seg_y
            order = np.argsort(x_grid.astype(float))

            _plot_curve(
                x_unique=x_grid[order],
                y_unique=y_center[order],
                x_all=x_seg,
                title=str(seg),
                xlabel=feat,
                out_path=task_dir / seg / f"{feat}.png",
                dpi=dpi,
            )
