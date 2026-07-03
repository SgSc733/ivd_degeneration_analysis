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


def _plot_linear_shape(
    *,
    x_all: np.ndarray,
    beta: float,
    feature_name: str,
    out_path: Path,
    dpi: int,
) -> None:
    x_all = x_all.astype(float)
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    if x_min == x_max:
        return

    x_grid = np.linspace(x_min, x_max, 200)
    y_grid = beta * x_grid

    y_lo = float(np.percentile(y_grid, 1))
    y_hi = float(np.percentile(y_grid, 99))
    if y_hi == y_lo:
        y_hi = y_lo + 1.0
    margin = 0.15 * (y_hi - y_lo)
    y_min = y_lo - margin
    y_max = y_hi + margin

    plt.close()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_grid, y_grid, color="blue", linewidth=3)

    _density_blocks(ax, x_all, y_min=y_min, y_max=y_max)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(feature_name, fontsize="x-large")
    ax.set_ylabel("Linear contribution (beta * x)", fontsize="x-large")
    ax.set_title("shape function (linear)", fontsize=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def explain_shape_functions(
    *,
    X_raw: pd.DataFrame,
    fold_dir: Path,
    out_dir: Path,
    cfg: dict[str, Any],
) -> None:
    """
    Generate per-feature "shape function" plots for the sPLS-DA baseline.

    Note:
    - sPLS-DA is linear in the input space (after preprocessing).
    - Here we define the (additive) per-feature contribution on a chosen "severity axis":
        s = X * beta,  beta = R * a
      so each feature has a linear shape: f_j(x_j) = beta_j * x_j.
    """
    pre_path = fold_dir / "preprocessor.pkl"
    model_path = fold_dir / "model.pkl"
    feat_path = fold_dir / "feature_names.json"
    raw_input_path = fold_dir / "raw_input_features.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_names.json not found: {feat_path}")

    dpi = int(cfg.get("explain", {}).get("dpi", 300))
    only_selected = bool(cfg.get("explain", {}).get("only_selected", True))

    feat_meta = json.loads(feat_path.read_text(encoding="utf-8"))
    feature_names = feat_meta["feature_names"]
    if raw_input_path.exists():
        X_raw = pd.read_pickle(raw_input_path).reindex(columns=feature_names)
    else:
        X_raw = X_raw.reindex(columns=feature_names)

    # Fold-inner NaN imputation must match training fold statistics.
    imp_path = fold_dir / "imputer_median.csv"
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path, encoding="utf-8")
        if "feature" in imp_df.columns and "median" in imp_df.columns:
            med = pd.Series(imp_df["median"].to_numpy(dtype=float), index=imp_df["feature"].astype(str))
            X_raw = X_raw.fillna(med)

    if pre_path.exists():
        with open(pre_path, "rb") as f:
            pre = pickle.load(f)
        X_std = pre.transform(X_raw)
    else:
        # No preprocessing: use raw numeric X.
        X_std = X_raw.to_numpy(dtype=np.float64, copy=True)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Compute beta for the "severity axis" score s = X * beta, where beta = R a.
    # With nested CV, the selected H can vary by fold, so infer H from the checkpointed model.
    n_components = int(np.asarray(model.R_, dtype=np.float64).shape[1])
    axis_a = (cfg.get("postprocess", {}) or {}).get("axis_a", None)
    if axis_a is None:
        a = np.zeros((n_components,), dtype=np.float64)
        a[0] = 1.0
    else:
        a_in = np.asarray(axis_a, dtype=np.float64).reshape(-1)
        if a_in.size >= n_components:
            a = a_in[:n_components].copy()
        else:
            a = np.zeros((n_components,), dtype=np.float64)
            a[: a_in.size] = a_in
        if float(np.linalg.norm(a)) == 0.0:
            a[0] = 1.0

    beta = (np.asarray(model.R_, dtype=np.float64) @ a.reshape(-1, 1)).reshape(-1)

    # Decide which features to plot.
    mask_selected = (np.abs(np.asarray(model.W_, dtype=np.float64)) > 0).any(axis=1)
    feature_names_to_plot = feature_names
    if only_selected:
        feature_names_to_plot = [f for f, m in zip(feature_names, mask_selected.tolist()) if m]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "beta_axis.csv").write_text(
        pd.DataFrame({"feature": feature_names, "beta": beta}).to_csv(index=False),
        encoding="utf-8",
    )

    shape_dir = out_dir / "shape_functions"
    shape_dir.mkdir(parents=True, exist_ok=True)

    X_std = np.asarray(X_std, dtype=np.float64)
    for j, feat in enumerate(feature_names):
        if feat not in feature_names_to_plot:
            continue
        x_all = X_std[:, j]
        _plot_linear_shape(
            x_all=x_all,
            beta=float(beta[j]),
            feature_name=feat,
            out_path=shape_dir / f"{feat}.png",
            dpi=dpi,
        )

    # A quick global bar plot to help screen important features.
    top_k = int(cfg.get("explain", {}).get("top_k", 20))
    order = np.argsort(np.abs(beta))[::-1][: max(1, top_k)]
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in order][::-1], beta[order][::-1])
    ax.set_title("Top |beta| (severity axis)")
    ax.set_xlabel("beta")
    fig.tight_layout()
    fig.savefig(out_dir / "beta_top.png", dpi=dpi)
    plt.close(fig)
