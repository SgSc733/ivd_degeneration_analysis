from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lumbar_xgb.figures_explain_types import FoldShapContrib


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def compute_feature_importance(
    *,
    folds: list[FoldShapContrib],
) -> pd.DataFrame:
    """Aggregate global importance with fold stability (mean±SD across folds).

    Importance proxy:
      - mean_abs_contrib = mean(|phi_centered|) over samples
      - var_contrib      = var(phi_centered) over samples
    """
    common_features = list(folds[0].feature_names)
    for fd in folds[1:]:
        fd_set = set(fd.feature_names)
        common_features = [name for name in common_features if name in fd_set]
    if not common_features:
        raise ValueError("No common feature across folds; cannot aggregate SHAP fold stability.")

    per_fold_rows: list[dict[str, Any]] = []
    for fd in folds:
        phi = np.asarray(fd.phi_centered, dtype=float)
        name_to_idx = {name: i for i, name in enumerate(fd.feature_names)}
        for name in common_features:
            j = int(name_to_idx[name])
            col = phi[:, j]
            per_fold_rows.append(
                {
                    "fold": int(fd.fold),
                    "feature": str(name),
                    "mean_abs_contrib": float(np.mean(np.abs(col))),
                    "var_contrib": float(np.var(col)),
                }
            )

    per_fold = pd.DataFrame(per_fold_rows)
    agg = (
        per_fold.groupby("feature", sort=False)[["mean_abs_contrib", "var_contrib"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [
        "feature",
        "mean_abs_mean",
        "mean_abs_std",
        "var_mean",
        "var_std",
    ]
    agg = agg.sort_values("mean_abs_mean", ascending=False).reset_index(drop=True)
    agg["rank"] = np.arange(1, int(agg.shape[0]) + 1)
    return agg


def save_feature_importance_overview(
    *,
    importance: pd.DataFrame,
    out_path: Path,
    top_n: int = 20,
    title: str = "Global feature importance (TreeSHAP phi; mean±SD across folds)",
) -> None:
    df = importance.head(int(top_n)).copy()
    if df.empty:
        return
    feats = df["feature"].tolist()[::-1]
    mean_abs = df["mean_abs_mean"].to_numpy(dtype=float)[::-1]
    mean_abs_sd = df["mean_abs_std"].to_numpy(dtype=float)[::-1]
    var = df["var_mean"].to_numpy(dtype=float)[::-1]
    var_sd = df["var_std"].to_numpy(dtype=float)[::-1]

    y = np.arange(len(feats), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.0, 0.28 * len(feats))), sharey=True)
    ax = axes[0]
    ax.barh(y, mean_abs, xerr=mean_abs_sd, color="tab:blue", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("Mean |phi| (across samples; mean±SD across folds)")
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.5)
    ax.set_title("Mean absolute contribution")

    ax2 = axes[1]
    ax2.barh(y, var, xerr=var_sd, color="tab:orange", alpha=0.85)
    ax2.set_xlabel("Contribution variance (across samples; mean±SD across folds)")
    ax2.grid(True, axis="x", linewidth=0.3, alpha=0.5)
    ax2.set_title("Contribution variance")

    fig.suptitle(title, y=1.02, fontsize=12)
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

