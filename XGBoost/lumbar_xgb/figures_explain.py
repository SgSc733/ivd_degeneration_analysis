from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumbar_xgb.figures_explain_collect import collect_fold_shap_contributions
from lumbar_xgb.figures_explain_importance import compute_feature_importance, save_feature_importance_overview
from lumbar_xgb.figures_explain_stability import save_explain_stability_plots


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def generate_explain_figures(
    *,
    run_dir: str | Path,
    X_raw: pd.DataFrame,
    meta: pd.DataFrame | None,
    tasks: np.ndarray,
    task_names: list[str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Generate explanation stability + global overview figures (fold-stability SHAP)."""
    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    _ensure_dir(fig_dir)

    folds = collect_fold_shap_contributions(run_dir=run_dir, X_raw=X_raw)

    importance = compute_feature_importance(folds=folds)
    importance.to_csv(fig_dir / "feature_importance_shap_fold_stability.csv", index=False, encoding="utf-8")

    fig_cfg = cfg.get("figures", {}) or {}
    top_n = int(fig_cfg.get("explain_top_n", 20))
    max_points = int(fig_cfg.get("explain_max_points", 200))

    save_feature_importance_overview(
        importance=importance,
        out_path=fig_dir / "feature_importance_overview.png",
        top_n=top_n,
    )
    selected = save_explain_stability_plots(
        folds=folds,
        X_raw=X_raw,
        importance=importance,
        out_dir=fig_dir / "explain_stability",
        top_n=top_n,
        max_points=max_points,
    )

    return {
        "n_folds": int(len(folds)),
        "n_features": int(len(folds[0].feature_names)),
        "top_features_selected": list(selected),
    }

