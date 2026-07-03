from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumbar_ebm.figures_explain_cases import save_case_contribution_plots
from lumbar_ebm.figures_explain_collect import collect_fold_contributions
from lumbar_ebm.figures_explain_importance import compute_feature_importance, save_feature_importance_overview
from lumbar_ebm.figures_explain_stability import save_explain_stability_plots


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
    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    _ensure_dir(fig_dir)

    schema: dict[str, Any] = {}
    schema_path = run_dir / "data_schema.json"
    if schema_path.exists():
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception:
            schema = {}

    feature_names_num = list(schema.get("feature_names_num") or list(X_raw.columns))
    disc_level_feature_name = str(
        schema.get("disc_level_feature_name")
        or (cfg.get("ebm", {}) or {}).get("feature_disc_level_name")
        or "disc_level"
    )

    # For EBM, fold contribution collection needs the full X_ebm table (numeric + disc_level categorical).
    if meta is not None and disc_level_feature_name in meta.columns:
        X_ebm = X_raw.copy()
        X_ebm[disc_level_feature_name] = meta[disc_level_feature_name].astype(str).to_numpy()
    else:
        # Fallback: load disc_level from predictions table (generated during training).
        oof_path = run_dir / "oof_predictions_calibrated.csv"
        if not oof_path.exists():
            raise FileNotFoundError(f"Missing {oof_path}; cannot build X_ebm for explain figures.")
        oof = pd.read_csv(oof_path, encoding="utf-8")
        if disc_level_feature_name not in oof.columns:
            raise ValueError(f"Missing '{disc_level_feature_name}' in {oof_path}; cannot build X_ebm.")
        X_ebm = X_raw.copy()
        X_ebm[disc_level_feature_name] = oof[disc_level_feature_name].astype(str).to_numpy()

    folds = collect_fold_contributions(
        run_dir=run_dir,
        X_ebm=X_ebm,
        feature_names_num=feature_names_num,
        task_names=task_names,
        disc_level_feature_name=disc_level_feature_name,
        cfg=cfg,
    )

    importance = compute_feature_importance(folds=folds)
    importance.to_csv(fig_dir / "feature_importance_task_weighted.csv", index=False, encoding="utf-8")

    fig_cfg = cfg.get("figures", {}) or {}
    top_n = int(fig_cfg.get("explain_top_n", 20))
    max_points = int(fig_cfg.get("explain_max_points", 200))
    case_top_k = int(fig_cfg.get("case_top_k", 15))

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
    save_case_contribution_plots(
        run_dir=run_dir,
        folds=folds,
        tasks=tasks,
        out_dir=fig_dir / "case_explanations",
        top_k_features=case_top_k,
    )

    return {
        "n_folds": int(len(folds)),
        "n_features": int(len(folds[0].feature_names)),
        "top_features_selected": list(selected),
    }

