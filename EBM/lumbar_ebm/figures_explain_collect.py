from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumbar_ebm.explain import build_g_contrib_df, local_contributions_df
from lumbar_ebm.figures_explain_types import FoldContrib


def collect_fold_contributions(
    *,
    run_dir: str | Path,
    X_ebm: pd.DataFrame,
    feature_names_num: list[str],
    task_names: list[str],
    disc_level_feature_name: str,
    cfg: dict[str, Any],
) -> list[FoldContrib]:
    """Compute per-fold per-feature additive contributions (shared schema) for the full dataset."""
    run_dir = Path(run_dir)
    fold_dirs = sorted([p for p in (run_dir / "checkpoints").glob("fold_*") if p.is_dir()])
    out: list[FoldContrib] = []

    # If config explicitly asked for disc_level interactions, validate that folds contain them.
    inter_cfg = (cfg.get("ebm", {}) or {}).get("interactions", 0)
    requested_disc_inter = False
    if isinstance(inter_cfg, (list, tuple)):
        for it in inter_cfg:
            if isinstance(it, (list, tuple)):
                if any(str(x) == disc_level_feature_name for x in it):
                    requested_disc_inter = True
                    break

    for fold_dir in fold_dirs:
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except Exception:
            continue

        ckpt_path = fold_dir / "ebm.pkl"
        if not ckpt_path.exists():
            continue

        raw_input_path = fold_dir / "raw_input_features.pkl"
        if raw_input_path.exists():
            X_ebm_fold = pd.read_pickle(raw_input_path)
        else:
            X_ebm_fold = X_ebm.copy()

        with open(ckpt_path, "rb") as f:
            ebm = pickle.load(f)

        contrib_all = local_contributions_df(ebm, X_ebm=X_ebm_fold, include_intercept=False)
        if disc_level_feature_name not in contrib_all.columns:
            raise ValueError(
                f"Expected disc_level term '{disc_level_feature_name}' in local contribution columns, but missing. "
                "This prevents additive decomposition with segment effects."
            )

        # Validate disc_level interactions coverage when requested.
        if requested_disc_inter:
            feat_names_in = list(getattr(ebm, "feature_names_in_", []))
            if disc_level_feature_name not in feat_names_in:
                raise ValueError(
                    f"disc_level_feature_name '{disc_level_feature_name}' not found in fitted EBM feature_names_in_."
                )
            disc_idx = feat_names_in.index(disc_level_feature_name)
            term_features = list(getattr(ebm, "term_features_", []))
            term_names = list(getattr(ebm, "term_names_", []))
            disc_inter_terms = [
                str(tn)
                for feats, tn in zip(term_features, term_names)
                if isinstance(feats, tuple) and len(feats) == 2 and disc_idx in feats
            ]
            missing_disc_terms = [t for t in disc_inter_terms if t not in contrib_all.columns]
            if missing_disc_terms:
                raise ValueError(
                    "Detected disc_level interaction terms in the fitted EBM, but they are missing from explain_local() "
                    "contribution columns. This would make g_{ell,j} silently ignore interaction contributions.\n"
                    f"- missing terms: {missing_disc_terms}\n"
                    f"- available local contribution columns (first 20): {list(contrib_all.columns)[:20]}"
                )

        fold_numeric_names = [c for c in list(X_ebm_fold.columns) if c != disc_level_feature_name]
        g_contrib, _ = build_g_contrib_df(
            ebm,
            contrib_df=contrib_all,
            feature_names_num=fold_numeric_names,
            disc_level_feature_name=disc_level_feature_name,
        )

        # Include disc_level main-effect contribution as an extra "feature" so additive decomposition matches s_raw.
        feat_names = list(fold_numeric_names) + [disc_level_feature_name]
        f_raw = np.concatenate(
            [g_contrib.to_numpy(dtype=float), contrib_all[[disc_level_feature_name]].to_numpy(dtype=float)],
            axis=1,
        )
        f_mean = f_raw.mean(axis=0, keepdims=False)
        f_centered = f_raw - f_mean[None, :]

        intercept = float(getattr(ebm, "intercept_", 0.0))

        out.append(
            FoldContrib(
                fold=int(fold),
                feature_names=feat_names,
                task_names=list(task_names),
                X_raw=X_ebm_fold.drop(columns=[disc_level_feature_name], errors="ignore"),
                f_raw=f_raw,
                f_mean=f_mean,
                f_centered=f_centered,
                intercept=intercept,
            )
        )

    if not out:
        raise RuntimeError(f"No fold contributions computed under: {run_dir / 'checkpoints'}")
    return sorted(out, key=lambda x: x.fold)
