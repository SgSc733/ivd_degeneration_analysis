from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumbar_xgb.figures_explain_types import FoldShapContrib


def _require_xgboost():
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # pragma: no cover
        msg = (
            "Missing dependency: xgboost.\n\n"
            "Install it into your conda env (recommended):\n"
            "  conda install -n pnam -c conda-forge xgboost\n\n"
            "Or via pip:\n"
            "  pip install xgboost\n\n"
            f"Original error: {type(e).__name__}: {e}"
        )
        raise RuntimeError(msg) from e
    return xgb


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_fold_shap_contributions(
    *,
    run_dir: str | Path,
    X_raw: pd.DataFrame,
) -> list[FoldShapContrib]:
    """Compute per-fold TreeSHAP contributions for the full dataset.

    This mirrors ProtoNAM's `collect_fold_contributions()` but uses XGBoost pred_contribs
    (TreeSHAP) on the latent score s(x).
    """
    xgb = _require_xgboost()

    run_dir = Path(run_dir)
    fold_dirs = sorted([p for p in (run_dir / "checkpoints").glob("fold_*") if p.is_dir()])
    out: list[FoldShapContrib] = []
    for fold_dir in fold_dirs:
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except Exception:
            continue

        feat_path = fold_dir / "feature_names.json"
        model_path = fold_dir / "xgb_model.json"
        info_path = fold_dir / "fold_info.json"
        pre_path = fold_dir / "preprocessor.pkl"
        if not feat_path.exists() or not model_path.exists() or not info_path.exists():
            continue

        feat_meta = _load_json(feat_path)
        feature_names = list(feat_meta.get("feature_names") or [])
        cont_feature_names = list(feat_meta.get("cont_feature_names") or [])
        segment_feature_names = list(feat_meta.get("segment_feature_names") or [])
        preprocess_enabled = bool(feat_meta.get("preprocess_enabled", False))
        if not feature_names:
            continue

        pre = None
        if preprocess_enabled and cont_feature_names:
            if not pre_path.exists():
                raise FileNotFoundError(f"preprocess_enabled=true but preprocessor.pkl missing: {pre_path}")
            with open(pre_path, "rb") as f:
                pre = pickle.load(f)

        # Build model-input matrix in the exact feature order used in training.
        # Fold-inner feature selection can create per-fold PCA/pooling columns that do not exist
        # in the original wide feature table. In that case, use the per-fold saved matrix.
        raw_input_path = fold_dir / "raw_input_features.pkl"
        X_source = pd.read_pickle(raw_input_path) if raw_input_path.exists() else X_raw
        X_cont = X_source[cont_feature_names] if cont_feature_names else pd.DataFrame(index=X_source.index)
        X_seg = X_source[segment_feature_names] if segment_feature_names else pd.DataFrame(index=X_source.index)

        if cont_feature_names:
            X_cont = X_cont.apply(pd.to_numeric, errors="coerce")
            if X_cont.isna().any().any():
                X_cont = X_cont.fillna(X_cont.median(numeric_only=True))
            if pre is not None:
                X_cont_np = np.asarray(pre.transform(X_cont), dtype=np.float32)
            else:
                X_cont_np = X_cont.to_numpy(dtype=np.float32, copy=True)
        else:
            X_cont_np = np.zeros((len(X_source), 0), dtype=np.float32)

        if segment_feature_names:
            X_seg = X_seg.apply(pd.to_numeric, errors="coerce")
            if X_seg.isna().any().any():
                X_seg = X_seg.fillna(0.0)
            X_seg_np = X_seg.to_numpy(dtype=np.float32, copy=True)
        else:
            X_seg_np = np.zeros((len(X_source), 0), dtype=np.float32)

        feature_values = np.concatenate([
            X_cont.to_numpy(dtype=float, copy=True) if cont_feature_names else np.zeros((len(X_source), 0), dtype=float),
            X_seg.to_numpy(dtype=float, copy=True) if segment_feature_names else np.zeros((len(X_source), 0), dtype=float),
        ], axis=1)
        X_model = np.concatenate([X_cont_np, X_seg_np], axis=1)
        if X_model.shape[1] != len(feature_names):
            raise ValueError(
                f"X_model shape mismatch in {fold_dir}: got {X_model.shape[1]} cols, "
                f"expected n_features={len(feature_names)}"
            )

        info = _load_json(info_path)
        best_iter = info.get("xgb_selected_iteration", info.get("xgb_best_iteration"))
        iter_range = None
        if best_iter is not None:
            iter_range = (0, int(best_iter) + 1)  # end is exclusive

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        dmat = xgb.DMatrix(X_model, feature_names=feature_names)

        contribs = booster.predict(dmat, pred_contribs=True, iteration_range=iter_range)
        if contribs.shape[1] != len(feature_names) + 1:
            raise ValueError(
                f"Unexpected pred_contribs shape for {fold_dir}: {contribs.shape} vs n_features+1={len(feature_names)+1}"
            )
        phi = contribs[:, :-1].astype(float)
        phi_mean = phi.mean(axis=0, keepdims=False).astype(float)
        phi_centered = (phi - phi_mean[None, :]).astype(float)

        out.append(
            FoldShapContrib(
                fold=int(fold),
                feature_names=feature_names,
                phi_raw=phi,
                phi_mean=phi_mean,
                phi_centered=phi_centered,
                feature_values=feature_values,
            )
        )

    if not out:
        raise RuntimeError(f"No fold SHAP contributions computed under: {run_dir / 'checkpoints'}")
    return sorted(out, key=lambda x: x.fold)

