from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumbar_splsda.figures_explain_types import FoldContrib


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_beta_axis(fold_dir: Path, *, feature_names: list[str]) -> np.ndarray | None:
    """
    Load beta for the "severity axis" from the fold checkpoint.

    This keeps the figure logic consistent with the training pipeline (which writes beta_axis.csv).
    """
    p = fold_dir / "beta_axis.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, encoding="utf-8")
    if "feature" not in df.columns or "beta" not in df.columns:
        return None
    s = pd.Series(df["beta"].to_numpy(dtype=float), index=df["feature"].astype(str))
    beta = s.reindex(feature_names).to_numpy(dtype=float)
    if beta.size != len(feature_names):
        return None
    return beta


def _apply_fold_imputer(*, X: pd.DataFrame, fold_dir: Path) -> pd.DataFrame:
    """
    Apply fold-inner NaN imputation (median) to match training statistics.

    The training pipeline writes `imputer_median.csv` per fold.
    """
    imp_path = fold_dir / "imputer_median.csv"
    if not imp_path.exists():
        return X
    try:
        imp_df = pd.read_csv(imp_path, encoding="utf-8")
    except Exception:
        return X
    if "feature" not in imp_df.columns or "median" not in imp_df.columns:
        return X
    med = pd.Series(imp_df["median"].to_numpy(dtype=float), index=imp_df["feature"].astype(str))
    return X.fillna(med)


def collect_fold_contributions(
    *,
    run_dir: str | Path,
    X_raw: pd.DataFrame,
    meta: pd.DataFrame | None,
    task_names: list[str],
    cfg: dict[str, Any],
) -> list[FoldContrib]:
    """
    Compute per-fold per-feature contributions (shared shape) for the full dataset.

    For sPLS-DA, we align with sPLSDA.md + explain.py:
      - Define a continuous "severity axis" s = X * beta (beta is the fold checkpointed beta_axis).
      - Define per-feature shape output as f_j(x_j) = beta_j * x_j (x after fold preprocessor).
      - No additional task weights -> use identity weights w_last=1, b_last=0 to reuse ProtoNAM-style figures.
    """
    _ = meta  # reserved for future meta-aware preprocessors

    run_dir = Path(run_dir)
    fold_dirs = sorted([p for p in (run_dir / "checkpoints").glob("fold_*") if p.is_dir()])
    out: list[FoldContrib] = []
    for fold_dir in fold_dirs:
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except Exception:
            continue

        pre_path = fold_dir / "preprocessor.pkl"
        model_path = fold_dir / "model.pkl"
        feat_path = fold_dir / "feature_names.json"
        raw_input_path = fold_dir / "raw_input_features.pkl"
        if not model_path.exists() or not feat_path.exists():
            continue

        feat_meta = _load_json(feat_path)
        feature_names = list(feat_meta.get("feature_names") or [])
        if not feature_names:
            continue

        if raw_input_path.exists():
            X_use = pd.read_pickle(raw_input_path).reindex(columns=feature_names)
        else:
            X_use = X_raw.reindex(columns=feature_names)
        X_use = _apply_fold_imputer(X=X_use, fold_dir=fold_dir)

        if pre_path.exists():
            with open(pre_path, "rb") as f:
                pre = pickle.load(f)
            X_std = pre.transform(X_use)  # type: ignore[no-untyped-call]
        else:
            X_std = X_use.to_numpy(dtype=np.float64, copy=True)

        beta = _load_beta_axis(fold_dir, feature_names=feature_names)
        if beta is None:
            # Fallback: try to re-derive beta from checkpointed model (if present).
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            R_mat = np.asarray(getattr(model, "R_", None), dtype=np.float64)
            if R_mat.ndim != 2 or R_mat.shape[0] != len(feature_names):
                raise ValueError(f"Unexpected R_ shape in {model_path}: {R_mat.shape} vs n_feat={len(feature_names)}")
            n_components = int(R_mat.shape[1])
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
            beta = (R_mat @ a.reshape(-1, 1)).reshape(-1)

        beta = np.asarray(beta, dtype=float).reshape(-1)
        if beta.size != len(feature_names):
            raise ValueError(f"Unexpected beta size for fold {fold}: {beta.size} vs n_feat={len(feature_names)}")

        X_std_f = np.asarray(X_std, dtype=np.float64)
        f_raw = X_std_f * beta.reshape(1, -1)
        f_mean = f_raw.mean(axis=0, keepdims=False)
        f_centered = f_raw - f_mean[None, :]

        # Identity weights: keep task dimension to reuse ProtoNAM-style plots consistently.
        n_tasks = int(len(task_names))
        w_last = np.ones((n_tasks, int(len(feature_names))), dtype=float)
        b_last = np.zeros((n_tasks,), dtype=float)

        out.append(
            FoldContrib(
                fold=int(fold),
                feature_names=list(feature_names),
                task_names=list(task_names),
                X_raw=X_use.copy(),
                f_raw=f_raw.astype(float),
                f_mean=f_mean.astype(float),
                f_centered=f_centered.astype(float),
                w_last=w_last,
                b_last=b_last,
            )
        )

    if not out:
        raise RuntimeError(f"No fold contributions computed under: {run_dir / 'checkpoints'}")

    # Ensure deterministic order.
    out = sorted(out, key=lambda x: x.fold)
    return out
