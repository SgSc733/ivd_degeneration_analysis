from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumbar_enet.figures_explain_types import FoldContrib


def collect_fold_contributions(
    *,
    run_dir: str | Path,
    X_raw: pd.DataFrame,
    meta: pd.DataFrame | None,
    task_names: list[str],
    cfg: dict[str, Any],
) -> list[FoldContrib]:
    """
    Compute per-fold additive contributions for the full dataset.

    For Elastic Net, the model is linear in the fold-inner preprocessed space:
        y_cont = b_task + sum_j w_task_j * z_j
    where z_j is the RankGauss+ZScore transformed feature.

    We align the outputs with ProtoNAM's figure pipeline:
    - f_raw: z (shared per-feature "shape", before task weighting)
    - w_last/b_last: task-specific coefficients and intercepts
    """
    _ = meta, cfg  # kept for signature parity with ProtoNAM
    run_dir = Path(run_dir)

    fold_dirs = sorted([p for p in (run_dir / "checkpoints").glob("fold_*") if p.is_dir()])
    out: list[FoldContrib] = []

    for fold_dir in fold_dirs:
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except Exception:
            continue

        model_path = fold_dir / "best_model.pkl"
        raw_input_path = fold_dir / "raw_input_features.pkl"
        if not model_path.exists():
            continue

        with open(model_path, "rb") as f:
            est = pickle.load(f)

        if not hasattr(est, "pre") or not hasattr(est, "coef_") or not hasattr(est, "intercept_"):
            raise TypeError(f"Unexpected model object in {model_path}; expected FittedElasticNet-like fields.")

        pre = est.pre
        coef = np.asarray(est.coef_, dtype=np.float64).reshape(-1)
        intercept = float(est.intercept_)

        if raw_input_path.exists():
            X_raw_fold = pd.read_pickle(raw_input_path)
        else:
            X_raw_fold = X_raw.copy()

        # Full design for all samples (fold-inner preprocessing).
        X_design = pre.transform(X_raw_fold).astype(np.float64, copy=False)
        slices = pre.get_block_slices()
        f_raw = np.asarray(X_design[:, slices.base], dtype=np.float64)

        feature_names = list(getattr(pre, "base_feature_names_", []) or [])
        if not feature_names:
            seg_col = str(getattr(pre, "segment_col", "disc_level"))
            feature_names = [c for c in list(X_raw.columns) if c != seg_col]

        if f_raw.shape[1] != len(feature_names):
            raise ValueError(
                f"Unexpected base design shape in {fold_dir}: f_raw.shape={f_raw.shape}, n_features={len(feature_names)}"
            )

        seg_levels = list(getattr(pre, "segment_levels", []) or [])
        seg_ref = str(getattr(pre, "segment_reference", "L3-L4"))
        seg_nonref = [lvl for lvl in seg_levels if lvl != seg_ref]

        name_to_task = {str(n): i for i, n in enumerate(list(task_names))}
        n_tasks = len(task_names)

        w_last = np.tile(coef[slices.base][None, :], (n_tasks, 1)).astype(np.float64, copy=False)
        b_last = np.full((n_tasks,), intercept, dtype=np.float64)

        # Segment intercept offsets (gamma): applied to b_last for non-reference segments.
        if slices.seg_intercept is not None:
            gamma = np.asarray(coef[slices.seg_intercept], dtype=np.float64).reshape(-1)
            if gamma.size == len(seg_nonref):
                for i, lvl in enumerate(seg_nonref):
                    ti = name_to_task.get(str(lvl))
                    if ti is not None:
                        b_last[int(ti)] += float(gamma[i])

        # Segment-feature interactions (delta): applied to w_last for non-reference segments.
        if slices.interactions is not None and seg_nonref:
            delta_flat = np.asarray(coef[slices.interactions], dtype=np.float64).reshape(-1)
            expect = int(len(seg_nonref) * len(feature_names))
            if delta_flat.size == expect:
                delta = delta_flat.reshape(len(seg_nonref), len(feature_names))
                for i, lvl in enumerate(seg_nonref):
                    ti = name_to_task.get(str(lvl))
                    if ti is not None:
                        w_last[int(ti), :] += delta[int(i), :]

        f_mean = f_raw.mean(axis=0, keepdims=False)
        f_centered = f_raw - f_mean[None, :]

        out.append(
            FoldContrib(
                fold=int(fold),
                feature_names=feature_names,
                task_names=list(task_names),
                X_raw=X_raw_fold,
                f_raw=f_raw.astype(float),
                f_mean=f_mean.astype(float),
                f_centered=f_centered.astype(float),
                w_last=w_last.astype(float),
                b_last=b_last.astype(float),
            )
        )

    if not out:
        raise RuntimeError(f"No fold contributions computed under: {run_dir / 'checkpoints'}")
    return sorted(out, key=lambda x: x.fold)
