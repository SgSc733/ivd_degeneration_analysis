from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lumbar_enet.decision_thresholds import parse_threshold_matrix, predict_from_y_cont_by_task
from lumbar_enet.data import load_model_input_csv
from lumbar_enet.feature_selection import FeatureSelectionDataset, FoldFeatureSelector
from lumbar_enet.metrics import OrdinalMetrics, compute_ordinal_metrics
from lumbar_enet.modeling import FittedElasticNet
from lumbar_enet.preprocess import ElasticNetDesignMatrix
from lumbar_enet.solver import compute_lambda_max, fit_weighted_elastic_net_cd, fit_weighted_elastic_net_path


@dataclass(frozen=True)
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    metrics: OrdinalMetrics
    best_params: dict[str, Any]
    checkpoint_dir: Path


@dataclass(frozen=True)
class CVResult:
    fold_results: list[FoldResult]
    best_fold: int
    fold_mean_metrics: OrdinalMetrics
    oof_metrics: OrdinalMetrics


def _round_half_up(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.floor(x + 0.5)


def _round_clip(y_cont: np.ndarray, *, clip_min: int, clip_max: int) -> np.ndarray:
    y = _round_half_up(np.asarray(y_cont).astype(float)).astype(int)
    y = np.clip(y, clip_min, clip_max)
    return y


def _save_confusion_matrix(cm: np.ndarray, labels: list[int], out_path: Path, title: str) -> None:
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _lambda_1se_index(mean_score: np.ndarray, se: np.ndarray, lambdas_desc: np.ndarray) -> int:
    best_idx = int(np.argmin(mean_score))
    thr = float(mean_score[best_idx] + se[best_idx])
    ok = np.where(mean_score <= thr)[0]
    if ok.size == 0:
        return best_idx
    # Largest lambda => earliest index when descending.
    return int(np.min(ok))


def _inner_cv_scores_for_path(
    *,
    X_df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    segment_levels: list[str],
    segment_ref: str,
    segment_col: str,
    thr_by_task: np.ndarray,
    model_cfg: dict[str, Any],
    l1_ratio: float,
    lambdas: np.ndarray,
    inner_splits: int,
    clip_min: int,
    clip_max: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    gkf = GroupKFold(n_splits=inner_splits)

    scores = []
    conv_flags = []
    iters = []

    selection = str(model_cfg.get("selection", "cyclic"))
    if selection not in ("cyclic", "random"):
        selection = "cyclic"
    max_iter = int(model_cfg.get("max_iter", 10000))
    tol = float(model_cfg.get("tol", 1e-4))

    for tr_idx, va_idx in gkf.split(X_df, y, groups=groups):
        X_tr_df = X_df.iloc[tr_idx].reset_index(drop=True)
        y_tr = np.asarray(y)[tr_idx].astype(float)
        X_va_df = X_df.iloc[va_idx].reset_index(drop=True)
        y_va = np.asarray(y)[va_idx].astype(int)

        pre = ElasticNetDesignMatrix(
            segment_col=segment_col,
            segment_levels=segment_levels,
            segment_reference=segment_ref,
            include_segment_intercept=bool(model_cfg.get("include_segment_intercept", True)),
            include_segment_interactions=bool(model_cfg.get("include_segment_interactions", True)),
            segment_penalty_factor=float(model_cfg.get("segment_penalty_factor", 0.0)),
            interaction_penalty_factor=float(model_cfg.get("interaction_penalty_factor", 2.0)),
        ).fit(X_tr_df)
        X_tr = pre.transform(X_tr_df).astype(np.float64, copy=False)
        X_va = pre.transform(X_va_df).astype(np.float64, copy=False)
        v = pre.get_penalty_factors().astype(np.float64, copy=False)

        coefs, intercepts, n_iters, conv = fit_weighted_elastic_net_path(
            X=X_tr,
            y=y_tr,
            penalty_factors=v,
            l1_ratio=float(l1_ratio),
            lambdas=lambdas,
            fit_intercept=bool(model_cfg.get("fit_intercept", True)),
            selection=selection,  # type: ignore[arg-type]
            max_iter=max_iter,
            tol=tol,
            random_state=int(random_state),
        )

        y_cont_mat = X_va @ coefs.T + intercepts.reshape(1, -1)
        # Decode y_cont -> y_pred using the same threshold semantics as the outer evaluation.
        # This keeps the inner selection metric (MAE) aligned with the reported metrics.
        seg_va = X_va_df[segment_col].astype(str).to_numpy()
        level_to_task = {lvl: i for i, lvl in enumerate(list(segment_levels))}
        task_va = np.asarray([level_to_task.get(s, -1) for s in seg_va.tolist()], dtype=int)
        if (task_va < 0).any():
            bad = sorted({seg_va[i] for i in np.where(task_va < 0)[0][:10].tolist()})
            raise ValueError(f"Unknown segment levels in inner CV: {bad}. Expected one of: {segment_levels}")

        K = int(np.asarray(thr_by_task).shape[1] + 1)
        ks = np.arange(1, K, dtype=np.float64)  # (K-1,)
        thr_va = np.asarray(thr_by_task, dtype=np.float64)[task_va]  # (n_val, K-1)
        bounds = ks[None, :] + thr_va  # (n_val, K-1)
        y_pred_mat = 1 + (y_cont_mat[:, :, None] > bounds[:, None, :]).sum(axis=2).astype(int)
        y_pred_mat = np.clip(y_pred_mat, clip_min, clip_max)
        mae_path = np.mean(np.abs(y_va.reshape(-1, 1) - y_pred_mat), axis=0).astype(np.float64)
        scores.append(mae_path)

        conv_flags.append(conv.astype(int))
        iters.append(n_iters.astype(int))

    score_mat = np.stack(scores, axis=0)  # (k, n_lambdas)
    mean_mae = score_mat.mean(axis=0)
    if score_mat.shape[0] > 1:
        se_mae = score_mat.std(axis=0, ddof=1) / np.sqrt(score_mat.shape[0])
    else:
        se_mae = np.zeros_like(mean_mae)

    conv_mat = np.stack(conv_flags, axis=0)
    it_mat = np.stack(iters, axis=0)
    extra = {
        "inner_folds": int(score_mat.shape[0]),
        "converged_rate_mean": float(conv_mat.mean()),
        "max_iter_max": int(it_mat.max()),
    }
    return mean_mae, se_mae, extra


def train_group_kfold(
    *,
    X_raw: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    tasks: np.ndarray,
    task_names: list[str],
    run_dir: Path,
    cfg: dict[str, Any],
    meta: pd.DataFrame,
    id_col: str,
    feature_selection_dataset: FeatureSelectionDataset | None = None,
) -> CVResult:
    """
    Outer GroupKFold CV + inner GroupKFold lambda-path selection (lambda_min / lambda_1se).

    This implementation matches the plan-level notation in Elastic Net.md:
    - strict Rank->Phi^{-1} + Z-score preprocessing (fit on training fold only)
    - optional segment intercepts (gamma) and segment-feature interactions (delta)
    - per-column penalty factor v_j (gamma default v=0)
    - lambda path + 1SE rule
    """
    out_base = Path(run_dir) / "checkpoints"
    out_base.mkdir(parents=True, exist_ok=True)

    ord_cfg = cfg.get("ordinal", {}) or {}
    clip_min = int(ord_cfg.get("clip_min", 1))
    clip_max = int(ord_cfg.get("clip_max", 5))
    n_classes = int(ord_cfg.get("n_classes", int(clip_max - clip_min + 1)))
    thr_by_task_base, _thr5_placeholder = parse_threshold_matrix(
        ord_cfg=ord_cfg,
        task_names=list(task_names),
        n_classes=int(n_classes),
    )

    tr_cfg = cfg.get("training", {}) or {}
    outer_splits = int(tr_cfg.get("outer_splits", 5))
    inner_splits_cfg = int((cfg.get("search", {}) or {}).get("cv_inner_splits", 3))
    random_state = int(tr_cfg.get("random_state", 0))

    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    pooling_cfg = data_cfg.get("pooling", {}) or {}

    segment_levels = list((cfg.get("data", {}) or {}).get("segment_levels", task_names))
    segment_ref = str((cfg.get("data", {}) or {}).get("segment_reference", "L3-L4"))
    segment_col = str(model_cfg.get("segment_col", "disc_level"))

    search_cfg = cfg.get("search", {}) or {}
    l1_ratio_list = list(search_cfg.get("l1_ratio", [0.1, 0.3, 0.5, 0.7, 0.9]))
    n_lambdas = int(search_cfg.get("n_lambdas", 60))
    lambda_min_ratio = float(search_cfg.get("lambda_min_ratio", 1e-3))
    lambda_choice = str(search_cfg.get("lambda_choice", "lambda_1se"))
    if lambda_choice not in ("lambda_1se", "lambda_min"):
        raise ValueError("search.lambda_choice must be 'lambda_1se' or 'lambda_min'.")
    if n_lambdas < 2:
        raise ValueError("search.n_lambdas must be >= 2.")
    if not (0 < lambda_min_ratio < 1):
        raise ValueError("search.lambda_min_ratio must be in (0, 1).")

    outer = GroupKFold(n_splits=min(outer_splits, len(np.unique(groups))))
    fold_results: list[FoldResult] = []
    best_fold = 0
    best_kappa = -1e9

    oof_rows: list[dict[str, Any]] = []
    coef_rows_all: list[dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(outer.split(X_raw, y, groups=groups)):
        fold_dir = out_base / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fs_summary: dict[str, Any] | None = None
        if feature_selection_dataset is not None:
            fs_dir = fold_dir / "feature_selection"
            fs_dir.mkdir(parents=True, exist_ok=True)
            fs_cfg = cfg.get("feature_selection", {}) or {}
            fs_enabled = bool(fs_cfg.get("enabled", True))
            resolved_id_col = str(data_cfg.get("id_col", id_col))

            if fs_enabled:
                selector = FoldFeatureSelector.from_config(
                    fs_cfg,
                    segments=feature_selection_dataset.task_names,
                    id_col=resolved_id_col,
                )
                train_patient_ids = feature_selection_dataset.subset_patient_ids(train_idx)
                train_conditions = feature_selection_dataset.subset_conditions_by_patients(train_patient_ids)
                train_grade_long = feature_selection_dataset.subset_grade_long_by_patients(train_patient_ids)
                fs_result = selector.fit(conditions_raw=train_conditions, grade_long_df=train_grade_long)
                runtime_cfg = cfg.get("_runtime", {}) or {}
                config_dir = Path(str(runtime_cfg.get("config_dir", ".")))

                def _resolve_cfg_path(value: Any) -> str | None:
                    if value is None or not str(value).strip():
                        return None
                    path = Path(str(value))
                    if path.is_absolute():
                        return str(path)
                    return str((config_dir / path).resolve())

                selector.save_audit(
                    fs_dir,
                    pfirrmann_csv=_resolve_cfg_path((cfg.get("labels", {}) or {}).get("xlsx_path") or data_cfg.get("pfirrmann_csv_path")),
                    statistics_csv=_resolve_cfg_path(fs_cfg.get("statistics_csv_path")),
                )
                with open(fs_dir / "selector.pkl", "wb") as f:
                    pickle.dump(selector, f, protocol=pickle.HIGHEST_PROTOCOL)
                selected_full_df = selector.transform_gold(feature_selection_dataset.gold_long)
                fs_summary = {
                    "enabled": True,
                    "n_features_initial": int(fs_result.n_features_initial),
                    "n_features_after_cleaning": int(fs_result.n_features_after_cleaning),
                    "n_robust_features": int(len(fs_result.robust_features)),
                    "n_spearman_features": int(len(fs_result.selected_by_spearman)),
                    "n_final_features": int(len(fs_result.final_features)),
                    "audit_dir": str(fs_dir),
                }
            else:
                gold_long = feature_selection_dataset.gold_long
                case_id_series = gold_long.index.get_level_values(0).astype(str)
                segment_series = gold_long.index.get_level_values(1).astype(str)
                sample_id = pd.Series(
                    case_id_series + "_" + segment_series,
                    index=gold_long.index,
                    name=resolved_id_col,
                )
                selected_full_df = pd.concat([sample_id, gold_long], axis=1).reset_index(drop=True)
                fs_summary = {
                    "enabled": False,
                    "n_final_features": int(gold_long.shape[1]),
                    "audit_dir": str(fs_dir),
                }

            selected_full_df.insert(1, "__pfirrmann__", feature_selection_dataset.y.astype(int))
            selected_full_path = fs_dir / "fold_model_input_full.csv"
            selected_full_df.to_csv(selected_full_path, index=False, encoding="utf-8")

            pooling_feature_types = pooling_cfg.get("feature_types", None)
            if isinstance(pooling_feature_types, str):
                pooling_feature_types = [pooling_feature_types]
            elif pooling_feature_types is not None:
                pooling_feature_types = list(pooling_feature_types)

            loaded_fold = load_model_input_csv(
                selected_full_path,
                id_col=resolved_id_col,
                label_col="__pfirrmann__",
                drop_cols=data_cfg.get("drop_cols", []),
                keep_feature_patterns=data_cfg.get("keep_feature_patterns", []),
                encoding=str(data_cfg.get("encoding", "utf-8")),
                classic_prefix=str(data_cfg.get("classic_prefix", "classic_")),
                patient_id_from_id_col=bool(data_cfg.get("patient_id_from_id_col", True)),
                patient_id_sep=str(data_cfg.get("patient_id_sep", "_")),
                level_from_id_col=bool(data_cfg.get("level_from_id_col", True)),
                level_sep=str(data_cfg.get("level_sep", "_")),
                segment_levels=list(data_cfg.get("segment_levels", feature_selection_dataset.task_names)),
                pooling_stats=list(pooling_cfg.get("stats", []) or []),
                pooling_feature_types=pooling_feature_types,
                pooling_pyr_prefix=str(pooling_cfg.get("pyr_prefix", "PyRadiomics_")),
                pooling_deep_prefix=str(pooling_cfg.get("deep_prefix", "DeepPCA_")),
                pooling_tensor_prefix=str(pooling_cfg.get("tensor_prefix", "TensorPCA_")),
                pooling_out_prefix=str(pooling_cfg.get("out_prefix", "pool_")),
            )
            X_raw_fold = loaded_fold.X.copy()
            X_raw_fold[segment_col] = loaded_fold.meta["disc_level"].astype(str)
            X_raw_fold.to_pickle(fold_dir / "raw_input_features.pkl")
            X_raw_use = X_raw_fold
            y_use = loaded_fold.y
            tasks_use = loaded_fold.tasks
            groups_use = loaded_fold.groups
            meta_use = loaded_fold.meta
            task_names_use = loaded_fold.task_names
        else:
            X_raw_use = X_raw
            y_use = y
            tasks_use = tasks
            groups_use = groups
            meta_use = meta
            task_names_use = task_names

        X_tr_df = X_raw_use.iloc[train_idx].reset_index(drop=True)
        y_tr = np.asarray(y_use)[train_idx].astype(float)
        g_tr = np.asarray(groups_use)[train_idx]

        X_va_df = X_raw_use.iloc[val_idx].reset_index(drop=True)
        y_va = np.asarray(y_use)[val_idx].astype(int)

        n_groups_tr = int(len(np.unique(g_tr)))
        inner_splits = int(min(inner_splits_cfg, n_groups_tr))
        inner_splits = max(2, inner_splits)

        # Reference preprocessor for lambda scale.
        pre_ref = ElasticNetDesignMatrix(
            segment_col=segment_col,
            segment_levels=segment_levels,
            segment_reference=segment_ref,
            include_segment_intercept=bool(model_cfg.get("include_segment_intercept", True)),
            include_segment_interactions=bool(model_cfg.get("include_segment_interactions", True)),
            segment_penalty_factor=float(model_cfg.get("segment_penalty_factor", 0.0)),
            interaction_penalty_factor=float(model_cfg.get("interaction_penalty_factor", 2.0)),
        ).fit(X_tr_df)
        X_tr_ref = pre_ref.transform(X_tr_df).astype(np.float64, copy=False)
        v_ref = pre_ref.get_penalty_factors().astype(np.float64, copy=False)

        curve_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        best_sel: dict[str, Any] | None = None
        for l1 in l1_ratio_list:
            l1 = float(l1)
            lam_max = compute_lambda_max(
                X=X_tr_ref,
                y=y_tr,
                penalty_factors=v_ref,
                l1_ratio=l1,
                fit_intercept=bool(model_cfg.get("fit_intercept", True)),
            )
            if lam_max <= 0:
                lam_max = 1e-6
            lam_min = lam_max * lambda_min_ratio
            lambdas = np.geomspace(lam_max, lam_min, num=n_lambdas).astype(np.float64)

            mean_mae, se_mae, extra = _inner_cv_scores_for_path(
                X_df=X_tr_df,
                y=y_tr,
                groups=g_tr,
                segment_levels=segment_levels,
                segment_ref=segment_ref,
                segment_col=segment_col,
                thr_by_task=thr_by_task_base,
                model_cfg=model_cfg,
                l1_ratio=l1,
                lambdas=lambdas,
                inner_splits=inner_splits,
                clip_min=clip_min,
                clip_max=clip_max,
                random_state=random_state,
            )

            idx_min = int(np.argmin(mean_mae))
            idx_1se = _lambda_1se_index(mean_mae, se_mae, lambdas_desc=lambdas)

            lam_min_sel = float(lambdas[idx_min])
            lam_1se_sel = float(lambdas[idx_1se])
            mae_min = float(mean_mae[idx_min])
            mae_1se = float(mean_mae[idx_1se])
            se_best = float(se_mae[idx_min])

            lam_selected = lam_1se_sel if lambda_choice == "lambda_1se" else lam_min_sel
            mae_selected = mae_1se if lambda_choice == "lambda_1se" else mae_min

            summary_rows.append(
                {
                    "l1_ratio": l1,
                    "lambda_max": lam_max,
                    "lambda_min_ratio": lambda_min_ratio,
                    "lambda_min": lam_min_sel,
                    "lambda_1se": lam_1se_sel,
                    "mae_min": mae_min,
                    "mae_1se": mae_1se,
                    "se_at_best": se_best,
                    "lambda_choice": lambda_choice,
                    "lambda_selected": lam_selected,
                    "mae_selected": mae_selected,
                    "converged_rate_mean": extra.get("converged_rate_mean"),
                    "max_iter_max": extra.get("max_iter_max"),
                }
            )
            for lam, m, s in zip(lambdas.tolist(), mean_mae.tolist(), se_mae.tolist(), strict=False):
                curve_rows.append(
                    {
                        "l1_ratio": l1,
                        "lambda": float(lam),
                        "mean_mae": float(m),
                        "se_mae": float(s),
                    }
                )

            cand = {"l1_ratio": l1, "lambda_selected": lam_selected, "mae_selected": mae_selected}
            if best_sel is None or cand["mae_selected"] < best_sel["mae_selected"] - 1e-12:
                best_sel = cand
            elif best_sel is not None and abs(cand["mae_selected"] - best_sel["mae_selected"]) <= 1e-12:
                # Tie-break: prefer larger lambda (more regularization).
                if cand["lambda_selected"] > best_sel["lambda_selected"]:
                    best_sel = cand

        if best_sel is None:
            raise RuntimeError("Inner CV selection failed (no candidates).")

        pd.DataFrame(summary_rows).to_csv(fold_dir / "inner_cv_summary.csv", index=False, encoding="utf-8")
        pd.DataFrame(curve_rows).to_csv(fold_dir / "inner_cv_curve.csv", index=False, encoding="utf-8")

        best_l1 = float(best_sel["l1_ratio"])
        best_lambda = float(best_sel["lambda_selected"])

        # Final fit on outer-train (pre_ref is fitted on outer-train only).
        selection = str(model_cfg.get("selection", "cyclic"))
        if selection not in ("cyclic", "random"):
            selection = "cyclic"
        max_iter = int(model_cfg.get("max_iter", 10000))
        tol = float(model_cfg.get("tol", 1e-4))

        cd_res = fit_weighted_elastic_net_cd(
            X=X_tr_ref,
            y=y_tr,
            penalty_factors=v_ref,
            l1_ratio=best_l1,
            lambda_=best_lambda,
            fit_intercept=bool(model_cfg.get("fit_intercept", True)),
            selection=selection,  # type: ignore[arg-type]
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

        fitted = FittedElasticNet(
            pre=pre_ref,
            coef_=cd_res.coef.astype(np.float64),
            intercept_=float(cd_res.intercept),
            l1_ratio=best_l1,
            lambda_=best_lambda,
        )

        X_va_design = pre_ref.transform(X_va_df).astype(np.float64, copy=False)
        y_cont = fitted.predict_design(X_va_design)
        y_pred = predict_from_y_cont_by_task(y_cont=y_cont, task=tasks_use[val_idx], thr_by_task=thr_by_task_base)
        metrics = compute_ordinal_metrics(y_true=y_va, y_pred=y_pred, y_cont=y_cont, n_classes=int(n_classes))

        # Save fold predictions.
        meta_va = meta_use.iloc[val_idx].reset_index(drop=True)
        pred_df = pd.DataFrame(
            {
                "orig_index": val_idx.astype(int),
                id_col: meta_va[id_col].astype(str),
                "patient_id": meta_va["patient_id"].astype(str),
                "disc_level": meta_va["disc_level"].astype(str),
                "task": tasks_use[val_idx].astype(int),
                "task_name": [task_names_use[t] for t in tasks_use[val_idx].astype(int).tolist()],
                "y_true": y_va.astype(int),
                "y_cont": y_cont.astype(float),
                "y_pred": y_pred.astype(int),
                "fold": fold,
            }
        )
        pred_df.to_csv(fold_dir / "val_predictions.csv", index=False, encoding="utf-8")
        oof_rows.extend(pred_df.to_dict(orient="records"))

        # Confusion matrix (rounded predictions).
        labels = list(range(clip_min, clip_max + 1))
        cm = confusion_matrix(y_va, y_pred, labels=labels)
        _save_confusion_matrix(cm, labels, fold_dir / "confusion_matrix_val.png", title=f"Fold {fold} (val)")

        # Per-task metrics.
        by_task_rows: list[dict[str, Any]] = []
        for t in range(len(task_names_use)):
            mask = pred_df["task"].to_numpy().astype(int) == t
            if int(mask.sum()) == 0:
                continue
            m = compute_ordinal_metrics(
                y_true=pred_df.loc[mask, "y_true"].to_numpy(),
                y_pred=pred_df.loc[mask, "y_pred"].to_numpy(),
                y_cont=pred_df.loc[mask, "y_cont"].to_numpy(),
                n_classes=int(n_classes),
            )
            by_task_rows.append(
                {
                    "fold": fold,
                    "task": t,
                    "task_name": task_names_use[t],
                    "n": int(mask.sum()),
                    "mae": m.mae,
                    "kappa_quadratic": m.kappa_quadratic,
                    "spearman": m.spearman,
                    "acc_pm1": m.acc_pm1,
                    "acc": m.acc,
                    "bacc": m.bacc,
                    "macro_f1": m.macro_f1,
                    "weighted_f1": m.weighted_f1,
                    "ccc": m.ccc,
                }
            )
        pd.DataFrame(by_task_rows).to_csv(fold_dir / "val_metrics_by_task.csv", index=False, encoding="utf-8")

        # Save model.
        with open(fold_dir / "best_model.pkl", "wb") as f:
            pickle.dump(fitted, f, protocol=pickle.HIGHEST_PROTOCOL)

        feat_names = fitted.get_feature_names_out()
        v_pen = fitted.get_penalty_factors().astype(float).tolist()
        (fold_dir / "feature_names.json").write_text(
            json.dumps(
                {
                    "design_feature_names": feat_names,
                    "task_names": task_names_use,
                    "segment_reference": segment_ref,
                    "segment_levels": segment_levels,
                    "segment_col": segment_col,
                    "penalty_factors": v_pen,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Coefficients.
        rows_coef: list[dict[str, Any]] = []
        for name, c, vv in zip(feat_names, fitted.coef_.astype(float).tolist(), v_pen, strict=False):
            rows_coef.append(
                {
                    "feature": name,
                    "coef_model": float(c),
                    "coef_effective": float(c),
                    "penalty_factor": float(vv),
                    "penalized": bool(float(vv) > 0),
                }
            )
        rows_coef.append(
            {
                "feature": "__intercept__",
                "coef_model": float(fitted.intercept_),
                "coef_effective": float(fitted.intercept_),
                "penalty_factor": 0.0,
                "penalized": False,
            }
        )
        pd.DataFrame(rows_coef).to_csv(fold_dir / "coefficients.csv", index=False, encoding="utf-8")
        for r in rows_coef:
            coef_rows_all.append({**r, "fold": int(fold)})

        # Fold info.
        (fold_dir / "fold_info.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "best_params": {
                        "l1_ratio": best_l1,
                        "lambda": best_lambda,
                        "lambda_choice": lambda_choice,
                    },
                    "metrics": {
                        "mae": metrics.mae,
                        "kappa_quadratic": metrics.kappa_quadratic,
                        "spearman": metrics.spearman,
                        "acc_pm1": metrics.acc_pm1,
                        "acc": metrics.acc,
                        "bacc": metrics.bacc,
                        "macro_f1": metrics.macro_f1,
                        "weighted_f1": metrics.weighted_f1,
                        "ccc": metrics.ccc,
                    },
                    "convergence": {
                        "converged": bool(cd_res.converged),
                        "n_iter": int(cd_res.n_iter),
                        "max_iter": int(max_iter),
                        "tol": float(tol),
                    },
                    **({} if fs_summary is None else {"feature_selection": fs_summary}),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        fold_results.append(
            FoldResult(
                fold=fold,
                n_train=int(len(train_idx)),
                n_val=int(len(val_idx)),
                metrics=metrics,
                best_params={
                    "l1_ratio": best_l1,
                    "lambda": best_lambda,
                    "lambda_choice": lambda_choice,
                },
                checkpoint_dir=fold_dir,
            )
        )

        if metrics.kappa_quadratic > best_kappa:
            best_kappa = metrics.kappa_quadratic
            best_fold = fold

    # --- 4.4.3 point estimates ---
    # Fold-mean (each outer fold equal weight).
    fold_mean_metrics = OrdinalMetrics(
        mae=float(np.mean([fr.metrics.mae for fr in fold_results])),
        kappa_quadratic=float(np.mean([fr.metrics.kappa_quadratic for fr in fold_results])),
        spearman=float(np.mean([fr.metrics.spearman for fr in fold_results])),
        acc_pm1=float(np.mean([fr.metrics.acc_pm1 for fr in fold_results])),
        acc=float(np.mean([fr.metrics.acc for fr in fold_results])),
        bacc=float(np.mean([fr.metrics.bacc for fr in fold_results])),
        macro_f1=float(np.mean([fr.metrics.macro_f1 for fr in fold_results])),
        weighted_f1=float(np.mean([fr.metrics.weighted_f1 for fr in fold_results])),
        ccc=float(np.mean([fr.metrics.ccc for fr in fold_results])),
    )

    # OOF pooled (each disc contributes once via its validation fold prediction).
    oof_df = pd.DataFrame(oof_rows)
    if int(len(oof_df)) != int(len(y)):
        raise RuntimeError(
            "OOF pooled predictions do not cover all samples exactly once: "
            f"n_oof={int(len(oof_df))} vs n_samples={int(len(y))}"
        )
    oof_df.to_csv(Path(run_dir) / "oof_predictions.csv", index=False, encoding="utf-8")
    oof_metrics = compute_ordinal_metrics(
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred"].to_numpy(),
        y_cont=oof_df["y_cont"].to_numpy(),
        n_classes=int(n_classes),
    )

    # Coefficient stability across outer folds.
    if coef_rows_all:
        coef_df = pd.DataFrame(coef_rows_all)
        coef_df.to_csv(Path(run_dir) / "coefficients_all_folds.csv", index=False, encoding="utf-8")

        df2 = coef_df[coef_df["feature"] != "__intercept__"].copy()
        if not df2.empty:
            eps = 1e-12
            df2["abs_coef"] = df2["coef_effective"].abs()
            df2["nonzero"] = (df2["abs_coef"] > eps).astype(int)
            summary = (
                df2.groupby("feature", as_index=False)
                .agg(
                    mean_coef=("coef_effective", "mean"),
                    std_coef=("coef_effective", "std"),
                    mean_abs=("abs_coef", "mean"),
                    std_abs=("abs_coef", "std"),
                    nonzero_freq=("nonzero", "mean"),
                )
                .sort_values(["mean_abs", "nonzero_freq"], ascending=[False, False])
                .reset_index(drop=True)
            )
            summary.to_csv(Path(run_dir) / "coefficients_stability.csv", index=False, encoding="utf-8")

    return CVResult(
        fold_results=fold_results,
        best_fold=best_fold,
        fold_mean_metrics=fold_mean_metrics,
        oof_metrics=oof_metrics,
    )
