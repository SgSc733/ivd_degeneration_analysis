from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from lumbar_ebm.calibration import fit_calibrator, parse_decision_threshold_default
from lumbar_ebm.data import load_model_input_csv
from lumbar_ebm.feature_selection import FeatureSelectionDataset, FoldFeatureSelector
from lumbar_ebm.metrics import OrdinalMetrics, compute_ordinal_metrics
from lumbar_ebm.modeling import build_ebm_regressor, make_feature_types


def _thr_vec_from_any(v: Any, *, n_bounds: int) -> list[float]:
    if isinstance(v, (int, float)):
        return [float(v)] * int(n_bounds)
    if isinstance(v, (list, tuple, np.ndarray)):
        vv = list(v)
        if len(vv) < int(n_bounds):
            raise ValueError(f"threshold list must have length >= {int(n_bounds)} (got {len(vv)})")
        return [float(x) for x in vv[: int(n_bounds)]]
    raise TypeError("threshold must be a number or a list/tuple/ndarray.")


def _parse_threshold_matrix(
    *,
    ord_cfg: dict[str, Any],
    task_names: list[str],
    n_classes: int,
) -> np.ndarray:
    """
    Parse config.ordinal.decision_threshold to a (n_tasks, K-1) matrix.
    Matches ProtoNAM schema to keep comparisons fair.
    """
    K = int(n_classes)
    n_bounds = int(K - 1)
    spec = ord_cfg.get("decision_threshold", 0.5)

    default_vec = _thr_vec_from_any(0.5, n_bounds=n_bounds)
    if isinstance(spec, dict):
        default_raw = spec.get("default", 0.5)
        default_vec = _thr_vec_from_any(default_raw, n_bounds=n_bounds)
        thr = np.tile(np.asarray(default_vec, dtype=np.float64)[None, :], (len(task_names), 1))

        by_task = spec.get("by_task", None)
        if by_task is None:
            return thr

        name_to_idx = {name: i for i, name in enumerate(list(task_names))}
        if isinstance(by_task, dict):
            for k, v in by_task.items():
                if isinstance(k, str) and k in name_to_idx:
                    idx = int(name_to_idx[k])
                else:
                    idx = int(k)
                thr[idx, :] = np.asarray(_thr_vec_from_any(v, n_bounds=n_bounds), dtype=np.float64)
            return thr

        if isinstance(by_task, (list, tuple, np.ndarray)) and by_task and isinstance(list(by_task)[0], (list, tuple, np.ndarray)):
            by_task_list = list(by_task)
            if len(by_task_list) != len(task_names):
                raise ValueError(f"by_task rows must equal n_tasks={len(task_names)} (got {len(by_task_list)})")
            thr = np.asarray([_thr_vec_from_any(row, n_bounds=n_bounds) for row in by_task_list], dtype=np.float64)
            return thr

        raise TypeError("decision_threshold.by_task must be a dict or a list of lists.")

    if isinstance(spec, (int, float, list, tuple, np.ndarray)):
        default_vec = _thr_vec_from_any(spec, n_bounds=n_bounds)
        thr = np.tile(np.asarray(default_vec, dtype=np.float64)[None, :], (len(task_names), 1))
        return thr

    raise TypeError("ordinal.decision_threshold must be a number, list, or dict.")


def _compute_sample_weight(
    *,
    y: np.ndarray,
    n_classes: int,
    mode: str,
    power: float,
) -> np.ndarray | None:
    mode = str(mode or "none").lower().strip()
    if mode in {"none", "null", "false", "0"}:
        return None
    if mode not in {"balanced"}:
        raise ValueError(f"Unsupported sample_weight mode: {mode!r}. Supported: 'none', 'balanced'.")

    y = np.asarray(y).astype(int)
    n = float(len(y))
    if n <= 0:
        return None

    counts = np.bincount(y, minlength=n_classes + 1).astype(float)
    # weight_c = n / (K * n_c), then optionally softened by power in (0,1].
    w = np.zeros_like(counts)
    for c in range(1, n_classes + 1):
        if counts[c] > 0:
            w[c] = (n / (float(n_classes) * counts[c])) ** float(power)
    sw = w[y]
    # Normalize so the mean weight is 1.0 (keeps loss scale comparable across runs).
    m = float(np.mean(sw)) if sw.size else 1.0
    if m > 0:
        sw = sw / m
    return sw.astype(np.float32)


@dataclass(frozen=True)
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    metrics: OrdinalMetrics
    checkpoint_dir: Path


@dataclass(frozen=True)
class CVResult:
    fold_results: list[FoldResult]
    best_fold: int
    # CV summary metrics computed on concatenated validation predictions across folds.
    # This is sample-size-weighted by construction and avoids bias when fold sizes differ.
    mean_metrics: OrdinalMetrics
    # Unweighted average of per-fold metrics (kept for reference).
    mean_metrics_unweighted: OrdinalMetrics


def _save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def train_group_kfold(
    *,
    X_ebm: pd.DataFrame,
    X_num: pd.DataFrame,
    y: np.ndarray,
    tasks: np.ndarray,
    task_names: list[str],
    groups: np.ndarray,
    run_dir: Path,
    cfg: dict[str, Any],
    meta: pd.DataFrame | None = None,
    id_col: str | None = None,
    feature_selection_dataset: FeatureSelectionDataset | None = None,
) -> CVResult:
    n_splits = int(cfg["cv"]["n_splits"])
    n_classes = int(cfg["ordinal"]["n_classes"])
    decision_threshold_default = parse_decision_threshold_default(cfg=cfg, n_classes=n_classes)
    decision_rule = str(cfg["ordinal"].get("decision_rule", "threshold"))
    seed = int((cfg.get("ebm", {}) or {}).get("random_state", 0))
    thr_by_task = _parse_threshold_matrix(ord_cfg=(cfg.get("ordinal", {}) or {}), task_names=task_names, n_classes=n_classes)

    y = np.asarray(y).astype(int)
    tasks = np.asarray(tasks).astype(int)
    if y.min() == 0 and y.max() == n_classes - 1:
        y = y + 1
    if y.min() < 1 or y.max() > n_classes:
        raise ValueError(f"Labels must be in 1..{n_classes} (got min={y.min()}, max={y.max()}).")
    if tasks.min() < 0 or tasks.max() >= len(task_names):
        raise ValueError(f"tasks must be in [0..{len(task_names)-1}] (got min={tasks.min()}, max={tasks.max()}).")

    disc_level_feature_name = str((cfg.get("ebm", {}) or {}).get("feature_disc_level_name", "disc_level"))
    feature_types = make_feature_types(X_ebm.columns, disc_level_feature_name=disc_level_feature_name)

    gkf = GroupKFold(n_splits=n_splits)

    fold_results: list[FoldResult] = []
    best_fold = 0
    best_kappa = -1e9
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_cont_all: list[np.ndarray] = []
    oof_pred_dfs: list[pd.DataFrame] = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_ebm, y, groups=groups)):
        fold_dir = run_dir / "checkpoints" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fs_summary: dict[str, Any] | None = None
        if feature_selection_dataset is not None:
            data_cfg = cfg.get("data", {}) or {}
            pooling_cfg = data_cfg.get("pooling", {}) or {}
            fs_cfg = cfg.get("feature_selection", {}) or {}
            fs_enabled = bool(fs_cfg.get("enabled", True))
            fs_dir = fold_dir / "feature_selection"
            fs_dir.mkdir(parents=True, exist_ok=True)
            resolved_id_col = str(data_cfg.get("id_col", id_col or feature_selection_dataset.id_col))
            label_col = str(data_cfg.get("label_col") or "pfirrmann")

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
                selector.save_audit(
                    fs_dir,
                    pfirrmann_csv=(cfg.get("labels", {}) or {}).get("xlsx_path") or data_cfg.get("pfirrmann_csv_path"),
                    statistics_csv=fs_cfg.get("statistics_csv_path"),
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
                case_id = gold_long.index.get_level_values(0).astype(str)
                segment = gold_long.index.get_level_values(1).astype(str)
                sample_id = pd.Series(case_id + "_" + segment, index=gold_long.index, name=resolved_id_col)
                selected_full_df = pd.concat([sample_id, gold_long], axis=1).reset_index(drop=True)
                fs_summary = {"enabled": False, "audit_dir": str(fs_dir)}

            selected_full_df.insert(1, label_col, feature_selection_dataset.y.astype(int))
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
                label_col=label_col,
                drop_cols=data_cfg.get("drop_cols", []),
                keep_feature_patterns=data_cfg.get("keep_feature_patterns", []),
                classic_prefix=data_cfg.get("classic_prefix", "classic_"),
                patient_id_from_id_col=bool(data_cfg.get("patient_id_from_id_col", True)),
                patient_id_sep=str(data_cfg.get("patient_id_sep", "_")),
                level_from_id_col=bool(data_cfg.get("level_from_id_col", True)),
                level_sep=str(data_cfg.get("level_sep", "_")),
                segment_levels=list(data_cfg.get("segment_levels", feature_selection_dataset.task_names)),
                disc_level_feature_name=disc_level_feature_name,
                pooling_stats=list(pooling_cfg.get("stats", []) or []),
                pooling_feature_types=pooling_feature_types,
                pooling_pyr_prefix=str(pooling_cfg.get("pyr_prefix", "PyRadiomics_")),
                pooling_deep_prefix=str(pooling_cfg.get("deep_prefix", "DeepPCA_")),
                pooling_tensor_prefix=str(pooling_cfg.get("tensor_prefix", "TensorPCA_")),
                pooling_out_prefix=str(pooling_cfg.get("out_prefix", "pool_")),
            )
            X_fold = loaded_fold.X_ebm
            X_fold.to_pickle(fold_dir / "raw_input_features.pkl")
            X_train = X_fold.iloc[train_idx].reset_index(drop=True)
            y_train = loaded_fold.y[train_idx]
            task_train = loaded_fold.tasks[train_idx]
            X_val = X_fold.iloc[val_idx].reset_index(drop=True)
            y_val = loaded_fold.y[val_idx]
            task_val = loaded_fold.tasks[val_idx]
            fold_feature_types = make_feature_types(X_fold.columns, disc_level_feature_name=disc_level_feature_name)
        else:
            X_train = X_ebm.iloc[train_idx].reset_index(drop=True)
            y_train = y[train_idx]
            task_train = tasks[train_idx]
            X_val = X_ebm.iloc[val_idx].reset_index(drop=True)
            y_val = y[val_idx]
            task_val = tasks[val_idx]
            fold_feature_types = feature_types

        # --- Stage-1: EBM ---
        fit_cfg = cfg.get("fit", {}) or {}
        sample_weight = _compute_sample_weight(
            y=y_train,
            n_classes=n_classes,
            mode=str(fit_cfg.get("sample_weight", "none")),
            power=float(fit_cfg.get("sample_weight_power", 1.0)),
        )
        ebm = build_ebm_regressor(cfg=cfg, feature_names=X_train.columns, feature_types=fold_feature_types)
        ebm.fit(X_train, y_train, sample_weight=sample_weight)
        s_train = ebm.predict(X_train).astype(np.float32)
        s_val = ebm.predict(X_val).astype(np.float32)

        # --- Stage-2: CORAL calibration ---
        calib_res = fit_calibrator(s_train=s_train, y_train=y_train, cfg=cfg, seed=seed + fold)
        calib = calib_res.calibrator
        calib.eval()

        import torch

        device = str((cfg.get("calibration", {}) or {}).get("device", "cpu"))
        s_val_t = torch.from_numpy(np.asarray(s_val, dtype=np.float32)).to(device)
        logits = calib(s_val_t).detach()
        decision_threshold_val: float | list[float] | np.ndarray
        if str(decision_rule or "threshold").lower().strip() in {"threshold", "thresh"}:
            # Match ProtoNAM: support per-task thresholds via ordinal.decision_threshold.by_task.
            # If by_task is not set, thr_by_task is still a tiled (n_tasks, K-1) matrix from default.
            decision_threshold_val = thr_by_task[task_val.astype(int)]
        else:
            decision_threshold_val = decision_threshold_default
        out = calib.decode(logits, decision_threshold=decision_threshold_val, decision_rule=decision_rule)
        y_pred = out.y_pred.detach().cpu().numpy().astype(int)
        y_cont = out.y_cont.detach().cpu().numpy().astype(float)
        p_gt = out.p_gt.detach().cpu().numpy().astype(float)
        class_probs = out.class_probs.detach().cpu().numpy().astype(float)

        metrics = compute_ordinal_metrics(y_true=y_val, y_pred=y_pred, y_cont=y_cont, n_classes=n_classes)
        y_true_all.append(y_val.astype(int, copy=False))
        y_pred_all.append(y_pred.astype(int, copy=False))
        y_cont_all.append(y_cont.astype(float, copy=False))

        # --- Save predictions ---
        task_idx0 = task_val.astype(int)
        task_idx1 = task_idx0 + 1
        task_name = np.asarray([task_names[int(t)] for t in task_idx0], dtype=object)
        pred_df = pd.DataFrame(
            {
                "orig_index": val_idx,
                "task": task_idx0,
                "y_true": y_val.astype(int),
                "y_pred": y_pred.astype(int),
                "y_cont": y_cont.astype(float),
                "s_raw": s_val.astype(float),
                "task_idx0": task_idx0,
                "task_idx1": task_idx1,
                "task_name": task_name,
            }
        )
        if meta is not None and id_col is not None and id_col in meta.columns:
            meta_val = meta.iloc[val_idx].reset_index(drop=True)
            pred_df.insert(0, "disc_level", meta_val["disc_level"].astype(str))
            pred_df.insert(0, "patient_id", meta_val["patient_id"].astype(str))
            pred_df.insert(0, id_col, meta_val[id_col].astype(str))

        for k in range(n_classes - 1):
            pred_df[f"p_gt_{k+1}"] = p_gt[:, k]
        for k in range(n_classes):
            pred_df[f"class_prob_{k+1}"] = class_probs[:, k]
        pred_df.to_csv(fold_dir / "val_predictions.csv", index=False, encoding="utf-8")
        oof_pred_dfs.append(pred_df)

        # --- Per-task metrics ---
        by_task_rows: list[dict[str, Any]] = []
        for t in range(len(task_names)):
            mask = pred_df["task_idx0"].to_numpy().astype(int) == t
            if int(mask.sum()) == 0:
                continue
            m = compute_ordinal_metrics(
                y_true=pred_df.loc[mask, "y_true"].to_numpy(),
                y_pred=pred_df.loc[mask, "y_pred"].to_numpy(),
                y_cont=pred_df.loc[mask, "y_cont"].to_numpy(),
                n_classes=n_classes,
            )
            by_task_rows.append(
                {
                    "fold": fold,
                    "task_idx0": t,
                    "task_idx1": t + 1,
                    "task_name": task_names[t],
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

        # --- Save fold artifacts ---
        with open(fold_dir / "ebm.pkl", "wb") as f:
            pickle.dump(ebm, f, protocol=pickle.HIGHEST_PROTOCOL)

        torch.save(
            {
                "state_dict": calib.state_dict(),
                "n_classes": n_classes,
                "learn_alpha": bool((cfg.get("calibration", {}) or {}).get("learn_alpha", True)),
            },
            fold_dir / "calibration.pt",
        )

        alpha = float(calib.alpha.detach().cpu().item())
        theta = calib.thresholds.detach().cpu().numpy().astype(float).tolist()
        theta_s_raw = list(theta)
        theta_scaled = (alpha * np.asarray(theta, dtype=float)).tolist()
        _save_json(
            fold_dir / "fold_info.json",
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
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
                "calibration": {
                    "alpha": alpha,
                    # thresholds on s_raw axis (matches `EBM.md`: p_gt_k = sigmoid(alpha*(s-theta_k)))
                    "thresholds_theta": theta_s_raw,
                    "thresholds_theta_s_raw": theta_s_raw,
                    # thresholds on the scaled (alpha*s_raw) axis for debugging/traceability
                    "thresholds_theta_scaled": theta_scaled,
                    "best_epoch": calib_res.best_epoch,
                    "best_val_loss": calib_res.best_val_loss,
                },
                "ebm": {
                    "feature_names": list(getattr(ebm, "feature_names_in_", list(X_train.columns))),
                    "feature_types": list(getattr(ebm, "feature_types_in_", fold_feature_types)),
                    "term_names": list(getattr(ebm, "term_names_", [])),
                    "intercept": float(getattr(ebm, "intercept_", 0.0)),
                },
                **({} if fs_summary is None else {"feature_selection": fs_summary}),
            },
        )

        fold_results.append(
            FoldResult(
                fold=fold,
                n_train=int(len(train_idx)),
                n_val=int(len(val_idx)),
                metrics=metrics,
                checkpoint_dir=fold_dir,
            )
        )

        if metrics.kappa_quadratic > best_kappa:
            best_kappa = metrics.kappa_quadratic
            best_fold = fold

    # Pooled OOF predictions (for post-hoc threshold calibration; no retraining needed).
    if oof_pred_dfs:
        oof_df = pd.concat(oof_pred_dfs, axis=0, ignore_index=True)
        if "orig_index" in oof_df.columns:
            oof_df = oof_df.sort_values("orig_index").reset_index(drop=True)
        oof_df.to_csv(run_dir / "oof_predictions.csv", index=False, encoding="utf-8")

    # Unweighted average of per-fold metrics (legacy).
    mae_u = float(np.mean([fr.metrics.mae for fr in fold_results]))
    kappa_u = float(np.mean([fr.metrics.kappa_quadratic for fr in fold_results]))
    spearman_u = float(np.mean([fr.metrics.spearman for fr in fold_results]))
    acc_pm1_u = float(np.mean([fr.metrics.acc_pm1 for fr in fold_results]))
    acc_u = float(np.mean([fr.metrics.acc for fr in fold_results]))
    bacc_u = float(np.mean([fr.metrics.bacc for fr in fold_results]))
    macro_f1_u = float(np.mean([fr.metrics.macro_f1 for fr in fold_results]))
    weighted_f1_u = float(np.mean([fr.metrics.weighted_f1 for fr in fold_results]))
    ccc_u = float(np.mean([fr.metrics.ccc for fr in fold_results]))
    mean_metrics_unweighted = OrdinalMetrics(
        mae=mae_u,
        kappa_quadratic=kappa_u,
        spearman=spearman_u,
        acc_pm1=acc_pm1_u,
        acc=acc_u,
        bacc=bacc_u,
        macro_f1=macro_f1_u,
        weighted_f1=weighted_f1_u,
        ccc=ccc_u,
    )

    # Sample-size-weighted summary via concatenation (recommended).
    y_true_cat = np.concatenate(y_true_all, axis=0) if y_true_all else np.asarray([], dtype=int)
    y_pred_cat = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.asarray([], dtype=int)
    y_cont_cat = np.concatenate(y_cont_all, axis=0) if y_cont_all else np.asarray([], dtype=float)
    mean_metrics = compute_ordinal_metrics(y_true=y_true_cat, y_pred=y_pred_cat, y_cont=y_cont_cat, n_classes=n_classes)

    return CVResult(
        fold_results=fold_results,
        best_fold=best_fold,
        mean_metrics=mean_metrics,
        mean_metrics_unweighted=mean_metrics_unweighted,
    )
