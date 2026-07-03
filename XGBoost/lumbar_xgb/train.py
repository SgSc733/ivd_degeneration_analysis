from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

from lumbar_xgb.metrics import OrdinalMetrics, compute_ordinal_metrics
from lumbar_xgb.ordinal import build_decision_thresholds, coral_loss_np, decode_coral, fit_coral_thresholds
from lumbar_xgb.preprocess import RankGaussZScore
from lumbar_xgb.data import load_model_input_csv
from lumbar_xgb.feature_selection import FeatureSelectionDataset, FoldFeatureSelector

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    metrics: OrdinalMetrics
    checkpoint_dir: Path


@dataclass(frozen=True)
class CVResult:
    fold_results: List[FoldResult]
    best_fold: int
    # Per 方案1.md 4.4.2, the point estimate should be computed on the pooled OOF set
    # (each disc uses its validation-fold prediction), not by averaging per-fold metrics.
    mean_metrics: OrdinalMetrics  # alias for oof_metrics
    fold_mean_metrics: OrdinalMetrics
    oof_metrics: OrdinalMetrics


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


def _make_sample_weights(*, meta: pd.DataFrame, y: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Default weighting:
    - patient_balance: each patient has total weight 1.0 (w=1/D_i per disc)
    - optional class_weight: (1/n_c)^{gamma_w}, computed on the fold's training set only.
    """
    n = int(len(meta))
    w = np.ones(n, dtype=np.float32)

    weights_cfg = cfg.get("weights", {}) or {}
    if bool(weights_cfg.get("patient_balance", True)):
        pid = meta["patient_id"].astype(str).to_numpy()
        # Count discs per patient within this fold.
        uniq, counts = np.unique(pid, return_counts=True)
        count_map = {u: c for u, c in zip(uniq.tolist(), counts.tolist())}
        w_pat = np.asarray([1.0 / float(count_map[p]) for p in pid], dtype=np.float32)
        w *= w_pat

    cls_cfg = (weights_cfg.get("class_weight", {}) or {}) if isinstance(weights_cfg.get("class_weight", {}), dict) else {}
    if bool(cls_cfg.get("enabled", False)):
        gamma_w = float(cls_cfg.get("gamma_w", 0.0))
        w_max = float(cls_cfg.get("w_max", 5.0))
        y = np.asarray(y).astype(int)
        uniq_y, counts_y = np.unique(y, return_counts=True)
        count_map_y = {int(c): int(nc) for c, nc in zip(uniq_y.tolist(), counts_y.tolist())}
        w_cls = np.asarray([(1.0 / float(count_map_y[int(yy)])) ** gamma_w for yy in y], dtype=np.float32)
        w *= w_cls
        w = np.minimum(w, np.float32(w_max))

    return w


def _save_confusion_matrix(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    out_path: Path,
    normalize: bool,
    title: str,
) -> None:
    labels = np.arange(1, n_classes + 1, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm_plot = cm.astype(float)
        row_sum = cm_plot.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm_plot = cm_plot / row_sum
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels([str(i) for i in labels])
    ax.set_yticklabels([str(i) for i in labels])

    thresh = cm_plot.max() / 2.0 if cm_plot.size else 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_plot[i, j]
            txt = format(val, fmt)
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_ycont_scatter(*, y_true: np.ndarray, y_cont: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_cont, s=18, alpha=0.7)
    ax.set_xlabel("True grade")
    ax.set_ylabel("Continuous output (CORAL y_cont)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _metrics_by_task(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_cont: np.ndarray,
    tasks: np.ndarray,
    task_names: list[str],
    n_classes: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tasks = np.asarray(tasks).astype(int)
    for t, name in enumerate(task_names):
        m = tasks == t
        if not bool(np.any(m)):
            continue
        met = compute_ordinal_metrics(y_true=y_true[m], y_pred=y_pred[m], y_cont=y_cont[m], n_classes=n_classes)
        rows.append(
            {
                "task": int(t),
                "task_name": name,
                "n": int(m.sum()),
                "mae": met.mae,
                "kappa_quadratic": met.kappa_quadratic,
                "spearman": met.spearman,
                "acc_pm1": met.acc_pm1,
                "acc": met.acc,
                "bacc": met.bacc,
                "macro_f1": met.macro_f1,
                "weighted_f1": met.weighted_f1,
                "ccc": met.ccc,
            }
        )
    return pd.DataFrame(rows)


def _split_cont_and_segment_cols(feature_names: list[str], *, seg_prefix: str = "seg_") -> Tuple[list[str], list[str]]:
    seg_cols = [c for c in feature_names if str(c).startswith(seg_prefix)]
    seg_set = set(seg_cols)
    cont_cols = [c for c in feature_names if c not in seg_set]
    return cont_cols, seg_cols


def _is_higher_better_eval_metric(metric_key: str) -> bool:
    # A small heuristic; our default regression metrics (mae/rmse) are lower-is-better.
    m = str(metric_key).lower()
    return any(tok in m for tok in ("auc", "map", "ndcg"))


def _make_iteration_candidates(
    *,
    strategy: str,
    best_iter_reg: int,
    train_hist: list[float],
    val_hist: list[float],
    metric_key: str,
    topk: int,
    window: int,
) -> list[int]:
    n_iter = int(max(len(train_hist), len(val_hist)))
    if n_iter <= 0:
        return [int(best_iter_reg)]

    strategy = str(strategy)
    best_iter_reg = int(np.clip(int(best_iter_reg), 0, n_iter - 1))

    if strategy == "topk_reg":
        hist = np.asarray(val_hist if len(val_hist) > 0 else train_hist, dtype=float)
        if hist.size <= 0:
            cands = [best_iter_reg]
        else:
            k = int(max(1, min(int(topk), int(hist.size))))
            higher_better = _is_higher_better_eval_metric(metric_key)
            order = np.argsort(-hist) if higher_better else np.argsort(hist)
            cands = order[:k].astype(int).tolist()
            cands.append(best_iter_reg)
    elif strategy == "window":
        w = int(max(0, int(window)))
        lo = max(0, best_iter_reg - w)
        hi = min(n_iter - 1, best_iter_reg + w)
        cands = list(range(lo, hi + 1))
    elif strategy == "all":
        cands = list(range(n_iter))
    else:
        raise ValueError(f"Unknown coral.selection.candidates.strategy: {strategy!r}")

    # Ensure uniqueness + sorted for stable tie-breaking.
    cands = sorted({int(c) for c in cands if 0 <= int(c) < n_iter})
    if best_iter_reg not in cands:
        cands.append(best_iter_reg)
        cands = sorted(cands)
    return cands


def train_group_kfold(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    tasks: np.ndarray,
    task_names: list[str],
    groups: np.ndarray,
    run_dir: Path,
    cfg: Dict[str, Any],
    meta: pd.DataFrame,
    id_col: str,
    feature_selection_dataset: FeatureSelectionDataset | None = None,
) -> CVResult:
    xgb = _require_xgboost()

    cv_cfg = cfg.get("cv", {}) or {}
    n_splits = int(cv_cfg.get("n_splits", 5))

    preprocess_cfg = cfg.get("preprocess", {}) or {}
    preprocess_enabled = bool(preprocess_cfg.get("enabled", True))

    xgb_cfg = cfg.get("xgb", {}) or {}
    params = dict(xgb_cfg.get("params", {}) or {})
    num_boost_round = int(xgb_cfg.get("num_boost_round", 2000))
    early_stopping_rounds = int(xgb_cfg.get("early_stopping_rounds", 50))
    verbose_eval = xgb_cfg.get("verbose_eval", 50)
    eval_metric = str(xgb_cfg.get("eval_metric", "mae"))
    params.setdefault("eval_metric", eval_metric)

    coral_cfg = cfg.get("coral", {}) or {}
    ord_cfg = cfg.get("ordinal", {}) or {}
    n_classes = int(ord_cfg.get("n_classes", coral_cfg.get("n_classes", 5)))
    decision_thresholds = build_decision_thresholds(ord_cfg=ord_cfg, task_names=task_names, n_classes=n_classes)
    coral_fit_cfg = coral_cfg.get("fit", {}) or {}
    coral_fit_use_sample_weight = bool(coral_fit_cfg.get("use_sample_weight", False))

    coral_sel_cfg = coral_cfg.get("selection", {}) or {}
    coral_sel_enabled = bool(coral_sel_cfg.get("enabled", False))
    coral_sel_metric = str(coral_sel_cfg.get("metric", "mae")).lower()
    if coral_sel_metric not in {"mae", "coral_loss"}:
        raise ValueError("coral.selection.metric must be 'mae' or 'coral_loss'")
    coral_sel_cands_cfg = coral_sel_cfg.get("candidates", {}) or {}
    coral_sel_strategy = str(coral_sel_cands_cfg.get("strategy", "topk_reg"))
    coral_sel_topk = int(coral_sel_cands_cfg.get("topk", 50))
    coral_sel_window = int(coral_sel_cands_cfg.get("window", 80))
    coral_sel_fit_cfg = coral_sel_cfg.get("fit", {}) or {}
    coral_sel_fit_use_sample_weight = bool(
        coral_sel_fit_cfg.get("use_sample_weight", coral_fit_use_sample_weight)
    )
    coral_sel_refit_theta_final = bool(coral_sel_cfg.get("refit_theta_final", True))

    out_base = run_dir / "checkpoints"
    out_base.mkdir(parents=True, exist_ok=True)

    gkf = GroupKFold(n_splits=n_splits)

    fold_results: list[FoldResult] = []
    fold_metrics: list[OrdinalMetrics] = []
    oof_pred_dfs: list[pd.DataFrame] = []

    X = X.copy()
    y = np.asarray(y).astype(int)
    tasks = np.asarray(tasks).astype(int)
    groups = np.asarray(groups)

    data_cfg = cfg.get("data", {}) or {}
    pooling_cfg = data_cfg.get("pooling", {}) or {}
    pooling_feature_types = pooling_cfg.get("feature_types", None)
    if pooling_feature_types is None:
        pooling_feature_types = None
    elif isinstance(pooling_feature_types, str):
        pooling_feature_types = [pooling_feature_types]
    else:
        pooling_feature_types = list(pooling_feature_types)
    label_col = str(data_cfg.get("label_col", "pfirrmann") or "pfirrmann")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        fold_dir = out_base / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fs_summary: dict[str, Any] | None = None
        fs_cfg = cfg.get("feature_selection", {}) or {}
        fs_enabled = bool(fs_cfg.get("enabled", True))
        resolved_id_col = str(data_cfg.get("id_col", id_col or "case_id_层级"))

        if feature_selection_dataset is not None:
            fs_dir = fold_dir / "feature_selection"
            fs_dir.mkdir(parents=True, exist_ok=True)

            if fs_enabled:
                selector = FoldFeatureSelector.from_config(
                    fs_cfg,
                    segments=feature_selection_dataset.task_names,
                    id_col=resolved_id_col,
                )
                train_patient_ids = feature_selection_dataset.subset_patient_ids(train_idx)
                train_conditions = feature_selection_dataset.subset_conditions_by_patients(train_patient_ids)
                train_grade_long = feature_selection_dataset.subset_grade_long_by_patients(train_patient_ids)

                fs_result = selector.fit(
                    conditions_raw=train_conditions,
                    grade_long_df=train_grade_long,
                )
                selector.save_audit(
                    fs_dir,
                    pfirrmann_csv=(cfg.get("labels", {}) or {}).get("xlsx_path"),
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
                case_id_series = gold_long.index.get_level_values(0).astype(str)
                segment_series = gold_long.index.get_level_values(1).astype(str)
                sample_id = pd.Series(
                    case_id_series + "_" + segment_series,
                    index=gold_long.index,
                    name=resolved_id_col,
                )
                selected_full_df = pd.concat([sample_id, gold_long], axis=1).reset_index(drop=True)
                fs_summary = {"enabled": False, "audit_dir": str(fs_dir)}

            selected_full_df.insert(1, label_col, feature_selection_dataset.y.astype(int))
            selected_full_path = fs_dir / "fold_model_input_full.csv"
            selected_full_df.to_csv(selected_full_path, index=False, encoding="utf-8")

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
                add_segment_onehot=bool(data_cfg.get("add_segment_onehot", False)),
                pooling_stats=list(pooling_cfg.get("stats", []) or []),
                pooling_feature_types=pooling_feature_types,
                pooling_pyr_prefix=str(pooling_cfg.get("pyr_prefix", "PyRadiomics_")),
                pooling_deep_prefix=str(pooling_cfg.get("deep_prefix", "DeepPCA_")),
                pooling_tensor_prefix=str(pooling_cfg.get("tensor_prefix", "TensorPCA_")),
                pooling_out_prefix=str(pooling_cfg.get("out_prefix", "pool_")),
            )
            loaded_fold.X.to_pickle(fold_dir / "raw_input_features.pkl")

            X_train = loaded_fold.X.iloc[train_idx].reset_index(drop=True)
            X_val = loaded_fold.X.iloc[val_idx].reset_index(drop=True)
            y_train = loaded_fold.y[train_idx]
            y_val = loaded_fold.y[val_idx]
            task_train = loaded_fold.tasks[train_idx]
            task_val = loaded_fold.tasks[val_idx]
            meta_train = loaded_fold.meta.iloc[train_idx].reset_index(drop=True)
            meta_val = loaded_fold.meta.iloc[val_idx].reset_index(drop=True)
            fold_task_names = list(loaded_fold.task_names)
        else:
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_val = X.iloc[val_idx].reset_index(drop=True)
            y_train = y[train_idx]
            y_val = y[val_idx]
            task_train = tasks[train_idx]
            task_val = tasks[val_idx]
            meta_train = meta.iloc[train_idx].reset_index(drop=True)
            meta_val = meta.iloc[val_idx].reset_index(drop=True)
            fold_task_names = list(task_names)

        cont_feature_names, segment_feature_names = _split_cont_and_segment_cols(list(X_train.columns))
        thr_val = decision_thresholds.for_samples(task_val)

        # Sample weights (train fold only).
        w_train = _make_sample_weights(meta=meta_train, y=y_train, cfg=cfg)

        # Fold-inner preprocessing.
        #
        # If segment one-hot features exist (prefix 'seg_'), we *do not* apply RankGauss+ZScore
        # to them. This matches the doc definition z=[x, onehot(level)] where x is preprocessed
        # continuous features, and onehot is passthrough {0,1}.
        feature_names = cont_feature_names + segment_feature_names
        X_train_cont = X_train[cont_feature_names] if cont_feature_names else pd.DataFrame(index=X_train.index)
        X_val_cont = X_val[cont_feature_names] if cont_feature_names else pd.DataFrame(index=X_val.index)

        pre = None
        if preprocess_enabled and cont_feature_names:
            pre = RankGaussZScore().fit(X_train_cont)
            X_train_cont_np = pre.transform(X_train_cont)
            X_val_cont_np = pre.transform(X_val_cont)
            with open(fold_dir / "preprocessor.pkl", "wb") as f:
                pickle.dump(pre, f)
        elif cont_feature_names:
            X_train_cont_np = X_train_cont.to_numpy(dtype=np.float32, copy=True)
            X_val_cont_np = X_val_cont.to_numpy(dtype=np.float32, copy=True)
        else:
            X_train_cont_np = np.zeros((len(X_train), 0), dtype=np.float32)
            X_val_cont_np = np.zeros((len(X_val), 0), dtype=np.float32)

        if segment_feature_names:
            X_train_seg_np = X_train[segment_feature_names].to_numpy(dtype=np.float32, copy=True)
            X_val_seg_np = X_val[segment_feature_names].to_numpy(dtype=np.float32, copy=True)
            X_train_np = np.concatenate([X_train_cont_np, X_train_seg_np], axis=1)
            X_val_np = np.concatenate([X_val_cont_np, X_val_seg_np], axis=1)
        else:
            X_train_np = X_train_cont_np
            X_val_np = X_val_cont_np

        (fold_dir / "feature_names.json").write_text(
            json.dumps(
                {
                    "feature_names": feature_names,
                    "cont_feature_names": cont_feature_names,
                    "segment_feature_names": segment_feature_names,
                    "preprocess_enabled": preprocess_enabled,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Train XGBoost (Stage A).
        dtrain = xgb.DMatrix(X_train_np, label=y_train.astype(float), weight=w_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val_np, label=y_val.astype(float), feature_names=feature_names)

        evals_result: dict[str, dict[str, list[float]]] = {}
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            verbose_eval=verbose_eval,
            early_stopping_rounds=early_stopping_rounds,
        )

        booster.save_model(fold_dir / "xgb_model.json")

        # Training history.
        hist_rows: list[dict[str, Any]] = []
        # Use the first metric key for logging; users can extend params['eval_metric'] to a list if needed.
        metric_key = eval_metric
        train_hist = (evals_result.get("train", {}) or {}).get(metric_key, [])
        val_hist = (evals_result.get("val", {}) or {}).get(metric_key, [])
        n_iter = int(max(len(train_hist), len(val_hist)))
        for i in range(n_iter):
            hist_rows.append(
                {
                    "iter": i,
                    f"train_{metric_key}": float(train_hist[i]) if i < len(train_hist) else np.nan,
                    f"val_{metric_key}": float(val_hist[i]) if i < len(val_hist) else np.nan,
                }
            )
        pd.DataFrame(hist_rows).to_csv(fold_dir / "training_history.csv", index=False, encoding="utf-8")

        best_iter_reg = int(getattr(booster, "best_iteration", n_iter - 1))
        best_iter_reg = int(np.clip(best_iter_reg, 0, max(0, n_iter - 1)))

        # Select the final boosting iteration t* based on CORAL-calibrated ordinal outputs.
        selected_iter = best_iter_reg
        theta_sweep_best: np.ndarray | None = None
        sweep_rows: list[dict[str, Any]] = []

        if coral_sel_enabled:
            cands = _make_iteration_candidates(
                strategy=coral_sel_strategy,
                best_iter_reg=best_iter_reg,
                train_hist=train_hist,
                val_hist=val_hist,
                metric_key=metric_key,
                topk=coral_sel_topk,
                window=coral_sel_window,
            )

            best_key: tuple[float, float, int] | None = None
            for t in cands:
                # In xgboost, iteration_range end is exclusive.
                s_train_t = booster.predict(dtrain, iteration_range=(0, int(t) + 1))
                s_val_t = booster.predict(dval, iteration_range=(0, int(t) + 1))

                theta_t = fit_coral_thresholds(
                    s_train=s_train_t,
                    y_train=y_train,
                    n_classes=n_classes,
                    cfg_fit=coral_sel_fit_cfg,
                    sample_weight=w_train if coral_sel_fit_use_sample_weight else None,
                )
                decoded_t = decode_coral(s=s_val_t, theta=theta_t, decision_threshold=thr_val)
                met_t = compute_ordinal_metrics(
                    y_true=y_val,
                    y_pred=decoded_t.y_pred,
                    y_cont=decoded_t.y_cont,
                    n_classes=n_classes,
                )
                loss_t = coral_loss_np(s=s_val_t, y_true=y_val, theta=theta_t, n_classes=n_classes)

                sweep_rows.append(
                    {
                        "iter": int(t),
                        f"train_{metric_key}": float(train_hist[int(t)]) if int(t) < len(train_hist) else np.nan,
                        f"val_{metric_key}": float(val_hist[int(t)]) if int(t) < len(val_hist) else np.nan,
                        "mae": met_t.mae,
                        "kappa_quadratic": met_t.kappa_quadratic,
                        "spearman": met_t.spearman,
                        "acc_pm1": met_t.acc_pm1,
                        "acc": met_t.acc,
                        "bacc": met_t.bacc,
                        "macro_f1": met_t.macro_f1,
                        "weighted_f1": met_t.weighted_f1,
                        "ccc": met_t.ccc,
                        "coral_loss_val": float(loss_t),
                    }
                )

                if coral_sel_metric == "mae":
                    key = (float(met_t.mae), float(loss_t), int(t))
                else:
                    key = (float(loss_t), float(met_t.mae), int(t))

                if best_key is None or key < best_key:
                    best_key = key
                    selected_iter = int(t)
                    theta_sweep_best = theta_t

            df_sweep = pd.DataFrame(sweep_rows).sort_values("iter").reset_index(drop=True)
            df_sweep["selected"] = df_sweep["iter"].astype(int) == int(selected_iter)
            df_sweep.to_csv(fold_dir / "coral_iteration_sweep.csv", index=False, encoding="utf-8")

        # In xgboost, iteration_range end is exclusive.
        s_train = booster.predict(dtrain, iteration_range=(0, int(selected_iter) + 1))
        s_val = booster.predict(dval, iteration_range=(0, int(selected_iter) + 1))

        if coral_sel_enabled and (not coral_sel_refit_theta_final) and theta_sweep_best is not None:
            theta = theta_sweep_best
        else:
            # Fit CORAL thresholds on s_train (Stage B).
            theta = fit_coral_thresholds(
                s_train=s_train,
                y_train=y_train,
                n_classes=n_classes,
                cfg_fit=coral_fit_cfg,
                sample_weight=w_train if coral_fit_use_sample_weight else None,
            )
        (fold_dir / "coral_thresholds.json").write_text(
            json.dumps({"thresholds_theta": theta.tolist()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        decoded_val = decode_coral(s=s_val, theta=theta, decision_threshold=thr_val)
        metrics = compute_ordinal_metrics(
            y_true=y_val,
            y_pred=decoded_val.y_pred,
            y_cont=decoded_val.y_cont,
            n_classes=n_classes,
        )
        fold_metrics.append(metrics)

        # Per-task metrics.
        m_task = _metrics_by_task(
            y_true=y_val,
            y_pred=decoded_val.y_pred,
            y_cont=decoded_val.y_cont,
            tasks=task_val,
            task_names=fold_task_names,
            n_classes=n_classes,
        )
        m_task.to_csv(fold_dir / "val_metrics_by_task.csv", index=False, encoding="utf-8")

        # Validation predictions table.
        pred_df = meta_val.copy()
        pred_df.insert(0, "orig_index", val_idx)
        pred_df["y_true"] = y_val.astype(int)
        pred_df["s_pred"] = s_val.astype(float)
        pred_df["y_pred"] = decoded_val.y_pred.astype(int)
        pred_df["y_cont"] = decoded_val.y_cont.astype(float)
        for k in range(n_classes - 1):
            pred_df[f"p_gt_{k+1}"] = decoded_val.p_gt[:, k].astype(float)
        for k in range(n_classes):
            pred_df[f"p_cls_{k+1}"] = decoded_val.class_probs[:, k].astype(float)
        pred_df["task"] = task_val.astype(int)
        pred_df["task_name"] = [fold_task_names[int(t)] for t in task_val.tolist()]
        pred_df.to_csv(fold_dir / "val_predictions.csv", index=False, encoding="utf-8")
        oof_pred_dfs.append(pred_df)

        # Plots.
        _save_confusion_matrix(
            y_true=y_val,
            y_pred=decoded_val.y_pred,
            n_classes=n_classes,
            out_path=fold_dir / "confusion_matrix_val.png",
            normalize=False,
            title=f"Fold {fold} - Confusion Matrix (val)",
        )
        _save_confusion_matrix(
            y_true=y_val,
            y_pred=decoded_val.y_pred,
            n_classes=n_classes,
            out_path=fold_dir / "confusion_matrix_val_norm.png",
            normalize=True,
            title=f"Fold {fold} - Confusion Matrix (val, norm)",
        )
        _save_ycont_scatter(
            y_true=y_val,
            y_cont=decoded_val.y_cont,
            out_path=fold_dir / "ycont_scatter_val.png",
            title=f"Fold {fold} - y_cont vs y_true (val)",
        )

        # Fold info summary for reproducibility and leakage checks.
        fold_info = {
            "fold": int(fold),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_train_patients": int(pd.Series(meta_train["patient_id"]).nunique()),
            "n_val_patients": int(pd.Series(meta_val["patient_id"]).nunique()),
            "train_patients": sorted(pd.Series(meta_train["patient_id"]).astype(str).unique().tolist()),
            "val_patients": sorted(pd.Series(meta_val["patient_id"]).astype(str).unique().tolist()),
            # NOTE: best_iteration_reg comes from XGBoost early stopping on the regression metric.
            # selected_iteration is the final t* used for all Stage B outputs (predictions/metrics/explanations).
            "xgb_best_iteration_reg": int(best_iter_reg),
            "xgb_selected_iteration": int(selected_iter),
            "xgb_best_iteration": int(best_iter_reg),  # legacy key (kept for backward compatibility)
            "xgb_params": params,
            "preprocess_enabled": preprocess_enabled,
            "feature_names": feature_names,
            "cont_feature_names": cont_feature_names,
            "segment_feature_names": segment_feature_names,
            "weights_cfg": cfg.get("weights", {}) or {},
            "coral_use_sample_weight": bool(coral_fit_use_sample_weight),
            "coral_selection": {
                "enabled": bool(coral_sel_enabled),
                "metric": coral_sel_metric,
                "strategy": coral_sel_strategy,
                "topk": int(coral_sel_topk),
                "window": int(coral_sel_window),
                "fit_use_sample_weight": bool(coral_sel_fit_use_sample_weight),
                "refit_theta_final": bool(coral_sel_refit_theta_final),
            },
            "coral_thresholds_theta": theta.tolist(),
            "metrics_val": {
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
            **({} if fs_summary is None else {"feature_selection": fs_summary}),
        }
        (fold_dir / "fold_info.json").write_text(json.dumps(fold_info, ensure_ascii=False, indent=2), encoding="utf-8")

        fold_results.append(
            FoldResult(
                fold=fold,
                n_train=int(len(train_idx)),
                n_val=int(len(val_idx)),
                metrics=metrics,
                checkpoint_dir=fold_dir,
            )
        )

    # --- pooled OOF predictions + metrics (point estimate, 方案1.md 4.4.2) ---
    if not oof_pred_dfs:
        raise RuntimeError("Missing fold validation predictions; cannot compute pooled OOF metrics.")

    oof_df = pd.concat(oof_pred_dfs, axis=0, ignore_index=True)
    if "orig_index" not in oof_df.columns:
        raise RuntimeError("val_predictions missing orig_index; cannot pool OOF predictions.")
    if int(oof_df["orig_index"].nunique()) != int(len(y)):
        raise RuntimeError(
            "OOF pooled predictions do not cover all samples exactly once: "
            f"n_unique_orig_index={int(oof_df['orig_index'].nunique())} vs n_samples={int(len(y))}"
        )
    oof_df = oof_df.sort_values("orig_index").reset_index(drop=True)
    oof_df.to_csv(run_dir / "oof_predictions.csv", index=False, encoding="utf-8")

    oof_metrics = compute_ordinal_metrics(
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred"].to_numpy(),
        y_cont=oof_df["y_cont"].to_numpy(),
        n_classes=n_classes,
    )

    _save_confusion_matrix(
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred"].to_numpy(),
        n_classes=n_classes,
        out_path=run_dir / "confusion_matrix_oof.png",
        normalize=False,
        title="OOF pooled confusion matrix",
    )
    _save_confusion_matrix(
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred"].to_numpy(),
        n_classes=n_classes,
        out_path=run_dir / "confusion_matrix_oof_norm.png",
        normalize=True,
        title="OOF pooled confusion matrix (row-normalized)",
    )
    _save_ycont_scatter(
        y_true=oof_df["y_true"].to_numpy(),
        y_cont=oof_df["y_cont"].to_numpy(),
        out_path=run_dir / "ycont_scatter_oof.png",
        title="OOF pooled continuous output vs true",
    )

    # Fold-mean summary (for debugging only; not the point estimate).
    fold_mean_metrics = OrdinalMetrics(
        mae=float(np.mean([m.mae for m in fold_metrics])),
        kappa_quadratic=float(np.mean([m.kappa_quadratic for m in fold_metrics])),
        spearman=float(np.mean([m.spearman for m in fold_metrics])),
        acc_pm1=float(np.mean([m.acc_pm1 for m in fold_metrics])),
        acc=float(np.mean([m.acc for m in fold_metrics])),
        bacc=float(np.mean([m.bacc for m in fold_metrics])),
        macro_f1=float(np.mean([m.macro_f1 for m in fold_metrics])),
        weighted_f1=float(np.mean([m.weighted_f1 for m in fold_metrics])),
        ccc=float(np.mean([m.ccc for m in fold_metrics])),
    )

    best_fold = int(max(fold_results, key=lambda fr: fr.metrics.kappa_quadratic).fold)

    return CVResult(
        fold_results=fold_results,
        best_fold=best_fold,
        mean_metrics=oof_metrics,
        fold_mean_metrics=fold_mean_metrics,
        oof_metrics=oof_metrics,
    )
