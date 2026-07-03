from __future__ import annotations

# Example:
#   & D:/anaconda3/envs/pnam/python.exe e:/Elastic Net/tools/calibrate_decision_thresholds.py --run-dir e:/Elastic Net/outputs/run_YYYYMMDD_HHMMSS --apply

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error

from lumbar_enet.decision_thresholds import parse_threshold_matrix, predict_from_y_cont, predict_from_y_cont_by_task


def _safe_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return 0.0


def _metrics(*, y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    K = int(n_classes)
    out = {
        "acc": float(np.mean(y_true == y_pred)) if y_true.size else 0.0,
        "qwk": _safe_qwk(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    labels = list(range(1, K + 1))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Track a common error pattern: true class-2 predicted as class-1.
    if K >= 2:
        true2 = int(cm[1, :].sum())
        mis21 = int(cm[1, 0])
        out["mis_2_to_1"] = float(mis21)
        out["mis_2_to_1_rate"] = float(mis21 / true2) if true2 else float("nan")
        out["true2_n"] = float(true2)
    else:
        out["mis_2_to_1"] = float("nan")
        out["mis_2_to_1_rate"] = float("nan")
        out["true2_n"] = float("nan")
    return out


def _thr_cols(*, thr: np.ndarray, n_thr_cols: int, placeholder: float) -> dict[str, float]:
    thr = np.asarray(thr, dtype=float).reshape(-1)
    cols: dict[str, float] = {}
    for i in range(int(n_thr_cols)):
        if i < thr.size:
            cols[f"thr_{i+1}"] = float(thr[i])
        elif i == 4:
            cols[f"thr_{i+1}"] = float(placeholder)
        else:
            cols[f"thr_{i+1}"] = float("nan")
    return cols


def _objective_tuple(m: dict[str, float]) -> tuple[float, float, float]:
    # Primary: acc (max). Tie-breakers: qwk (max), -mae (max).
    return (float(m.get("acc", 0.0)), float(m.get("qwk", 0.0)), -float(m.get("mae", 0.0)))


def _calibrate_cd(
    *,
    y_cont: np.ndarray,
    y_true: np.ndarray,
    init_thr: np.ndarray,
    grid: np.ndarray,
    n_iter: int,
) -> np.ndarray:
    """
    Coordinate-descent calibration for per-boundary threshold offsets.

    For Elastic Net, we decode:
        y_pred = 1 + sum_k I(y_cont > k + thr_k), k=1..K-1
    where thr_k is in [0, 1] (default 0.5).
    """
    y_cont = np.asarray(y_cont, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    thr = np.asarray(init_thr, dtype=np.float64).reshape(-1).copy()
    if thr.size == 0:
        return thr

    K = int(thr.size + 1)
    for _ in range(int(max(1, n_iter))):
        for j in range(int(thr.size)):
            best_thr = float(thr[j])
            best_met = None
            for cand in grid.tolist():
                thr_try = thr.copy()
                thr_try[j] = float(cand)
                y_hat = predict_from_y_cont(y_cont, thr_try)
                met = _metrics(y_true=y_true, y_pred=y_hat, n_classes=K)
                if best_met is None or _objective_tuple(met) > _objective_tuple(best_met):
                    best_met = met
                    best_thr = float(cand)
            thr[j] = best_thr
    return thr


def _apply_thresholds_by_task(
    *,
    df: pd.DataFrame,
    thr_by_task: np.ndarray,
    n_classes: int,
    out_col: str = "y_pred_calibrated",
) -> pd.DataFrame:
    """Add an integer column `out_col` by decoding y_cont with per-task thresholds."""
    K = int(n_classes)
    need_cols = ["task", "y_cont"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"apply_thresholds_by_task missing required columns: {miss}")

    out = df.copy()
    task = pd.to_numeric(out["task"], errors="raise").astype(int).to_numpy()
    y_cont = pd.to_numeric(out["y_cont"], errors="raise").astype(float).to_numpy()
    y_hat = np.zeros((len(out),), dtype=int)
    for t in np.unique(task).tolist():
        m = task == int(t)
        if int(m.sum()) == 0:
            continue
        y_hat[m] = predict_from_y_cont(y_cont[m], thr_by_task[int(t)])
    out[out_col] = y_hat.astype(int)

    # Keep K in signature for config-driven checks (and parity with ProtoNAM).
    _ = K
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # Clip for numerical stability (exp overflow).
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _ordinal_probs_from_y_cont(
    *,
    y_cont: np.ndarray,
    task: np.ndarray,
    thr_by_task: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive soft ordinal probabilities from y_cont using calibrated thresholds.

    We treat y_cont as a latent continuous score and convert to cumulative probs:
        p_gt_k = sigmoid((y_cont - (k + thr_k(task))) / temperature), k=1..K-1
    then recover class probs:
        p_cls_1 = 1 - p_gt_1
        p_cls_k = p_gt_{k-1} - p_gt_k
        p_cls_K = p_gt_{K-1}

    This does NOT change the discrete prediction (still based on threshold crossing);
    it's only a probability "readout" to enable calibration figures comparable to ProtoNAM.
    """
    y_cont = np.asarray(y_cont, dtype=np.float64).reshape(-1)
    task = np.asarray(task, dtype=int).reshape(-1)
    thr_by_task = np.asarray(thr_by_task, dtype=np.float64)
    if y_cont.shape[0] != task.shape[0]:
        raise ValueError(f"y_cont/task length mismatch: {y_cont.shape[0]} vs {task.shape[0]}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    K = int(thr_by_task.shape[1] + 1)
    ks = np.arange(1, K, dtype=np.float64)  # (K-1,)
    thr_mat = thr_by_task[task]  # (n, K-1)
    bounds = ks[None, :] + thr_mat  # (n, K-1)
    p_gt = _sigmoid((y_cont[:, None] - bounds) / float(temperature))  # (n, K-1)

    q = np.concatenate(
        [
            np.ones((p_gt.shape[0], 1), dtype=np.float64),
            p_gt,
            np.zeros((p_gt.shape[0], 1), dtype=np.float64),
        ],
        axis=1,
    )  # (n, K+1) with q0=1, qK=0
    p_cls = q[:, :-1] - q[:, 1:]  # (n, K)
    p_cls = np.clip(p_cls, 0.0, 1.0)
    # Re-normalize to sum to 1 (handles numerical corner cases).
    denom = p_cls.sum(axis=1, keepdims=True)
    p_cls = np.divide(p_cls, denom, out=np.full_like(p_cls, 1.0 / float(K)), where=denom > 0)
    return p_gt.astype(np.float32), p_cls.astype(np.float32)


def _fit_temperature_nll(
    *,
    y_cont: np.ndarray,
    y_true: np.ndarray,
    task: np.ndarray,
    thr_by_task: np.ndarray,
    temps: np.ndarray,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Grid-search temperature by minimizing NLL on OOF."""
    y_cont = np.asarray(y_cont, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    task = np.asarray(task, dtype=int).reshape(-1)
    temps = np.asarray(temps, dtype=np.float64).reshape(-1)
    if y_cont.size == 0:
        return 1.0, 0.0

    K = int(np.asarray(thr_by_task).shape[1] + 1)
    y_idx = np.clip(y_true - 1, 0, K - 1)

    best_temp = float(temps[0]) if temps.size else 1.0
    best_nll = float("inf")
    for t in temps.tolist():
        if float(t) <= 0.0:
            continue
        _p_gt, p_cls = _ordinal_probs_from_y_cont(
            y_cont=y_cont,
            task=task,
            thr_by_task=thr_by_task,
            temperature=float(t),
        )
        p_true = p_cls[np.arange(p_cls.shape[0]), y_idx]
        nll = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))
        if nll < best_nll:
            best_nll = nll
            best_temp = float(t)
    if not np.isfinite(best_nll):
        return float(best_temp), float("nan")
    return float(best_temp), float(best_nll)


def calibrate_and_apply(
    *,
    run_dir: str | Path,
    out_csv: str | Path | None = None,
    grid_min: float = 0.25,
    grid_max: float = 0.75,
    grid_step: float = 0.005,
    n_iter: int = 3,
    apply: bool = True,
) -> dict[str, Any]:
    """Calibrate thresholds from fold `val_predictions.csv`; optionally apply `calib_cd_median_apply_all` to pooled OOF."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    schema = json.loads((run_dir / "data_schema.json").read_text(encoding="utf-8"))
    task_names = list(schema.get("task_names") or [])
    if not task_names:
        raise ValueError("data_schema.json missing task_names")

    ord_cfg = cfg.get("ordinal", {}) or {}
    n_classes = int(ord_cfg.get("n_classes", 5))
    K = int(n_classes)
    n_bounds = int(K - 1)

    thr_by_task_base, thr5_placeholder = parse_threshold_matrix(ord_cfg=ord_cfg, task_names=task_names, n_classes=K)
    n_thr_cols = max(5, int(n_bounds))

    # Load per-fold OOF predictions with fold ids.
    rows = []
    ckpt_dir = run_dir / "checkpoints"
    for fold_dir in sorted(ckpt_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except Exception:
            continue
        p = fold_dir / "val_predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, encoding="utf-8")
        df["fold"] = int(fold)
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No fold val_predictions.csv found under: {ckpt_dir}")

    df_all = pd.concat(rows, axis=0, ignore_index=True)
    need_cols = ["fold", "task", "y_true", "y_cont"]
    miss = [c for c in need_cols if c not in df_all.columns]
    if miss:
        raise ValueError(f"Missing required columns in val_predictions.csv: {miss}")

    df_all["task"] = pd.to_numeric(df_all["task"], errors="raise").astype(int)
    df_all["y_true"] = pd.to_numeric(df_all["y_true"], errors="raise").astype(int)
    df_all["y_cont"] = pd.to_numeric(df_all["y_cont"], errors="raise").astype(float)

    grid = np.arange(float(grid_min), float(grid_max) + 1e-12, float(grid_step), dtype=np.float64)
    folds = sorted({int(x) for x in df_all["fold"].unique().tolist()})
    n_tasks = len(task_names)

    out_rows: list[dict[str, Any]] = []

    # Baseline on all pooled OOF (per task).
    for t in range(n_tasks):
        m = df_all["task"].to_numpy() == int(t)
        if int(m.sum()) == 0:
            continue
        y_cont = df_all.loc[m, "y_cont"].to_numpy(dtype=float)
        yt = df_all.loc[m, "y_true"].to_numpy(dtype=int)
        thr = thr_by_task_base[int(t)]
        yp = predict_from_y_cont(y_cont, thr)
        met = _metrics(y_true=yt, y_pred=yp, n_classes=K)
        out_rows.append(
            {
                "method": "baseline_apply_all",
                "fold": "all",
                "task": int(t),
                "task_name": task_names[int(t)],
                "n": int(m.sum()),
                **_thr_cols(thr=thr, n_thr_cols=n_thr_cols, placeholder=thr5_placeholder),
                **met,
            }
        )

    # Fold-in (optimistic) calibration on all pooled OOF.
    for t in range(n_tasks):
        m = df_all["task"].to_numpy() == int(t)
        if int(m.sum()) == 0:
            continue
        y_cont = df_all.loc[m, "y_cont"].to_numpy(dtype=float)
        yt = df_all.loc[m, "y_true"].to_numpy(dtype=int)
        init_thr = thr_by_task_base[int(t)]
        thr_hat = _calibrate_cd(y_cont=y_cont, y_true=yt, init_thr=init_thr, grid=grid, n_iter=int(n_iter))
        yp = predict_from_y_cont(y_cont, thr_hat)
        met = _metrics(y_true=yt, y_pred=yp, n_classes=K)
        out_rows.append(
            {
                "method": "calib_cd_train_all_eval_all",
                "fold": "all",
                "task": int(t),
                "task_name": task_names[int(t)],
                "n": int(m.sum()),
                **_thr_cols(thr=thr_hat, n_thr_cols=n_thr_cols, placeholder=thr5_placeholder),
                **met,
            }
        )

    # Cross-fitted calibration: for each fold, calibrate on other folds and evaluate on this fold.
    thr_by_fold = np.full((len(folds), n_tasks, n_bounds), np.nan, dtype=np.float64)
    for fi, fold in enumerate(folds):
        df_tr = df_all[df_all["fold"] != int(fold)]
        df_va = df_all[df_all["fold"] == int(fold)]
        for t in range(n_tasks):
            tr_m = df_tr["task"].to_numpy() == int(t)
            va_m = df_va["task"].to_numpy() == int(t)
            if int(tr_m.sum()) == 0 or int(va_m.sum()) == 0:
                continue

            y_va_cont = df_va.loc[va_m, "y_cont"].to_numpy(dtype=float)
            y_va = df_va.loc[va_m, "y_true"].to_numpy(dtype=int)
            init_thr = thr_by_task_base[int(t)]

            # Fold-in calibration (optimistic): calibrate on this fold and evaluate on this fold.
            thr_in = _calibrate_cd(y_cont=y_va_cont, y_true=y_va, init_thr=init_thr, grid=grid, n_iter=int(n_iter))
            y_hat_in = predict_from_y_cont(y_va_cont, thr_in)
            met_in = _metrics(y_true=y_va, y_pred=y_hat_in, n_classes=K)
            out_rows.append(
                {
                    "method": "calib_cd_train_fold_eval_fold",
                    "fold": int(fold),
                    "task": int(t),
                    "task_name": task_names[int(t)],
                    "n": int(va_m.sum()),
                    **_thr_cols(thr=thr_in, n_thr_cols=n_thr_cols, placeholder=thr5_placeholder),
                    **met_in,
                }
            )

            y_tr_cont = df_tr.loc[tr_m, "y_cont"].to_numpy(dtype=float)
            y_tr = df_tr.loc[tr_m, "y_true"].to_numpy(dtype=int)
            thr_hat = _calibrate_cd(y_cont=y_tr_cont, y_true=y_tr, init_thr=init_thr, grid=grid, n_iter=int(n_iter))
            thr_by_fold[fi, t, :] = thr_hat

            y_hat = predict_from_y_cont(y_va_cont, thr_hat)
            met = _metrics(y_true=y_va, y_pred=y_hat, n_classes=K)
            out_rows.append(
                {
                    "method": "calib_cd_train_others_eval_fold",
                    "fold": int(fold),
                    "task": int(t),
                    "task_name": task_names[int(t)],
                    "n": int(va_m.sum()),
                    **_thr_cols(thr=thr_hat, n_thr_cols=n_thr_cols, placeholder=thr5_placeholder),
                    **met,
                }
            )

    # Suggested thresholds: median across folds, then apply to all OOF for comparison.
    thr_suggested = np.zeros((n_tasks, n_bounds), dtype=np.float64)
    for t in range(n_tasks):
        thr_med = np.nanmedian(thr_by_fold[:, t, :], axis=0)
        if np.any(np.isnan(thr_med)):
            thr_med = thr_by_task_base[int(t)]
        thr_suggested[int(t), :] = thr_med

        m = df_all["task"].to_numpy() == int(t)
        if int(m.sum()) == 0:
            continue
        y_cont = df_all.loc[m, "y_cont"].to_numpy(dtype=float)
        yt = df_all.loc[m, "y_true"].to_numpy(dtype=int)
        yp = predict_from_y_cont(y_cont, thr_med)
        met = _metrics(y_true=yt, y_pred=yp, n_classes=K)
        out_rows.append(
            {
                "method": "calib_cd_median_apply_all",
                "fold": "all",
                "task": int(t),
                "task_name": task_names[int(t)],
                "n": int(m.sum()),
                **_thr_cols(thr=thr_med, n_thr_cols=n_thr_cols, placeholder=thr5_placeholder),
                **met,
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_path = Path(out_csv) if out_csv else (run_dir / "tmp_threshold_calibration.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "n_classes": int(K),
        "task_names": list(task_names),
        "method": "calib_cd_median_apply_all",
        "objective": {"primary": "acc", "tie_breakers": ["qwk", "-mae"]},
        "calibration_csv": str(out_path),
    }

    if not apply:
        print(f"Wrote: {out_path}")
        return summary

    # --- Apply suggested thresholds to pooled OOF and compute "final" metrics ---
    oof_path = run_dir / "oof_predictions.csv"
    if oof_path.exists():
        oof_df = pd.read_csv(oof_path, encoding="utf-8")
    else:
        # Fallback: pooled val_predictions are OOF by construction (order is not critical for metrics).
        oof_df = df_all.copy()

    oof_df_cal = _apply_thresholds_by_task(df=oof_df, thr_by_task=thr_suggested, n_classes=K, out_col="y_pred_calibrated")

    # If probabilities are missing (Elastic Net), add a soft ordinal probability readout to enable
    # the same calibration plots as ProtoNAM (p_gt_*, p_cls_* columns).
    prob_cols = [f"p_cls_{k}" for k in range(1, K + 1)]
    if any(c not in oof_df_cal.columns for c in prob_cols):
        try:
            y_cont = pd.to_numeric(oof_df_cal["y_cont"], errors="raise").astype(float).to_numpy()
            y_true = pd.to_numeric(oof_df_cal["y_true"], errors="raise").astype(int).to_numpy()
            task = pd.to_numeric(oof_df_cal["task"], errors="raise").astype(int).to_numpy()

            # Temperature grid on log-scale (broad but cheap).
            temps = np.logspace(np.log10(0.05), np.log10(5.0), num=50, dtype=np.float64)
            best_temp, best_nll = _fit_temperature_nll(
                y_cont=y_cont,
                y_true=y_true,
                task=task,
                thr_by_task=thr_suggested,
                temps=temps,
            )
            p_gt, p_cls = _ordinal_probs_from_y_cont(
                y_cont=y_cont,
                task=task,
                thr_by_task=thr_suggested,
                temperature=float(best_temp),
            )
            for kk in range(1, K):
                col = f"p_gt_{kk}"
                if col not in oof_df_cal.columns:
                    oof_df_cal[col] = p_gt[:, kk - 1].astype(float)
            for kk in range(1, K + 1):
                col = f"p_cls_{kk}"
                if col not in oof_df_cal.columns:
                    oof_df_cal[col] = p_cls[:, kk - 1].astype(float)

            prob_readout = run_dir / "probability_readout.json"
            prob_readout.write_text(
                json.dumps(
                    {
                        "method": "logistic_temperature_from_y_cont",
                        "temperature": float(best_temp),
                        "nll_oof": float(best_nll),
                        "note": (
                            "This file is only for generating probability calibration figures. "
                            "Discrete predictions still use threshold crossings on y_cont."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            summary["probability_readout_json"] = str(prob_readout)
            summary["probability_temperature"] = float(best_temp)
            summary["probability_nll_oof"] = float(best_nll)
        except Exception:
            # Probability readout is optional; keep calibration robust.
            pass
    oof_pred_out = run_dir / "oof_predictions_calibrated.csv"
    oof_df_cal.to_csv(oof_pred_out, index=False, encoding="utf-8")

    from lumbar_enet.metrics import compute_ordinal_metrics  # local import to keep CLI fast when apply=False

    if "y_cont" not in oof_df_cal.columns:
        raise ValueError("oof_predictions missing y_cont; cannot compute full metrics.")
    m_cal = compute_ordinal_metrics(
        y_true=oof_df_cal["y_true"].to_numpy(),
        y_pred=oof_df_cal["y_pred_calibrated"].to_numpy(),
        y_cont=oof_df_cal["y_cont"].to_numpy(),
        n_classes=int(K),
    )
    oof_metrics_out = run_dir / "oof_metrics_calibrated.json"
    oof_metrics_out.write_text(
        json.dumps(
            {
                "method": "calib_cd_median_apply_all",
                "objective": {"primary": "acc", "tie_breakers": ["qwk", "-mae"]},
                "decision_threshold_source": "decision_thresholds_calibrated.json",
                "point_estimate": {
                    "mae": float(m_cal.mae),
                    "kappa_quadratic": float(m_cal.kappa_quadratic),
                    "spearman": float(m_cal.spearman),
                    "acc_pm1": float(m_cal.acc_pm1),
                    "acc": float(m_cal.acc),
                    "bacc": float(m_cal.bacc),
                    "macro_f1": float(m_cal.macro_f1),
                    "weighted_f1": float(m_cal.weighted_f1),
                    "ccc": float(m_cal.ccc),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Per-task calibrated metrics.
    by_task_rows: list[dict[str, Any]] = []
    for t in range(n_tasks):
        mask = pd.to_numeric(oof_df_cal["task"], errors="raise").astype(int).to_numpy() == int(t)
        if int(mask.sum()) == 0:
            continue
        m_t = compute_ordinal_metrics(
            y_true=oof_df_cal.loc[mask, "y_true"].to_numpy(),
            y_pred=oof_df_cal.loc[mask, "y_pred_calibrated"].to_numpy(),
            y_cont=oof_df_cal.loc[mask, "y_cont"].to_numpy(),
            n_classes=int(K),
        )
        by_task_rows.append(
            {
                "task": int(t),
                "task_name": task_names[int(t)],
                "n": int(mask.sum()),
                "mae": float(m_t.mae),
                "kappa_quadratic": float(m_t.kappa_quadratic),
                "spearman": float(m_t.spearman),
                "acc_pm1": float(m_t.acc_pm1),
                "acc": float(m_t.acc),
                "bacc": float(m_t.bacc),
                "macro_f1": float(m_t.macro_f1),
                "weighted_f1": float(m_t.weighted_f1),
                "ccc": float(m_t.ccc),
            }
        )
    by_task_out = run_dir / "oof_metrics_by_task_calibrated.csv"
    pd.DataFrame(by_task_rows).to_csv(by_task_out, index=False, encoding="utf-8")

    # Persist final thresholds as an explicit artifact (can be treated as part of model config/weights).
    thr_dict: dict[str, list[float]] = {}
    for t, name in enumerate(list(task_names)):
        thr_dict[str(name)] = [float(x) for x in thr_suggested[int(t), :].tolist()]
    thr_out = run_dir / "decision_thresholds_calibrated.json"
    thr_out.write_text(
        json.dumps(
            {
                "method": "calib_cd_median_apply_all",
                "objective": {"primary": "acc", "tie_breakers": ["qwk", "-mae"]},
                "grid": {
                    "min": float(grid_min),
                    "max": float(grid_max),
                    "step": float(grid_step),
                    "n_iter": int(n_iter),
                },
                "n_classes": int(K),
                "note": "Decode: y_pred = 1 + sum_k I(y_cont > k + thr_k), k=1..K-1. Only K-1 thresholds are used.",
                "thresholds_by_task": thr_dict,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary.update(
        {
            "thresholds_json": str(thr_out),
            "oof_predictions_calibrated": str(oof_pred_out),
            "oof_metrics_calibrated": str(oof_metrics_out),
            "oof_metrics_by_task_calibrated": str(by_task_out),
            "thresholds_by_task": thr_dict,
            "oof_point_estimate_calibrated": {
                "mae": float(m_cal.mae),
                "kappa_quadratic": float(m_cal.kappa_quadratic),
                "spearman": float(m_cal.spearman),
                "acc_pm1": float(m_cal.acc_pm1),
                "acc": float(m_cal.acc),
                "bacc": float(m_cal.bacc),
                "macro_f1": float(m_cal.macro_f1),
                "weighted_f1": float(m_cal.weighted_f1),
                "ccc": float(m_cal.ccc),
            },
        }
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="e.g. outputs/run_YYYYMMDD_HHMMSS")
    ap.add_argument("--out-csv", default=None, help="default: {run_dir}/tmp_threshold_calibration.csv")
    ap.add_argument("--grid-min", type=float, default=0.25)
    ap.add_argument("--grid-max", type=float, default=0.75)
    ap.add_argument("--grid-step", type=float, default=0.005)
    ap.add_argument("--n-iter", type=int, default=3, help="coordinate-descent iterations")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="also write calibrated thresholds + apply to pooled OOF to compute final discrete grades + metrics",
    )
    args = ap.parse_args()

    calibrate_and_apply(
        run_dir=args.run_dir,
        out_csv=args.out_csv,
        grid_min=float(args.grid_min),
        grid_max=float(args.grid_max),
        grid_step=float(args.grid_step),
        n_iter=int(args.n_iter),
        apply=bool(args.apply),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
