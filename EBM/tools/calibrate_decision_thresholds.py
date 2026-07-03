from __future__ import annotations

# 示例：
#   & D:/anaconda3/envs/pnam/python.exe e:/EBM/tools/calibrate_decision_thresholds.py --run-dir e:/EBM/outputs/run_YYYYMMDD_HHMMSS --apply

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, confusion_matrix

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    # Allow running this script from anywhere while still importing `lumbar_ebm.*`.
    sys.path.insert(0, str(_REPO_ROOT))


def _safe_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    # Degenerate cases: kappa is undefined when only one class is present.
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return 0.0


def _predict_from_p_gt(p_gt: np.ndarray, thr_vec: np.ndarray) -> np.ndarray:
    """
    Match CORAL decode used in codebase:
        y_pred = 1 + sum_k I(p_gt_k > thr_k)
    """
    p_gt = np.asarray(p_gt, dtype=float)
    thr_vec = np.asarray(thr_vec, dtype=float)
    return 1 + (p_gt > thr_vec[None, :]).sum(axis=1).astype(int)


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


def _thr_vec_from_any(v: Any, *, n_bounds: int) -> list[float]:
    if isinstance(v, (int, float)):
        return [float(v)] * int(n_bounds)
    if isinstance(v, (list, tuple)):
        if len(v) < int(n_bounds):
            raise ValueError(f"threshold list must have length >= {int(n_bounds)} (got {len(v)})")
        return [float(x) for x in list(v)[: int(n_bounds)]]
    raise TypeError("threshold must be a number or a list/tuple.")


def _parse_threshold_matrix(
    *,
    ord_cfg: dict[str, Any],
    task_names: list[str],
    n_classes: int,
) -> tuple[np.ndarray, float]:
    """
    Parse config.ordinal.decision_threshold to a (n_tasks, K-1) matrix.

    Returns:
        thr_by_task: (n_tasks, K-1)
        thr5_placeholder: a value to fill the 5th column in output CSV (unused when K=5)
    """
    K = int(n_classes)
    n_bounds = int(K - 1)
    spec = ord_cfg.get("decision_threshold", 0.5)

    default_vec = _thr_vec_from_any(0.5, n_bounds=n_bounds)
    thr5_placeholder = 0.5

    if isinstance(spec, dict):
        default_raw = spec.get("default", 0.5)
        default_vec = _thr_vec_from_any(default_raw, n_bounds=n_bounds)
        # Keep something reasonable for the unused 5th slot if user provided a 5-length list.
        if isinstance(default_raw, (list, tuple)) and len(default_raw) >= 5:
            thr5_placeholder = float(default_raw[4])
        else:
            thr5_placeholder = float(default_vec[-1]) if default_vec else 0.5

        by_task = spec.get("by_task", None)
        thr = np.tile(np.asarray(default_vec, dtype=np.float64)[None, :], (len(task_names), 1))
        if by_task is None:
            return thr, thr5_placeholder

        name_to_idx = {name: i for i, name in enumerate(list(task_names))}
        if isinstance(by_task, dict):
            for k, v in by_task.items():
                if isinstance(k, str) and k in name_to_idx:
                    idx = int(name_to_idx[k])
                else:
                    idx = int(k)
                thr[idx, :] = np.asarray(_thr_vec_from_any(v, n_bounds=n_bounds), dtype=np.float64)
            return thr, thr5_placeholder

        if isinstance(by_task, (list, tuple)) and by_task and isinstance(by_task[0], (list, tuple)):
            if len(by_task) != len(task_names):
                raise ValueError(f"by_task rows must equal n_tasks={len(task_names)} (got {len(by_task)})")
            thr = np.asarray([_thr_vec_from_any(row, n_bounds=n_bounds) for row in list(by_task)], dtype=np.float64)
            return thr, thr5_placeholder

        raise TypeError("decision_threshold.by_task must be a dict or a list of lists.")

    if isinstance(spec, (int, float)):
        default_vec = _thr_vec_from_any(spec, n_bounds=n_bounds)
        thr = np.tile(np.asarray(default_vec, dtype=np.float64)[None, :], (len(task_names), 1))
        thr5_placeholder = float(spec)
        return thr, thr5_placeholder

    if isinstance(spec, (list, tuple)):
        default_vec = _thr_vec_from_any(spec, n_bounds=n_bounds)
        thr = np.tile(np.asarray(default_vec, dtype=np.float64)[None, :], (len(task_names), 1))
        thr5_placeholder = float(default_vec[-1]) if default_vec else 0.5
        return thr, thr5_placeholder

    raise TypeError("ordinal.decision_threshold must be a number, list, or dict.")


def _calibrate_cd(
    *,
    p_gt: np.ndarray,
    y_true: np.ndarray,
    init_thr: np.ndarray,
    grid: np.ndarray,
    n_iter: int,
) -> np.ndarray:
    """
    Coordinate descent on thresholds to maximize:
        primary: acc
        tie-breakers: qwk, -mae
    """
    p_gt = np.asarray(p_gt, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    thr = np.asarray(init_thr, dtype=float).copy()
    if thr.ndim != 1:
        raise ValueError("init_thr must be 1-D (K-1,)")

    K = int(max(int(y_true.max()), 1))
    n_bounds = int(thr.shape[0])
    if int(n_bounds) < 1:
        return thr

    def key_of(yp: np.ndarray) -> tuple[float, float, float]:
        met = _metrics(y_true=y_true, y_pred=yp, n_classes=K)
        # maximize: acc, then qwk, then -mae
        return (float(met["acc"]), float(met["qwk"]), -float(met["mae"]))

    best_key = key_of(_predict_from_p_gt(p_gt, thr))

    for _ in range(int(n_iter)):
        improved = False
        for j in range(n_bounds):
            cur = float(thr[j])
            local_best = best_key
            local_thr = float(cur)
            for v in grid.tolist():
                thr[j] = float(v)
                yp = _predict_from_p_gt(p_gt, thr)
                k = key_of(yp)
                if k > local_best:
                    local_best = k
                    local_thr = float(v)
            thr[j] = local_thr
            if local_best > best_key:
                best_key = local_best
                improved = True
        if not improved:
            break

    return thr


def _thr_cols(*, thr: np.ndarray, n_thr_cols: int, placeholder: float) -> dict[str, float]:
    """
    Output fixed-length columns thr1..thr{n_thr_cols} for csv readability.
    For K=5 we still output 5 columns to match historical Excel usage.
    """
    thr = np.asarray(thr, dtype=float)
    out: dict[str, float] = {}
    for i in range(int(n_thr_cols)):
        k = i + 1
        if i < int(thr.shape[0]):
            out[f"thr{k}"] = float(thr[i])
        else:
            out[f"thr{k}"] = float(placeholder)
    return out


def _metrics_to_dict(m: Any) -> dict[str, float]:
    # Convert `lumbar_ebm.metrics.OrdinalMetrics` (or a compatible object) to a JSON-friendly dict.
    return {
        "mae": float(m.mae),
        "kappa_quadratic": float(m.kappa_quadratic),
        "spearman": float(m.spearman),
        "acc_pm1": float(m.acc_pm1),
        "acc": float(m.acc),
        "bacc": float(m.bacc),
        "macro_f1": float(m.macro_f1),
        "weighted_f1": float(m.weighted_f1),
        "ccc": float(m.ccc),
    }


def _apply_thresholds_by_task(
    *,
    df: pd.DataFrame,
    thr_by_task: np.ndarray,
    n_classes: int,
    out_col: str = "y_pred_calibrated",
) -> pd.DataFrame:
    """Add an integer column `out_col` by decoding p_gt_1..p_gt_{K-1} with per-task thresholds."""
    K = int(n_classes)
    need_cols = ["task"] + [f"p_gt_{k}" for k in range(1, K)]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"apply_thresholds_by_task missing required columns: {miss}")

    out = df.copy()
    task = pd.to_numeric(out["task"], errors="raise").astype(int).to_numpy()
    p_gt = out[[f"p_gt_{k}" for k in range(1, K)]].to_numpy(dtype=float)
    y_hat = np.zeros((len(out),), dtype=int)
    for t in np.unique(task).tolist():
        m = task == int(t)
        if int(m.sum()) == 0:
            continue
        y_hat[m] = _predict_from_p_gt(p_gt[m], thr_by_task[int(t)])
    out[out_col] = y_hat.astype(int)
    return out


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

    thr_by_task_base, thr5_placeholder = _parse_threshold_matrix(ord_cfg=ord_cfg, task_names=task_names, n_classes=K)
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
    need_cols = ["fold", "task", "y_true"] + [f"p_gt_{k}" for k in range(1, K)]
    miss = [c for c in need_cols if c not in df_all.columns]
    if miss:
        raise ValueError(f"Missing required columns in val_predictions.csv: {miss}")

    df_all["task"] = pd.to_numeric(df_all["task"], errors="raise").astype(int)
    df_all["y_true"] = pd.to_numeric(df_all["y_true"], errors="raise").astype(int)

    grid = np.arange(float(grid_min), float(grid_max) + 1e-12, float(grid_step), dtype=np.float64)
    folds = sorted({int(x) for x in df_all["fold"].unique().tolist()})
    n_tasks = len(task_names)

    out_rows: list[dict[str, Any]] = []

    # Baseline on all pooled OOF (per task).
    for t in range(n_tasks):
        m = df_all["task"].to_numpy() == int(t)
        if int(m.sum()) == 0:
            continue
        p_gt = df_all.loc[m, [f"p_gt_{k}" for k in range(1, K)]].to_numpy(dtype=float)
        yt = df_all.loc[m, "y_true"].to_numpy(dtype=int)
        thr = thr_by_task_base[int(t)]
        yp = _predict_from_p_gt(p_gt, thr)
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
        p_gt = df_all.loc[m, [f"p_gt_{k}" for k in range(1, K)]].to_numpy(dtype=float)
        yt = df_all.loc[m, "y_true"].to_numpy(dtype=int)
        init_thr = thr_by_task_base[int(t)]
        thr_hat = _calibrate_cd(p_gt=p_gt, y_true=yt, init_thr=init_thr, grid=grid, n_iter=int(n_iter))
        yp = _predict_from_p_gt(p_gt, thr_hat)
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
    # Use NaN init so tasks missing in a fold won't bias nanmedian.
    thr_by_fold = np.full((len(folds), n_tasks, n_bounds), np.nan, dtype=np.float64)
    for fi, fold in enumerate(folds):
        df_tr = df_all[df_all["fold"] != int(fold)]
        df_va = df_all[df_all["fold"] == int(fold)]
        for t in range(n_tasks):
            tr_m = df_tr["task"].to_numpy() == int(t)
            va_m = df_va["task"].to_numpy() == int(t)
            if int(tr_m.sum()) == 0 or int(va_m.sum()) == 0:
                continue

            p_va = df_va.loc[va_m, [f"p_gt_{k}" for k in range(1, K)]].to_numpy(dtype=float)
            y_va = df_va.loc[va_m, "y_true"].to_numpy(dtype=int)
            init_thr = thr_by_task_base[int(t)]

            # Fold-in calibration (optimistic): calibrate on this fold and evaluate on this fold.
            thr_in = _calibrate_cd(p_gt=p_va, y_true=y_va, init_thr=init_thr, grid=grid, n_iter=int(n_iter))
            y_hat_in = _predict_from_p_gt(p_va, thr_in)
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

            p_tr = df_tr.loc[tr_m, [f"p_gt_{k}" for k in range(1, K)]].to_numpy(dtype=float)
            y_tr = df_tr.loc[tr_m, "y_true"].to_numpy(dtype=int)
            thr_hat = _calibrate_cd(p_gt=p_tr, y_true=y_tr, init_thr=init_thr, grid=grid, n_iter=int(n_iter))
            thr_by_fold[fi, t, :] = thr_hat

            y_hat = _predict_from_p_gt(p_va, thr_hat)
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
        p_gt = df_all.loc[m, [f"p_gt_{k}" for k in range(1, K)]].to_numpy(dtype=float)
        yt = df_all.loc[m, "y_true"].to_numpy(dtype=int)
        yp = _predict_from_p_gt(p_gt, thr_med)
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
    if not oof_path.exists():
        # Fallback: pool from fold-level predictions (OOF by construction).
        if "orig_index" not in df_all.columns:
            raise RuntimeError("Missing oof_predictions.csv and fold val_predictions.csv has no orig_index.")
        oof_df = df_all.sort_values("orig_index").reset_index(drop=True)
    else:
        oof_df = pd.read_csv(oof_path, encoding="utf-8")

    # Add calibrated discrete predictions column.
    oof_df_cal = _apply_thresholds_by_task(df=oof_df, thr_by_task=thr_suggested, n_classes=K, out_col="y_pred_calibrated")
    oof_pred_out = run_dir / "oof_predictions_calibrated.csv"
    oof_df_cal.to_csv(oof_pred_out, index=False, encoding="utf-8")

    # Compute calibrated OOF pooled metrics using the same implementation as training.
    from lumbar_ebm.metrics import compute_ordinal_metrics  # local import to keep CLI fast when apply=False

    if "y_cont" not in oof_df_cal.columns:
        raise ValueError("oof_predictions missing y_cont; cannot compute full metrics.")
    m_cal = compute_ordinal_metrics(
        y_true=oof_df_cal["y_true"].to_numpy(),
        y_pred=oof_df_cal["y_pred_calibrated"].to_numpy(),
        y_cont=oof_df_cal["y_cont"].to_numpy(),
        n_classes=K,
    )
    oof_metrics_out = run_dir / "oof_metrics_calibrated.json"
    oof_metrics_out.write_text(
        json.dumps(
            {
                "method": "calib_cd_median_apply_all",
                "objective": {"primary": "acc", "tie_breakers": ["qwk", "-mae"]},
                "decision_threshold_source": "decision_thresholds_calibrated.json",
                "point_estimate": _metrics_to_dict(m_cal),
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
            n_classes=K,
        )
        by_task_rows.append(
            {
                "task": int(t),
                "task_name": task_names[int(t)],
                "n": int(mask.sum()),
                **_metrics_to_dict(m_t),
            }
        )
    by_task_out = run_dir / "oof_metrics_by_task_calibrated.csv"
    pd.DataFrame(by_task_rows).to_csv(by_task_out, index=False, encoding="utf-8")

    # Persist final thresholds as an explicit artifact (can be treated as part of model weights).
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
                "note": "CORAL decode: y_pred = 1 + sum_k I(p_gt_k > thr_k), k=1..K-1. Only K-1 thresholds are used.",
                "thresholds_by_task": thr_dict,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {out_path}")
    print(f"Wrote: {thr_out}")
    print(f"Wrote: {oof_pred_out}")
    print(f"Wrote: {oof_metrics_out}")
    print(f"Wrote: {by_task_out}")

    summary.update(
        {
            "thresholds_json": str(thr_out),
            "oof_predictions_calibrated": str(oof_pred_out),
            "oof_metrics_calibrated": str(oof_metrics_out),
            "oof_metrics_by_task_calibrated": str(by_task_out),
            "thresholds_by_task": thr_dict,
            "oof_point_estimate_calibrated": _metrics_to_dict(m_cal),
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

