from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumbar_splsda.metrics import OrdinalMetrics, compute_ordinal_metrics
from lumbar_splsda.figures_eval_calibration import (
    save_reliability_diagram,
    save_reliability_diagram_by_class,
    save_reliability_diagram_cumulative,
)
from lumbar_splsda.figures_eval_consistency import lin_ccc, save_bland_altman
from lumbar_splsda.figures_eval_plots import (
    save_abs_error_distribution,
    save_confusion_matrix,
    save_performance_stability_plot,
    save_ycont_scatter,
)


@dataclass(frozen=True)
class CalibratedThresholds:
    """Decision thresholds for CORAL decode, aligned with task indices."""

    task_names: list[str]
    thr_by_task: np.ndarray  # (n_tasks, K-1)
    n_classes: int


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_task_names(run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    schema = json.loads((run_dir / "data_schema.json").read_text(encoding="utf-8"))
    task_names = list(schema.get("task_names") or [])
    if not task_names:
        raise ValueError(f"data_schema.json missing task_names: {run_dir / 'data_schema.json'}")
    return task_names


def load_calibrated_thresholds(run_dir: str | Path) -> CalibratedThresholds:
    run_dir = Path(run_dir)
    thr_path = run_dir / "decision_thresholds_calibrated.json"
    if not thr_path.exists():
        raise FileNotFoundError(f"Missing calibrated thresholds: {thr_path}")
    j = json.loads(thr_path.read_text(encoding="utf-8"))
    K = int(j.get("n_classes") or 0)
    if K <= 1:
        raise ValueError(f"Invalid n_classes in {thr_path}: {j.get('n_classes')!r}")
    task_names = load_task_names(run_dir)
    thr_by_name = dict(j.get("thresholds_by_task") or {})
    thr_rows: list[list[float]] = []
    for name in task_names:
        v = thr_by_name.get(str(name))
        if v is None:
            raise ValueError(f"thresholds_by_task missing key {name!r} in {thr_path}")
        vv = [float(x) for x in list(v)]
        if len(vv) < (K - 1):
            raise ValueError(f"threshold list too short for task {name!r}: len={len(vv)} < {K-1}")
        thr_rows.append(vv[: K - 1])
    thr_by_task = np.asarray(thr_rows, dtype=float)
    return CalibratedThresholds(task_names=task_names, thr_by_task=thr_by_task, n_classes=K)


def apply_calibrated_thresholds(
    *,
    df: pd.DataFrame,
    thr: CalibratedThresholds,
    out_col: str = "y_pred_calibrated",
) -> pd.DataFrame:
    """Decode p_gt_1..p_gt_{K-1} using calibrated per-task thresholds."""
    K = int(thr.n_classes)
    need_cols = ["task"] + [f"p_gt_{k}" for k in range(1, K)]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"apply_calibrated_thresholds missing required columns: {miss}")

    out = df.copy()
    task = pd.to_numeric(out["task"], errors="raise").astype(int).to_numpy()
    p_gt = out[[f"p_gt_{k}" for k in range(1, K)]].to_numpy(dtype=float)
    thr_mat = thr.thr_by_task[task]  # (n, K-1)
    y_hat = 1 + (p_gt > thr_mat).sum(axis=1).astype(int)
    out[out_col] = y_hat.astype(int)
    return out


def compute_fold_metrics_calibrated(
    *,
    run_dir: Path,
    thr: CalibratedThresholds,
    fold_dirs: list[Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_dir in fold_dirs:
        p = fold_dir / "val_predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, encoding="utf-8")
        df_cal = apply_calibrated_thresholds(df=df, thr=thr, out_col="y_pred_calibrated")
        m: OrdinalMetrics = compute_ordinal_metrics(
            y_true=df_cal["y_true"].to_numpy(),
            y_pred=df_cal["y_pred_calibrated"].to_numpy(),
            y_cont=df_cal["y_cont"].to_numpy(),
            n_classes=int(thr.n_classes),
        )
        rows.append(
            {
                "fold": int(fold_dir.name.split("_")[-1]),
                "n": int(df_cal.shape[0]),
                "mae": float(m.mae),
                "kappa_quadratic": float(m.kappa_quadratic),
                "ccc": float(m.ccc),
                "spearman": float(m.spearman),
                "acc": float(m.acc),
                "acc_pm1": float(m.acc_pm1),
            }
        )
    out = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    return out


def generate_eval_figures(*, run_dir: str | Path) -> dict[str, Any]:
    """Generate evaluation figures based on calibrated discrete grades."""
    run_dir = Path(run_dir)
    thr = load_calibrated_thresholds(run_dir)
    K = int(thr.n_classes)

    fig_dir = run_dir / "figures"
    _ensure_dir(fig_dir)

    oof_path = run_dir / "oof_predictions_calibrated.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing calibrated OOF predictions: {oof_path}")
    oof_df = pd.read_csv(oof_path, encoding="utf-8")
    if "y_pred_calibrated" not in oof_df.columns:
        raise ValueError(f"oof_predictions_calibrated missing y_pred_calibrated: {oof_path}")

    # Overwrite key "headline" figures using calibrated predicted grade.
    save_confusion_matrix(
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred_calibrated"].to_numpy(),
        n_classes=K,
        out_path=run_dir / "confusion_matrix_oof.png",
        normalize=False,
        title="OOF pooled confusion matrix (calibrated thresholds)",
    )
    save_confusion_matrix(
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred_calibrated"].to_numpy(),
        n_classes=K,
        out_path=run_dir / "confusion_matrix_oof_norm.png",
        normalize=True,
        title="OOF pooled confusion matrix (row-normalized, calibrated thresholds)",
    )
    save_ycont_scatter(
        y_true=oof_df["y_true"].to_numpy(),
        y_cont=oof_df["y_cont"].to_numpy(),
        y_pred=oof_df["y_pred_calibrated"].to_numpy(),
        out_path=run_dir / "ycont_scatter_oof.png",
        title="OOF pooled continuous output vs true (styled by calibrated grade error)",
    )

    # Fold-level figures (based on calibrated per-task thresholds).
    fold_dirs = sorted([p for p in (run_dir / "checkpoints").glob("fold_*") if p.is_dir()])
    for fold_dir in fold_dirs:
        p = fold_dir / "val_predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, encoding="utf-8")
        df_cal = apply_calibrated_thresholds(df=df, thr=thr, out_col="y_pred_calibrated")
        save_confusion_matrix(
            y_true=df_cal["y_true"].to_numpy(),
            y_pred=df_cal["y_pred_calibrated"].to_numpy(),
            n_classes=K,
            out_path=fold_dir / "confusion_matrix_val.png",
            normalize=False,
            title=f"Fold {fold_dir.name} val confusion matrix (calibrated thresholds)",
        )
        save_confusion_matrix(
            y_true=df_cal["y_true"].to_numpy(),
            y_pred=df_cal["y_pred_calibrated"].to_numpy(),
            n_classes=K,
            out_path=fold_dir / "confusion_matrix_val_norm.png",
            normalize=True,
            title=f"Fold {fold_dir.name} val confusion matrix (row-normalized, calibrated thresholds)",
        )
        save_ycont_scatter(
            y_true=df_cal["y_true"].to_numpy(),
            y_cont=df_cal["y_cont"].to_numpy(),
            y_pred=df_cal["y_pred_calibrated"].to_numpy(),
            out_path=fold_dir / "ycont_scatter_val.png",
            title=f"Fold {fold_dir.name} val continuous output vs true (styled by calibrated grade error)",
        )

    # Consistency plots: disc-level and patient-level.
    consistency: dict[str, Any] = {}
    consistency["disc"] = save_bland_altman(
        y_true=oof_df["y_true"].to_numpy(),
        y_cont=oof_df["y_cont"].to_numpy(),
        out_path=fig_dir / "bland_altman_disc.png",
        title="Bland–Altman (disc-level): y_cont vs y_true",
    )
    if "patient_id" in oof_df.columns:
        g = oof_df.groupby("patient_id", sort=False)
        y_true_p = g["y_true"].mean().to_numpy(dtype=float)
        y_cont_p = g["y_cont"].mean().to_numpy(dtype=float)
        consistency["patient_mean"] = save_bland_altman(
            y_true=y_true_p,
            y_cont=y_cont_p,
            out_path=fig_dir / "bland_altman_patient_mean.png",
            title="Bland–Altman (patient-mean): mean(y_cont) vs mean(y_true)",
        )
        consistency["patient_mean"]["ccc"] = float(lin_ccc(y_true=y_true_p, y_cont=y_cont_p))
        consistency["patient_mean"]["n_patients"] = float(y_true_p.size)

    (fig_dir / "consistency_summary.json").write_text(
        json.dumps(consistency, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Performance stability (calibrated discrete predictions per fold).
    by_fold = compute_fold_metrics_calibrated(run_dir=run_dir, thr=thr, fold_dirs=fold_dirs)
    by_fold.to_csv(fig_dir / "cv_metrics_by_fold_calibrated.csv", index=False, encoding="utf-8")
    if not by_fold.empty:
        save_performance_stability_plot(
            by_fold=by_fold,
            out_path=fig_dir / "performance_stability_by_fold_calibrated.png",
            metrics=["mae", "kappa_quadratic", "ccc", "spearman", "acc"],
            title="Performance stability across folds (calibrated thresholds)",
        )

    # |y_true - y_pred| distribution (OOF).
    save_abs_error_distribution(
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred_calibrated"].to_numpy(),
        n_classes=K,
        out_path=fig_dir / "abs_error_distance_oof_calibrated.png",
        title="OOF pooled |y_true - y_pred| distance distribution (calibrated)",
    )

    # Probability calibration / reliability.
    prob_cols = [f"p_cls_{k}" for k in range(1, K + 1)]
    miss_prob = [c for c in prob_cols if c not in oof_df.columns]
    calib_summary: dict[str, Any] = {}
    if not miss_prob:
        probs = oof_df[prob_cols].to_numpy(dtype=float)
        calib_summary = save_reliability_diagram(
            y_true=oof_df["y_true"].to_numpy(),
            y_pred=oof_df["y_pred_calibrated"].to_numpy(),
            probs=probs,
            n_classes=K,
            out_path=fig_dir / "probability_calibration_reliability.png",
            title="Probability calibration (based on predicted calibrated grade)",
            n_bins=10,
        )
        (fig_dir / "probability_calibration_summary.json").write_text(
            json.dumps(calib_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Optional: per-class and cumulative-boundary calibration curves (more directly answers "each grade prob / threshold").
        try:
            by_class = save_reliability_diagram_by_class(
                y_true=oof_df["y_true"].to_numpy(),
                probs=probs,
                n_classes=K,
                out_path=fig_dir / "probability_calibration_by_class.png",
                title="Probability calibration by class (OOF)",
                n_bins=10,
            )
            (fig_dir / "probability_calibration_by_class.json").write_text(
                json.dumps(by_class, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        pgt_cols = [f"p_gt_{k}" for k in range(1, K)]
        if all(c in oof_df.columns for c in pgt_cols):
            try:
                by_bound = save_reliability_diagram_cumulative(
                    y_true=oof_df["y_true"].to_numpy(),
                    p_gt=oof_df[pgt_cols].to_numpy(dtype=float),
                    n_classes=K,
                    out_path=fig_dir / "probability_calibration_cumulative.png",
                    title="Probability calibration of cumulative CORAL outputs P(y>k) (OOF)",
                    n_bins=10,
                )
                (fig_dir / "probability_calibration_cumulative.json").write_text(
                    json.dumps(by_bound, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

    return {
        "n_classes": int(K),
        "task_names": list(thr.task_names),
        "fig_dir": str(fig_dir),
        "consistency": consistency,
        "calibration": calib_summary,
    }

