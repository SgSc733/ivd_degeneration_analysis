from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumbar_splsda.data import attach_pfirrmann_from_xlsx, load_model_input_csv
from lumbar_splsda.metrics import compute_ordinal_metrics
from lumbar_splsda.train import _iter_group_kfold_splits


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _threshold_predict_from_scores(
    *, s_train: np.ndarray, y_train: np.ndarray, s_val: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """
    Ordinal thresholding on a 1D severity score:
      mu_k = mean(s | y=k) on train
      theta_k = (mu_k + mu_{k+1}) / 2
      y_hat = 1 + sum_{k}(s > theta_k)
    """
    y_train = np.asarray(y_train).astype(int)
    s_train = np.asarray(s_train).astype(float)
    s_val = np.asarray(s_val).astype(float)
    classes = np.asarray(classes).astype(int)

    mus = []
    for c in classes.tolist():
        mask = y_train == int(c)
        if int(mask.sum()) == 0:
            mus.append(float("nan"))
        else:
            mus.append(float(np.mean(s_train[mask])))

    # If any class is missing, fall back to using global mean ordering (rare with 5-fold on this dataset).
    if not np.isfinite(mus).all():
        mu_all = float(np.mean(s_train))
        mus = [mu_all + 1e-6 * i for i in range(len(classes))]

    mus = np.asarray(mus, dtype=float)
    thetas = 0.5 * (mus[:-1] + mus[1:])

    # Predict by binning.
    y_pred_idx = np.sum(s_val.reshape(-1, 1) > thetas.reshape(1, -1), axis=1)
    y_pred = classes[np.clip(y_pred_idx, 0, len(classes) - 1)]
    return y_pred.astype(int)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to outputs/run_YYYYMMDD_HHMMSS")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = _load_json(run_dir / "config.json")

    # Load full dataset (same as run.py)
    data_cfg = cfg["data"]
    loaded = load_model_input_csv(
        data_cfg["csv_path"],
        id_col=data_cfg["id_col"],
        label_col=data_cfg.get("label_col"),
        drop_cols=data_cfg.get("drop_cols", []),
        classic_prefix=data_cfg.get("classic_prefix", "classic_"),
        patient_id_from_id_col=bool(data_cfg.get("patient_id_from_id_col", True)),
        patient_id_sep=str(data_cfg.get("patient_id_sep", "_")),
        level_from_id_col=bool(data_cfg.get("level_from_id_col", True)),
        level_sep=str(data_cfg.get("level_sep", "_")),
        segment_levels=list(data_cfg.get("segment_levels", ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"])),
        add_segment_onehot=bool(data_cfg.get("add_segment_onehot", False)),
    )

    if loaded.y is None:
        lab_cfg = cfg.get("labels", {}) or {}
        loaded = attach_pfirrmann_from_xlsx(
            loaded,
            xlsx_path=lab_cfg["xlsx_path"],
            sheet_name=lab_cfg.get("sheet_name"),
            patient_id_col=str(lab_cfg.get("patient_id_col", "序号")),
            level_to_col=dict(lab_cfg["level_to_col"]),
            on_missing=str(lab_cfg.get("on_missing", "error")),
        )
    if loaded.y is None:
        raise RuntimeError("Label missing after loading xlsx.")

    groups = np.asarray(loaded.groups)
    y_all = np.asarray(loaded.y).astype(int)
    classes_global = np.sort(np.unique(y_all)).astype(int)

    cv_cfg = cfg.get("cv", {}) or {}
    n_splits = int(cv_cfg.get("n_splits", 5))
    seed = int((cfg.get("training", {}) or {}).get("seed", 0))

    splits = _iter_group_kfold_splits(groups=groups, n_splits=n_splits, seed=seed)

    y_true_cat: list[np.ndarray] = []
    y_pred_cat: list[np.ndarray] = []
    y_cont_cat: list[np.ndarray] = []

    print("[Axis Threshold Evaluation]")
    print(f"- run_dir: {run_dir}")

    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_dir = run_dir / "checkpoints" / f"fold_{fold}"
        model_path = fold_dir / "model.pkl"
        pre_path = fold_dir / "preprocessor.pkl"
        if not model_path.exists() or not pre_path.exists():
            raise FileNotFoundError(f"Missing model/preprocessor for fold {fold}: {fold_dir}")

        with model_path.open("rb") as f:
            model = pickle.load(f)
        with pre_path.open("rb") as f:
            pre = pickle.load(f)

        X_train_df = loaded.X.iloc[train_idx].reset_index(drop=True)
        X_val_df = loaded.X.iloc[val_idx].reset_index(drop=True)
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]

        # Fold-inner median imputation.
        med = X_train_df.median(numeric_only=True)
        X_train_df = X_train_df.fillna(med)
        X_val_df = X_val_df.fillna(med)

        X_train = pre.transform(X_train_df)
        X_val = pre.transform(X_val_df)

        T_train = model.transform(X_train)
        T_val = model.transform(X_val)

        # Least-squares axis in component space: s = T @ a  (a in R^H)
        a_ls, *_ = np.linalg.lstsq(T_train, y_train.astype(float), rcond=None)
        s_train = T_train @ a_ls.reshape(-1)
        s_val = T_val @ a_ls.reshape(-1)

        y_pred = _threshold_predict_from_scores(
            s_train=s_train, y_train=y_train, s_val=s_val, classes=classes_global
        )

        # Use the raw axis score as a continuous proxy for Spearman (it is on the label scale by construction).
        metrics = compute_ordinal_metrics(y_true=y_val, y_pred=y_pred, y_cont=s_val)
        print(
            f"- fold_{fold}: MAE={metrics.mae:.4f} Kappa(q)={metrics.kappa_quadratic:.4f} "
            f"Spearman={metrics.spearman:.4f} Acc±1={metrics.acc_pm1:.4f}"
        )

        y_true_cat.append(y_val.astype(int))
        y_pred_cat.append(y_pred.astype(int))
        y_cont_cat.append(s_val.astype(float))

    y_true = np.concatenate(y_true_cat)
    y_pred = np.concatenate(y_pred_cat)
    y_cont = np.concatenate(y_cont_cat)
    overall = compute_ordinal_metrics(y_true=y_true, y_pred=y_pred, y_cont=y_cont)
    print(
        "\n[Overall] "
        f"MAE={overall.mae:.4f} Kappa(q)={overall.kappa_quadratic:.4f} "
        f"Spearman={overall.spearman:.4f} Acc±1={overall.acc_pm1:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
