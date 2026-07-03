from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumbar_splsda.data import attach_pfirrmann_from_xlsx, load_model_input_csv
from lumbar_splsda.metrics import compute_ordinal_metrics
from lumbar_splsda.train import _iter_group_kfold_splits


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to outputs/run_YYYYMMDD_HHMMSS")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = _load_json(run_dir / "config.json")

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

    cv_cfg = cfg.get("cv", {}) or {}
    n_splits = int(cv_cfg.get("n_splits", 5))
    seed = int((cfg.get("training", {}) or {}).get("seed", 0))
    splits = _iter_group_kfold_splits(groups=groups, n_splits=n_splits, seed=seed)

    settings = [
        {"C": 0.1, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 10.0, "class_weight": None},
        {"C": 0.1, "class_weight": "balanced"},
        {"C": 1.0, "class_weight": "balanced"},
        {"C": 10.0, "class_weight": "balanced"},
    ]

    print("[LogReg Postprocess Evaluation]")
    print(f"- run_dir: {run_dir}")

    for s in settings:
        y_true_cat: list[np.ndarray] = []
        y_pred_cat: list[np.ndarray] = []
        y_cont_cat: list[np.ndarray] = []

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

            med = X_train_df.median(numeric_only=True)
            X_train_df = X_train_df.fillna(med)
            X_val_df = X_val_df.fillna(med)

            X_train = pre.transform(X_train_df)
            X_val = pre.transform(X_val_df)

            T_train = model.transform(X_train)
            T_val = model.transform(X_val)

            clf = LogisticRegression(
                C=float(s["C"]),
                class_weight=s["class_weight"],
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=500,
            )
            clf.fit(T_train, y_train)
            proba = clf.predict_proba(T_val)
            classes = clf.classes_.astype(int)

            y_pred = clf.predict(T_val).astype(int)
            # Continuous output as expected label under predicted probabilities.
            y_cont = (proba * classes.reshape(1, -1)).sum(axis=1)

            y_true_cat.append(y_val.astype(int))
            y_pred_cat.append(y_pred.astype(int))
            y_cont_cat.append(y_cont.astype(float))

        y_true = np.concatenate(y_true_cat)
        y_pred = np.concatenate(y_pred_cat)
        y_cont = np.concatenate(y_cont_cat)
        overall = compute_ordinal_metrics(y_true=y_true, y_pred=y_pred, y_cont=y_cont)
        cw = "balanced" if s["class_weight"] == "balanced" else "none"
        print(
            f"- C={s['C']:<4} class_weight={cw:<8} "
            f"MAE={overall.mae:.4f} Kappa(q)={overall.kappa_quadratic:.4f} "
            f"Spearman={overall.spearman:.4f} Acc±1={overall.acc_pm1:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

