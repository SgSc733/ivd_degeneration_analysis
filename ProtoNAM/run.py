from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import pandas as pd

from lumbar_pnam.data import (
    DEFAULT_LEVEL_TO_COL,
    DEFAULT_SEGMENT_LEVELS,
    attach_pfirrmann_from_xlsx,
    load_model_input_csv,
)
from lumbar_pnam.explain import explain_features
from lumbar_pnam.feature_selection import load_feature_selection_dataset
from lumbar_pnam.figures import generate_all_figures
from lumbar_pnam.train import train_group_kfold
from tools.calibrate_decision_thresholds import calibrate_and_apply


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _make_run_dir(base_dir: str, run_name: str | None) -> Path:
    if not run_name:
        run_name = _dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = Path(base_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _normalize_pooling_feature_types(pooling_cfg: dict) -> list[str] | None:
    pooling_feature_types = pooling_cfg.get("feature_types", None)
    if pooling_feature_types is None:
        return None
    if isinstance(pooling_feature_types, str):
        return [pooling_feature_types]
    return list(pooling_feature_types)


def _load_final_csv_mode(cfg: dict) -> tuple[object, dict]:
    data_cfg = cfg["data"]
    pooling_cfg = data_cfg.get("pooling", {}) or {}
    case_meta_cfg = data_cfg.get("case_meta_onehot", {}) or {}
    pooling_feature_types = _normalize_pooling_feature_types(pooling_cfg)

    loaded = load_model_input_csv(
        data_cfg["csv_path"],
        id_col=data_cfg["id_col"],
        label_col=data_cfg.get("label_col"),
        drop_cols=data_cfg.get("drop_cols", []),
        drop_patterns=data_cfg.get("drop_patterns", []),
        feature_order=str(data_cfg.get("feature_order", "csv")),
        classic_prefix=data_cfg.get("classic_prefix", "classic_"),
        patient_id_from_id_col=bool(data_cfg.get("patient_id_from_id_col", True)),
        patient_id_sep=str(data_cfg.get("patient_id_sep", "_")),
        level_from_id_col=bool(data_cfg.get("level_from_id_col", True)),
        level_sep=str(data_cfg.get("level_sep", "_")),
        segment_levels=list(data_cfg.get("segment_levels", DEFAULT_SEGMENT_LEVELS)),
        add_segment_onehot=bool(data_cfg.get("add_segment_onehot", True)),
        pooling_stats=list(pooling_cfg.get("stats", []) or []),
        pooling_feature_types=pooling_feature_types,
        pooling_pyr_prefix=str(pooling_cfg.get("pyr_prefix", "PyRadiomics_")),
        pooling_deep_prefix=str(pooling_cfg.get("deep_prefix", "DeepPCA_")),
        pooling_tensor_prefix=str(pooling_cfg.get("tensor_prefix", "TensorPCA_")),
        pooling_out_prefix=str(pooling_cfg.get("out_prefix", "pool_")),
        add_case_meta_onehot=bool(case_meta_cfg.get("enabled", False)),
        case_meta_csv_path=case_meta_cfg.get("statistics_csv_path"),
        case_meta_case_id_col=str(case_meta_cfg.get("case_id_col", "image_id")),
        case_meta_cols=list(case_meta_cfg.get("columns", []) or []),
        case_meta_onehot_prefix=str(case_meta_cfg.get("prefix", "meta_")),
        case_meta_onehot_drop_first=bool(case_meta_cfg.get("drop_first", False)),
        case_meta_on_missing=str(case_meta_cfg.get("on_missing", "error")),
    )

    if loaded.y is None:
        lab_cfg = cfg.get("labels", {}) or {}
        xlsx_path = lab_cfg.get("xlsx_path")
        if not xlsx_path:
            raise ValueError("Label is required but not found in CSV, and labels.xlsx_path is not set in config.json.")

        loaded = attach_pfirrmann_from_xlsx(
            loaded,
            xlsx_path=xlsx_path,
            sheet_name=lab_cfg.get("sheet_name"),
            patient_id_col=str(lab_cfg.get("patient_id_col", "序号")),
            level_to_col=dict(
                lab_cfg.get(
                    "level_to_col",
                    DEFAULT_LEVEL_TO_COL,
                )
            ),
            on_missing=str(lab_cfg.get("on_missing", "error")),
        )

    print("[Data Summary]")
    print("- mode: final_csv")
    print(f"- rows: {len(loaded.X)}")
    print(f"- cols (X): {loaded.X.shape[1]}")
    print(f"- id_col: {data_cfg['id_col']}")
    print(f"- label_col (csv): {data_cfg.get('label_col')}")
    print(f"- has_label: {loaded.y is not None}")
    print(f"- classic_features: {len(loaded.classic_feature_names)}")
    print(f"- tasks (segments): {len(set(loaded.tasks.tolist()))}/{len(loaded.task_names)}")

    schema = {
        "mode": "final_csv",
        "csv_path": str(Path(data_cfg["csv_path"]).resolve()),
        "id_col": data_cfg["id_col"],
        "label_col_csv": data_cfg.get("label_col"),
        "label_xlsx_path": (cfg.get("labels", {}) or {}).get("xlsx_path"),
        "n_rows": int(len(loaded.X)),
        "feature_names": loaded.feature_names,
        "classic_feature_names": loaded.classic_feature_names,
        "task_names": loaded.task_names,
    }
    return loaded, schema


def _load_dual_csv_mode(cfg: dict):
    data_cfg = cfg["data"]
    lab_cfg = cfg.get("labels", {}) or {}
    pfirr_path = lab_cfg.get("xlsx_path") or data_cfg.get("pfirrmann_csv_path")
    if not pfirr_path:
        raise ValueError("dual_csv 模式需要 labels.xlsx_path 或 data.pfirrmann_csv_path。")

    dataset = load_feature_selection_dataset(
        unperturbed_csv=data_cfg.get("unperturbed_csv_path", "未扰动.csv"),
        perturbed_csv=data_cfg.get("perturbed_csv_path", "扰动后.csv"),
        pfirrmann_csv=pfirr_path,
        segment_levels=list(data_cfg.get("segment_levels", DEFAULT_SEGMENT_LEVELS)),
        id_col=str(data_cfg.get("id_col", "case_id_层级")),
    )

    print("[Data Summary]")
    print("- mode: dual_csv")
    print(f"- rows: {len(dataset.X_split)}")
    print(f"- raw feature cols: {dataset.X_split.shape[1]}")
    print(f"- id_col: {dataset.id_col}")
    print(f"- unperturbed_csv: {data_cfg.get('unperturbed_csv_path', '未扰动.csv')}")
    print(f"- perturbed_csv: {data_cfg.get('perturbed_csv_path', '扰动后.csv')}")
    print(f"- tasks (segments): {len(set(dataset.tasks.tolist()))}/{len(dataset.task_names)}")

    schema = {
        "mode": "dual_csv",
        "unperturbed_csv_path": str(Path(data_cfg.get("unperturbed_csv_path", "未扰动.csv")).resolve()),
        "perturbed_csv_path": str(Path(data_cfg.get("perturbed_csv_path", "扰动后.csv")).resolve()),
        "id_col": dataset.id_col,
        "label_xlsx_path": (cfg.get("labels", {}) or {}).get("xlsx_path"),
        "n_rows": int(len(dataset.X_split)),
        "n_raw_feature_cols": int(dataset.X_split.shape[1]),
        "task_names": dataset.task_names,
    }
    return dataset, schema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_json(cfg_path)

    data_cfg = cfg["data"]
    data_mode = str(data_cfg.get("mode", "final_csv")).strip().lower()
    feature_cfg = cfg.get("feature_selection", {}) or {}
    feature_selection_enabled = bool(feature_cfg.get("enabled", True))

    if data_mode in {"dual_csv", "fold_inner_feature_selection"}:
        if feature_selection_enabled:
            loaded, data_schema = _load_dual_csv_mode(cfg)
            x_for_figures = loaded.X_split
            meta_for_figures = loaded.meta
            tasks_for_figures = loaded.tasks
            task_names_for_figures = loaded.task_names
            cv_res = None
            run_input = {
                "X": loaded.X_split,
                "y": loaded.y,
                "tasks": loaded.tasks,
                "task_names": loaded.task_names,
                "groups": loaded.groups,
                "classic_feature_names": [],
                "meta": loaded.meta,
                "id_col": loaded.id_col,
                "feature_selection_dataset": loaded,
            }
        else:
            loaded, data_schema = _load_final_csv_mode(cfg)
            x_for_figures = loaded.X
            meta_for_figures = loaded.meta
            tasks_for_figures = loaded.tasks
            task_names_for_figures = loaded.task_names
            cv_res = None
            run_input = {
                "X": loaded.X,
                "y": loaded.y,
                "tasks": loaded.tasks,
                "task_names": loaded.task_names,
                "groups": loaded.groups,
                "classic_feature_names": loaded.classic_feature_names,
                "meta": loaded.meta,
                "id_col": data_cfg["id_col"],
                "feature_selection_dataset": None,
            }
    else:
        loaded, data_schema = _load_final_csv_mode(cfg)
        if loaded.y is None:
            raise ValueError("Label is required but still missing after attempting to load from pfirr_data.xlsx.")
        x_for_figures = loaded.X
        meta_for_figures = loaded.meta
        tasks_for_figures = loaded.tasks
        task_names_for_figures = loaded.task_names
        cv_res = None
        run_input = {
            "X": loaded.X,
            "y": loaded.y,
            "tasks": loaded.tasks,
            "task_names": loaded.task_names,
            "groups": loaded.groups,
            "classic_feature_names": loaded.classic_feature_names,
            "meta": loaded.meta,
            "id_col": data_cfg["id_col"],
            "feature_selection_dataset": None,
        }

    run_dir = _make_run_dir(cfg["output"]["base_dir"], cfg["output"].get("run_name"))
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "data_schema.json").write_text(json.dumps(data_schema, ensure_ascii=False, indent=2), encoding="utf-8")

    cv_res = train_group_kfold(
        X=run_input["X"],
        y=run_input["y"],
        tasks=run_input["tasks"],
        task_names=run_input["task_names"],
        groups=run_input["groups"],
        classic_feature_names=run_input["classic_feature_names"],
        run_dir=run_dir,
        cfg=cfg,
        meta=run_input["meta"],
        id_col=run_input["id_col"],
        feature_selection_dataset=run_input["feature_selection_dataset"],
    )

    tcal_cfg = cfg.get("threshold_calibration", {}) or {}
    calib = calibrate_and_apply(
        run_dir=run_dir,
        grid_min=float(tcal_cfg.get("grid_min", 0.25)),
        grid_max=float(tcal_cfg.get("grid_max", 0.75)),
        grid_step=float(tcal_cfg.get("grid_step", 0.005)),
        n_iter=int(tcal_cfg.get("n_iter", 3)),
        objective=tcal_cfg.get("objective", "acc"),
        tie_breakers=tcal_cfg.get("tie_breakers", None),
        apply=True,
    )

    if bool((cfg.get("figures", {}) or {}).get("enabled", True)):
        generate_all_figures(
            run_dir=run_dir,
            X_raw=x_for_figures,
            meta=meta_for_figures,
            tasks=tasks_for_figures,
            task_names=task_names_for_figures,
            cfg=cfg,
        )

    rows = []
    for fr in cv_res.fold_results:
        rows.append(
            {
                "fold": fr.fold,
                "n_train": fr.n_train,
                "n_val": fr.n_val,
                "mae": fr.metrics.mae,
                "kappa_quadratic": fr.metrics.kappa_quadratic,
                "spearman": fr.metrics.spearman,
                "acc_pm1": fr.metrics.acc_pm1,
                "acc": fr.metrics.acc,
                "bacc": fr.metrics.bacc,
                "macro_f1": fr.metrics.macro_f1,
                "weighted_f1": fr.metrics.weighted_f1,
                "ccc": fr.metrics.ccc,
                "checkpoint_dir": str(fr.checkpoint_dir),
            }
        )
    rows.append(
        {
            "fold": "fold_mean",
            "n_train": "-",
            "n_val": "-",
            "mae": cv_res.fold_mean_metrics.mae,
            "kappa_quadratic": cv_res.fold_mean_metrics.kappa_quadratic,
            "spearman": cv_res.fold_mean_metrics.spearman,
            "acc_pm1": cv_res.fold_mean_metrics.acc_pm1,
            "acc": cv_res.fold_mean_metrics.acc,
            "bacc": cv_res.fold_mean_metrics.bacc,
            "macro_f1": cv_res.fold_mean_metrics.macro_f1,
            "weighted_f1": cv_res.fold_mean_metrics.weighted_f1,
            "ccc": cv_res.fold_mean_metrics.ccc,
            "checkpoint_dir": "-",
        }
    )
    mean_row = {
        "fold": "mean",
        "n_train": "-",
        "n_val": "-",
        "mae": cv_res.mean_metrics.mae,
        "kappa_quadratic": cv_res.mean_metrics.kappa_quadratic,
        "spearman": cv_res.mean_metrics.spearman,
        "acc_pm1": cv_res.mean_metrics.acc_pm1,
        "acc": cv_res.mean_metrics.acc,
        "bacc": cv_res.mean_metrics.bacc,
        "macro_f1": cv_res.mean_metrics.macro_f1,
        "weighted_f1": cv_res.mean_metrics.weighted_f1,
        "ccc": cv_res.mean_metrics.ccc,
        "checkpoint_dir": "-",
    }
    if cv_res.oof_bootstrap_ci95:
        for key, (lo, hi) in cv_res.oof_bootstrap_ci95.items():
            mean_row[f"{key}_ci95_low"] = lo
            mean_row[f"{key}_ci95_high"] = hi
        mean_row["bootstrap_n"] = int(cv_res.oof_bootstrap_n)
    rows.append(mean_row)

    if isinstance(calib, dict) and "oof_point_estimate_calibrated" in calib:
        m_cal = dict(calib["oof_point_estimate_calibrated"])
        rows.append(
            {
                "fold": "mean_calibrated",
                "n_train": "-",
                "n_val": "-",
                "mae": m_cal.get("mae"),
                "kappa_quadratic": m_cal.get("kappa_quadratic"),
                "spearman": m_cal.get("spearman"),
                "acc_pm1": m_cal.get("acc_pm1"),
                "acc": m_cal.get("acc"),
                "bacc": m_cal.get("bacc"),
                "macro_f1": m_cal.get("macro_f1"),
                "weighted_f1": m_cal.get("weighted_f1"),
                "ccc": m_cal.get("ccc"),
                "checkpoint_dir": "-",
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "cv_metrics.csv", index=False, encoding="utf-8")

    print("\n[CV Summary]")
    print(f"- best_fold (by kappa): {cv_res.best_fold}")
    print(f"- OOF pooled MAE: {cv_res.oof_metrics.mae:.4f}")
    print(f"- OOF pooled Kappa(q): {cv_res.oof_metrics.kappa_quadratic:.4f}")
    print(f"- OOF pooled Spearman: {cv_res.oof_metrics.spearman:.4f}")
    print(f"- OOF pooled Acc±1: {cv_res.oof_metrics.acc_pm1:.4f}")
    print(f"- OOF pooled Acc: {cv_res.oof_metrics.acc:.4f}")
    print(f"- OOF pooled BAcc: {cv_res.oof_metrics.bacc:.4f}")
    print(f"- OOF pooled Macro-F1: {cv_res.oof_metrics.macro_f1:.4f}")
    print(f"- OOF pooled Weighted-F1: {cv_res.oof_metrics.weighted_f1:.4f}")
    print(f"- OOF pooled CCC: {cv_res.oof_metrics.ccc:.4f}")
    if cv_res.oof_bootstrap_ci95:
        print(f"- bootstrap CI95 (patient-level) saved: {run_dir / 'oof_bootstrap_ci95.json'}")
    print(f"- outputs: {run_dir}")

    if isinstance(calib, dict) and "thresholds_by_task" in calib and "oof_point_estimate_calibrated" in calib:
        m_cal = dict(calib["oof_point_estimate_calibrated"])
        print("\n[Decision Threshold Calibration]")
        print(f"- method: {calib.get('method')}")
        if calib.get("objective") is not None:
            print(f"- objective: {calib.get('objective')}")
        print(f"- thresholds: {calib.get('thresholds_json')}")
        print(f"- OOF pooled MAE (calibrated): {float(m_cal.get('mae', 0.0)):.4f}")
        print(f"- OOF pooled Kappa(q) (calibrated): {float(m_cal.get('kappa_quadratic', 0.0)):.4f}")
        print(f"- OOF pooled Spearman (calibrated): {float(m_cal.get('spearman', 0.0)):.4f}")
        print(f"- OOF pooled Acc±1 (calibrated): {float(m_cal.get('acc_pm1', 0.0)):.4f}")
        print(f"- OOF pooled Acc (calibrated): {float(m_cal.get('acc', 0.0)):.4f}")
        print(f"- OOF pooled BAcc (calibrated): {float(m_cal.get('bacc', 0.0)):.4f}")
        print(f"- OOF pooled Macro-F1 (calibrated): {float(m_cal.get('macro_f1', 0.0)):.4f}")
        print(f"- OOF pooled Weighted-F1 (calibrated): {float(m_cal.get('weighted_f1', 0.0)):.4f}")
        print(f"- OOF pooled CCC (calibrated): {float(m_cal.get('ccc', 0.0)):.4f}")
        n_classes = int(calib.get("n_classes") or 0)
        n_bounds = max(1, n_classes - 1) if n_classes else None
        print(f"- thresholds_by_task (thr1..thr{n_bounds}):" if n_bounds else "- thresholds_by_task:")
        thr_by_task = dict(calib["thresholds_by_task"])
        for task_name in task_names_for_figures:
            values = thr_by_task.get(str(task_name))
            if values is None:
                continue
            thr_str = ", ".join([f"{float(x):.3f}" for x in list(values)])
            print(f"  - {task_name}: [{thr_str}]")

    if bool(cfg.get("explain", {}).get("enabled", True)):
        fold_cfg = cfg["explain"].get("fold", "best")
        fold = cv_res.best_fold if fold_cfg == "best" else int(fold_cfg)
        fold_dir = run_dir / "checkpoints" / f"fold_{fold}"
        out_dir = run_dir / "explain_all"
        explain_features(
            X_raw=x_for_figures,
            meta=meta_for_figures,
            tasks=tasks_for_figures,
            task_names=task_names_for_figures,
            feature_names_to_plot=None,
            fold_dir=fold_dir,
            out_dir=out_dir,
            cfg=cfg,
        )
        print("\n[Explain] all feature plots saved to:")
        print(out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
