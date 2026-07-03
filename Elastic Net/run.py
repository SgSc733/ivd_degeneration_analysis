from __future__ import annotations

import argparse
import datetime as _dt
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

from lumbar_enet.data import attach_pfirrmann_from_xlsx, load_model_input_csv
from lumbar_enet.explain import explain_features
from lumbar_enet.feature_selection import load_feature_selection_dataset
from lumbar_enet.figures import generate_all_figures
from lumbar_enet.report import write_run_report
from lumbar_enet.train import train_group_kfold
from tools.calibrate_decision_thresholds import calibrate_and_apply


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_run_dir(base_dir: str, run_name: str | None) -> Path:
    if not run_name:
        run_name = _dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = Path(base_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_path(cfg_path: Path, p: str) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((cfg_path.parent / pp).resolve())


def _normalize_pooling_feature_types(pooling_cfg: dict) -> list[str] | None:
    pooling_feature_types = pooling_cfg.get("feature_types", None)
    if pooling_feature_types is None:
        return None
    if isinstance(pooling_feature_types, str):
        return [pooling_feature_types]
    return list(pooling_feature_types)


def _configure_stdio() -> None:
    # Best-effort: keep console output readable on Windows terminals / log collectors.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        return


def _fmt_non_ascii(v: object) -> str:
    s = str(v)
    if any(ord(ch) > 127 for ch in s):
        esc = s.encode("unicode_escape").decode("ascii")
        return f"{s} (unicode_escape: {esc})"
    return s


def main() -> int:
    _configure_stdio()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists() and not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent / cfg_path
    cfg_path = cfg_path.resolve()
    cfg = _load_json(cfg_path)
    cfg["_runtime"] = {"config_dir": str(cfg_path.parent)}

    data_cfg = cfg["data"]
    pooling_cfg = data_cfg.get("pooling", {}) or {}
    data_mode = str(data_cfg.get("mode", "final_csv")).strip().lower()
    feature_cfg = cfg.get("feature_selection", {}) or {}
    feature_selection_enabled = bool(feature_cfg.get("enabled", True))

    if data_mode in {"dual_csv", "fold_inner_feature_selection"} and feature_selection_enabled:
        lab_cfg = cfg.get("labels", {}) or {}
        pfirr_path = lab_cfg.get("xlsx_path") or data_cfg.get("pfirrmann_csv_path")
        if not pfirr_path:
            raise ValueError("dual_csv 模式需要 labels.xlsx_path 或 data.pfirrmann_csv_path。")
        loaded = load_feature_selection_dataset(
            unperturbed_csv=_resolve_path(cfg_path, data_cfg.get("unperturbed_csv_path", "未扰动.csv")),
            perturbed_csv=_resolve_path(cfg_path, data_cfg.get("perturbed_csv_path", "扰动后.csv")),
            pfirrmann_csv=_resolve_path(cfg_path, pfirr_path),
            segment_levels=list(data_cfg.get("segment_levels", ["L3-L4", "L4-L5", "L5-S1"])),
            id_col=str(data_cfg.get("id_col", "case_id_层级")),
        )
        feature_selection_dataset = loaded
        x_for_figures = loaded.X_split.copy()
        x_for_figures[str((cfg.get("model", {}) or {}).get("segment_col", "disc_level"))] = loaded.meta["disc_level"].astype(str)
        data_schema = {
            "mode": "dual_csv",
            "unperturbed_csv_path": str(Path(_resolve_path(cfg_path, data_cfg.get("unperturbed_csv_path", "未扰动.csv"))).resolve()),
            "perturbed_csv_path": str(Path(_resolve_path(cfg_path, data_cfg.get("perturbed_csv_path", "扰动后.csv"))).resolve()),
            "id_col": loaded.id_col,
            "label_xlsx_path": pfirr_path,
            "n_rows": int(len(loaded.X_split)),
            "n_raw_feature_cols": int(loaded.X_split.shape[1]),
            "task_names": loaded.task_names,
        }
        print("[Data Summary]")
        print("- mode: dual_csv")
        print(f"- rows: {len(loaded.X_split)}")
        print(f"- raw feature cols: {loaded.X_split.shape[1]}")
        print(f"- id_col: {_fmt_non_ascii(loaded.id_col)}")
        print(f"- tasks (segments): {len(set(loaded.tasks.tolist()))}/{len(loaded.task_names)}")
    else:
        csv_path = _resolve_path(cfg_path, data_cfg["csv_path"])
        pooling_feature_types = _normalize_pooling_feature_types(pooling_cfg)
        loaded = load_model_input_csv(
            csv_path,
            id_col=data_cfg["id_col"],
            label_col=data_cfg.get("label_col"),
            drop_cols=data_cfg.get("drop_cols", []),
        keep_feature_patterns=data_cfg.get("keep_feature_patterns", []),
            encoding=str(data_cfg.get("encoding", "utf-8")),
            classic_prefix=str(data_cfg.get("classic_prefix", "classic_")),
            patient_id_from_id_col=bool(data_cfg.get("patient_id_from_id_col", True)),
            patient_id_sep=str(data_cfg.get("patient_id_sep", "_")),
            level_from_id_col=bool(data_cfg.get("level_from_id_col", True)),
            level_sep=str(data_cfg.get("level_sep", "_")),
            segment_levels=list(data_cfg.get("segment_levels", ["L3-L4", "L4-L5", "L5-S1"])),
            pooling_stats=list(pooling_cfg.get("stats", []) or []),
            pooling_feature_types=pooling_feature_types,
            pooling_pyr_prefix=str(pooling_cfg.get("pyr_prefix", "PyRadiomics_")),
            pooling_deep_prefix=str(pooling_cfg.get("deep_prefix", "DeepPCA_")),
            pooling_tensor_prefix=str(pooling_cfg.get("tensor_prefix", "TensorPCA_")),
            pooling_out_prefix=str(pooling_cfg.get("out_prefix", "pool_")),
        )

        if loaded.y is None:
            lab_cfg = cfg.get("labels", {}) or {}
            xlsx_path = lab_cfg.get("xlsx_path")
            if not xlsx_path:
                raise ValueError("Label is required but not found in CSV, and labels.xlsx_path is not set in config.json.")

            loaded = attach_pfirrmann_from_xlsx(
                loaded,
                xlsx_path=_resolve_path(cfg_path, xlsx_path),
                sheet_name=lab_cfg.get("sheet_name"),
                patient_id_col=str(lab_cfg.get("patient_id_col", "序号")),
                level_to_col=dict(lab_cfg.get("level_to_col", {"L3-L4": "L34", "L4-L5": "L45", "L5-S1": "L5S1"})),
                on_missing=str(lab_cfg.get("on_missing", "error")),
            )
        feature_selection_dataset = None
        segment_col = str((cfg.get("model", {}) or {}).get("segment_col", "disc_level"))
        x_for_figures = loaded.X.copy()
        x_for_figures[segment_col] = loaded.meta["disc_level"].astype(str)
        data_schema = {
            "mode": "final_csv",
            "csv_path": str(Path(csv_path).resolve()),
            "id_col": data_cfg["id_col"],
            "label_col_csv": data_cfg.get("label_col"),
            "label_xlsx_path": (cfg.get("labels", {}) or {}).get("xlsx_path"),
            "n_rows": int(len(loaded.X)),
            "feature_names": loaded.feature_names,
            "task_names": loaded.task_names,
        }
        print("[Data Summary]")
        print("- mode: final_csv")
        print(f"- rows: {len(loaded.X)}")
        print(f"- cols (X): {loaded.X.shape[1]}")
        print(f"- id_col: {_fmt_non_ascii(data_cfg['id_col'])}")
        print(f"- label_col (csv): {data_cfg.get('label_col')}")
        print(f"- has_label: {loaded.y is not None}")
        print(f"- tasks (segments): {len(set(loaded.tasks.tolist()))}/{len(loaded.task_names)}")

    run_dir = _make_run_dir(_resolve_path(cfg_path, cfg["output"]["base_dir"]), cfg["output"].get("run_name"))
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # Always save a minimal data schema snapshot for traceability.
    (run_dir / "data_schema.json").write_text(
        json.dumps(
            {
                **data_schema,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if loaded.y is None:
        raise ValueError("Label is required but still missing after attempting to load from pfirr_data.xlsx.")

    segment_col = str((cfg.get("model", {}) or {}).get("segment_col", "disc_level"))
    X_model = x_for_figures

    cv_res = train_group_kfold(
        X_raw=X_model,
        y=loaded.y,
        tasks=loaded.tasks,
        task_names=loaded.task_names,
        groups=loaded.groups,
        run_dir=run_dir,
        cfg=cfg,
        meta=loaded.meta,
        id_col=str(data_cfg.get("id_col", "case_id_层级")),
        feature_selection_dataset=feature_selection_dataset,
    )

    # Calibrate decision thresholds on pooled OOF and apply the suggested (median across folds)
    # thresholds to compute calibrated OOF metrics (aligned with ProtoNAM workflow).
    calib = calibrate_and_apply(run_dir=run_dir, apply=True)

    # Post-process figures/stat plots using calibrated discrete grades (and explanation stability).
    if bool((cfg.get("figures", {}) or {}).get("enabled", True)):
        generate_all_figures(
            run_dir=run_dir,
            X_raw=X_model,
            meta=loaded.meta,
            tasks=loaded.tasks,
            task_names=loaded.task_names,
            cfg=cfg,
        )

    # Save CV metrics.
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
                "best_params": json.dumps(fr.best_params, ensure_ascii=False),
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
            "best_params": "-",
        }
    )
    rows.append(
        {
            "fold": "mean",
            "n_train": "-",
            "n_val": "-",
            "mae": cv_res.oof_metrics.mae,
            "kappa_quadratic": cv_res.oof_metrics.kappa_quadratic,
            "spearman": cv_res.oof_metrics.spearman,
            "acc_pm1": cv_res.oof_metrics.acc_pm1,
            "acc": cv_res.oof_metrics.acc,
            "bacc": cv_res.oof_metrics.bacc,
            "macro_f1": cv_res.oof_metrics.macro_f1,
            "weighted_f1": cv_res.oof_metrics.weighted_f1,
            "ccc": cv_res.oof_metrics.ccc,
            "checkpoint_dir": "-",
            "best_params": "-",
        }
    )
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
                "best_params": "-",
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "cv_metrics.csv", index=False, encoding="utf-8")

    print("\n[CV Summary]")
    print(f"- best_fold (by kappa): {cv_res.best_fold}")
    print(f"- OOF pooled MAE: {cv_res.oof_metrics.mae:.4f}")
    print(f"- OOF pooled Kappa(q): {cv_res.oof_metrics.kappa_quadratic:.4f}")
    print(f"- OOF pooled Spearman: {cv_res.oof_metrics.spearman:.4f}")
    # Use ASCII "+/-" to avoid console encoding issues with "±".
    print(f"- OOF pooled Acc+/-1: {cv_res.oof_metrics.acc_pm1:.4f}")
    print(f"- OOF pooled Acc: {cv_res.oof_metrics.acc:.4f}")
    print(f"- OOF pooled BAcc: {cv_res.oof_metrics.bacc:.4f}")
    print(f"- OOF pooled Macro-F1: {cv_res.oof_metrics.macro_f1:.4f}")
    print(f"- OOF pooled Weighted-F1: {cv_res.oof_metrics.weighted_f1:.4f}")
    print(f"- OOF pooled CCC: {cv_res.oof_metrics.ccc:.4f}")
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
        print(f"- OOF pooled Acc+/-1 (calibrated): {float(m_cal.get('acc_pm1', 0.0)):.4f}")
        print(f"- OOF pooled Acc (calibrated): {float(m_cal.get('acc', 0.0)):.4f}")
        print(f"- OOF pooled BAcc (calibrated): {float(m_cal.get('bacc', 0.0)):.4f}")
        print(f"- OOF pooled Macro-F1 (calibrated): {float(m_cal.get('macro_f1', 0.0)):.4f}")
        print(f"- OOF pooled Weighted-F1 (calibrated): {float(m_cal.get('weighted_f1', 0.0)):.4f}")
        print(f"- OOF pooled CCC (calibrated): {float(m_cal.get('ccc', 0.0)):.4f}")

        K = int(calib.get("n_classes") or 0)
        n_bounds = max(1, K - 1) if K else None
        print(f"- thresholds_by_task (thr1..thr{n_bounds}):" if n_bounds else "- thresholds_by_task:")
        thr_by_task = dict(calib["thresholds_by_task"])
        for seg in loaded.task_names:
            v = thr_by_task.get(str(seg))
            if v is None:
                continue
            thr_str = ", ".join([f"{float(x):.3f}" for x in list(v)])
            print(f"  - {seg}: [{thr_str}]")

    if bool(cfg.get("explain", {}).get("enabled", True)):
        fold_cfg = cfg["explain"].get("fold", "best")
        fold = cv_res.best_fold if fold_cfg == "best" else int(fold_cfg)
        fold_dir = run_dir / "checkpoints" / f"fold_{fold}"
        model_path = fold_dir / "best_model.pkl"
        raw_input_path = fold_dir / "raw_input_features.pkl"
        with open(model_path, "rb") as f:
            est_for_explain = pickle.load(f)
        feature_names_to_plot = list(getattr(est_for_explain.pre, "base_feature_names_", []) or [])
        if not feature_names_to_plot and hasattr(loaded, "feature_names"):
            feature_names_to_plot = list(getattr(loaded, "feature_names"))

        explain_X = X_model
        if raw_input_path.exists():
            explain_X = pd.read_pickle(raw_input_path)

        out_dir = run_dir / "explain"
        if feature_names_to_plot:
            explain_features(
                X_raw=explain_X,
                segment_col=segment_col,
                segment_levels=list(data_cfg.get("segment_levels", loaded.task_names)),
                segment_reference=str(data_cfg.get("segment_reference", "L3-L4")),
                feature_names_to_plot=feature_names_to_plot,
                fold_dir=fold_dir,
                out_dir=out_dir,
                cfg=cfg,
            )
            print("\n[Explain] shape function plots saved to:")
            print(out_dir)

    if bool((cfg.get("report", {}) or {}).get("enabled", True)):
        write_run_report(run_dir=run_dir, cfg=cfg)
        print("\n[Report] saved to:")
        print(run_dir / "report.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
