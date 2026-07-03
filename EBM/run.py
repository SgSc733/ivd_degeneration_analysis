from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import pandas as pd
import torch  # noqa: F401  # Load torch before sklearn/InterpretML DLLs in the pnam conda env.

from lumbar_ebm.data import (
    DEFAULT_LEVEL_TO_COL,
    DEFAULT_SEGMENT_LEVELS,
    attach_pfirrmann_from_xlsx,
    load_model_input_csv,
)
from lumbar_ebm.explain import (
    build_g_contrib_df,
    explain_global_terms,
    export_feature_importance,
    local_contributions_df,
    segment_threshold_search,
)
from lumbar_ebm.figures import generate_all_figures
from lumbar_ebm.feature_selection import load_feature_selection_dataset
from lumbar_ebm.report import write_report
from lumbar_ebm.train import train_group_kfold
from tools.calibrate_decision_thresholds import calibrate_and_apply


def _load_json(path: Path) -> dict:
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


def _load_final_csv_mode(cfg: dict, *, disc_level_feature_name: str):
    data_cfg = cfg["data"]
    pooling_cfg = data_cfg.get("pooling", {}) or {}
    pooling_feature_types = _normalize_pooling_feature_types(pooling_cfg)

    loaded = load_model_input_csv(
        data_cfg["csv_path"],
        id_col=data_cfg["id_col"],
        label_col=data_cfg.get("label_col"),
        drop_cols=data_cfg.get("drop_cols", []),
        keep_feature_patterns=data_cfg.get("keep_feature_patterns", []),
        classic_prefix=data_cfg.get("classic_prefix", "classic_"),
        patient_id_from_id_col=bool(data_cfg.get("patient_id_from_id_col", True)),
        patient_id_sep=str(data_cfg.get("patient_id_sep", "_")),
        level_from_id_col=bool(data_cfg.get("level_from_id_col", True)),
        level_sep=str(data_cfg.get("level_sep", "_")),
        segment_levels=list(data_cfg.get("segment_levels", DEFAULT_SEGMENT_LEVELS)),
        disc_level_feature_name=disc_level_feature_name,
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
            xlsx_path=xlsx_path,
            sheet_name=lab_cfg.get("sheet_name"),
            patient_id_col=str(lab_cfg.get("patient_id_col", "序号")),
            level_to_col=dict(lab_cfg.get("level_to_col", DEFAULT_LEVEL_TO_COL)),
            on_missing=str(lab_cfg.get("on_missing", "error")),
        )

    print("[Data Summary]")
    print("- mode: final_csv")
    print(f"- rows: {len(loaded.X_ebm)}")
    print(f"- cols (X_num): {loaded.X_num.shape[1]}")
    print(f"- cols (X_ebm): {loaded.X_ebm.shape[1]} (include categorical '{disc_level_feature_name}')")
    print(f"- id_col: {data_cfg['id_col']}")
    print(f"- has_label: {loaded.y is not None}")
    print(f"- tasks (segments): {len(set(loaded.tasks.tolist()))}/{len(loaded.task_names)}")

    schema = {
        "mode": "final_csv",
        "csv_path": str(Path(data_cfg["csv_path"]).resolve()),
        "id_col": data_cfg["id_col"],
        "label_col_csv": data_cfg.get("label_col"),
        "label_xlsx_path": (cfg.get("labels", {}) or {}).get("xlsx_path"),
        "n_rows": int(len(loaded.X_ebm)),
        "feature_names_num": loaded.feature_names_num,
        "feature_names_ebm": loaded.feature_names_ebm,
        "task_names": loaded.task_names,
        "disc_level_feature_name": disc_level_feature_name,
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
    pooling_cfg = data_cfg.get("pooling", {}) or {}
    ebm_cfg = cfg.get("ebm", {}) or {}
    disc_level_feature_name = str(ebm_cfg.get("feature_disc_level_name", "disc_level"))

    data_mode = str(data_cfg.get("mode", "final_csv")).strip().lower()
    feature_cfg = cfg.get("feature_selection", {}) or {}
    feature_selection_enabled = bool(feature_cfg.get("enabled", True))
    if data_mode in {"dual_csv", "fold_inner_feature_selection"} and feature_selection_enabled:
        loaded, data_schema = _load_dual_csv_mode(cfg)
        X_for_train = loaded.X_split
        X_num_for_figures = loaded.X_split
        X_ebm_for_figures = loaded.X_split.copy()
        X_ebm_for_figures[disc_level_feature_name] = loaded.meta["disc_level"].astype(str).to_numpy()
        feature_selection_dataset = loaded
    else:
        loaded, data_schema = _load_final_csv_mode(cfg, disc_level_feature_name=disc_level_feature_name)
        X_for_train = loaded.X_ebm
        X_num_for_figures = loaded.X_num
        X_ebm_for_figures = loaded.X_ebm
        feature_selection_dataset = None

    run_dir = _make_run_dir(cfg["output"]["base_dir"], cfg["output"].get("run_name"))
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    (run_dir / "data_schema.json").write_text(json.dumps(data_schema, ensure_ascii=False, indent=2), encoding="utf-8")

    if loaded.y is None:
        raise ValueError("Label is required but still missing after attempting to load from pfirr_data.xlsx.")

    cv_res = train_group_kfold(
        X_ebm=X_for_train,
        X_num=X_for_train.drop(columns=[disc_level_feature_name], errors="ignore"),
        y=loaded.y,
        tasks=loaded.tasks,
        task_names=loaded.task_names,
        groups=loaded.groups,
        run_dir=run_dir,
        cfg=cfg,
        meta=loaded.meta,
        id_col=data_cfg["id_col"],
        feature_selection_dataset=feature_selection_dataset,
    )

    # Calibrate decision thresholds on pooled OOF (via per-fold val_predictions.csv) and apply the
    # recommended thresholds to compute final discrete grades + metrics without retraining.
    calib = calibrate_and_apply(run_dir=run_dir, apply=True)

    # Post-process figures/stat plots using calibrated discrete grades (and explanation stability).
    if bool((cfg.get("figures", {}) or {}).get("enabled", True)):
        generate_all_figures(
            run_dir=run_dir,
            X_raw=X_num_for_figures,
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
            }
        )
    rows.append(
        {
            "fold": "mean_unweighted",
            "n_train": "-",
            "n_val": "-",
            "mae": cv_res.mean_metrics_unweighted.mae,
            "kappa_quadratic": cv_res.mean_metrics_unweighted.kappa_quadratic,
            "spearman": cv_res.mean_metrics_unweighted.spearman,
            "acc_pm1": cv_res.mean_metrics_unweighted.acc_pm1,
            "acc": cv_res.mean_metrics_unweighted.acc,
            "bacc": cv_res.mean_metrics_unweighted.bacc,
            "macro_f1": cv_res.mean_metrics_unweighted.macro_f1,
            "weighted_f1": cv_res.mean_metrics_unweighted.weighted_f1,
            "ccc": cv_res.mean_metrics_unweighted.ccc,
            "checkpoint_dir": "-",
        }
    )
    rows.append(
        {
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
            }
        )
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(run_dir / "cv_metrics.csv", index=False, encoding="utf-8")

    print("\n[CV Summary]")
    print(f"- best_fold (by kappa): {cv_res.best_fold}")
    print(f"- mean (unweighted) MAE: {cv_res.mean_metrics_unweighted.mae:.4f}")
    print(f"- mean (unweighted) Kappa(q): {cv_res.mean_metrics_unweighted.kappa_quadratic:.4f}")
    print(f"- mean (unweighted) Spearman: {cv_res.mean_metrics_unweighted.spearman:.4f}")
    print(f"- mean (unweighted) Acc±1: {cv_res.mean_metrics_unweighted.acc_pm1:.4f}")
    print(f"- mean (unweighted) Acc: {cv_res.mean_metrics_unweighted.acc:.4f}")
    print(f"- mean (unweighted) BAcc: {cv_res.mean_metrics_unweighted.bacc:.4f}")
    print(f"- mean (unweighted) Macro-F1: {cv_res.mean_metrics_unweighted.macro_f1:.4f}")
    print(f"- mean (unweighted) Weighted-F1: {cv_res.mean_metrics_unweighted.weighted_f1:.4f}")
    print(f"- mean (unweighted) CCC: {cv_res.mean_metrics_unweighted.ccc:.4f}")
    print(f"- mean (all val, weighted) MAE: {cv_res.mean_metrics.mae:.4f}")
    print(f"- mean (all val, weighted) Kappa(q): {cv_res.mean_metrics.kappa_quadratic:.4f}")
    print(f"- mean (all val, weighted) Spearman: {cv_res.mean_metrics.spearman:.4f}")
    print(f"- mean (all val, weighted) Acc±1: {cv_res.mean_metrics.acc_pm1:.4f}")
    print(f"- mean (all val, weighted) Acc: {cv_res.mean_metrics.acc:.4f}")
    print(f"- mean (all val, weighted) BAcc: {cv_res.mean_metrics.bacc:.4f}")
    print(f"- mean (all val, weighted) Macro-F1: {cv_res.mean_metrics.macro_f1:.4f}")
    print(f"- mean (all val, weighted) Weighted-F1: {cv_res.mean_metrics.weighted_f1:.4f}")
    print(f"- mean (all val, weighted) CCC: {cv_res.mean_metrics.ccc:.4f}")
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
        K = int(calib.get("n_classes") or 0)
        n_bounds = max(1, K - 1) if K else None
        print(f"- thresholds_by_task (thr1..thr{n_bounds}):" if n_bounds else "- thresholds_by_task:")
        thr_by_task = dict(calib["thresholds_by_task"])
        for k in loaded.task_names:
            v = thr_by_task.get(str(k))
            if v is None:
                continue
            thr_str = ", ".join([f"{float(x):.3f}" for x in list(v)])
            print(f"  - {k}: [{thr_str}]")

    feature_imp_df = None

    if bool(cfg.get("explain", {}).get("enabled", True)):
        import pickle

        fold_cfg = cfg["explain"].get("fold", "best")
        fold = cv_res.best_fold if fold_cfg == "best" else int(fold_cfg)
        fold_dir = run_dir / "checkpoints" / f"fold_{fold}"
        ebm_path = fold_dir / "ebm.pkl"
        if not ebm_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ebm_path}")
        with open(ebm_path, "rb") as f:
            ebm = pickle.load(f)

        explain_cfg = cfg.get("explain", {}) or {}
        dpi = int(explain_cfg.get("dpi", 300))
        out_dir = run_dir / "explain_all"
        out_dir.mkdir(parents=True, exist_ok=True)

        explain_global_terms(ebm, out_dir=out_dir, dpi=dpi)
        feature_imp_df = export_feature_importance(ebm, out_path=out_dir / "feature_importance.csv")

        # Segment threshold search needs per-sample term contributions. We compute on the full dataset
        # using the selected fold model (best_fold by default).
        raw_input_path = fold_dir / "raw_input_features.pkl"
        X_explain_ebm = pd.read_pickle(raw_input_path) if raw_input_path.exists() else X_ebm_for_figures
        X_explain_num = X_explain_ebm.drop(columns=[disc_level_feature_name], errors="ignore")

        contrib_all = local_contributions_df(ebm, X_ebm=X_explain_ebm, include_intercept=False)

        # Validate that disc_level interactions (if present) are also present in explain_local columns.
        # Otherwise g_{ell,j} would silently degrade to main-effect-only.
        feat_names_in = list(getattr(ebm, "feature_names_in_", []))
        if disc_level_feature_name not in feat_names_in:
            raise ValueError(
                f"disc_level_feature_name '{disc_level_feature_name}' not found in fitted EBM feature_names_in_. "
                "Cannot build g_{ell,j} with segment interactions."
            )
        disc_idx = feat_names_in.index(disc_level_feature_name)
        term_features = list(getattr(ebm, "term_features_", []))
        term_names = list(getattr(ebm, "term_names_", []))
        disc_inter_terms = [
            str(tn)
            for feats, tn in zip(term_features, term_names)
            if isinstance(feats, tuple) and len(feats) == 2 and disc_idx in feats
        ]
        missing_disc_terms = [t for t in disc_inter_terms if t not in contrib_all.columns]
        if missing_disc_terms:
            raise ValueError(
                "Detected disc_level interaction terms in the fitted EBM, but they are missing from explain_local() "
                "contribution columns. This would make g_{ell,j} silently ignore interaction contributions.\n"
                f"- missing terms: {missing_disc_terms}\n"
                f"- available local contribution columns (first 20): {list(contrib_all.columns)[:20]}"
            )

        g_contrib_all, mapping = build_g_contrib_df(
            ebm,
            contrib_df=contrib_all,
            feature_names_num=list(X_explain_num.columns),
            disc_level_feature_name=disc_level_feature_name,
        )
        (out_dir / "interaction_mapping.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # If the config explicitly asks for disc_level interactions, ensure we found a mapping.
        inter_cfg = (cfg.get("ebm", {}) or {}).get("interactions", 0)
        requested_disc_inter = False
        if isinstance(inter_cfg, (list, tuple)):
            for it in inter_cfg:
                if isinstance(it, (list, tuple)):
                    if any(str(x) == disc_level_feature_name for x in it):
                        requested_disc_inter = True
                        break
        if requested_disc_inter and not disc_inter_terms:
            raise ValueError(
                "Config `ebm.interactions` explicitly requested a disc_level interaction, "
                "but the fitted EBM has no disc_level interaction terms."
            )
        if disc_inter_terms and not mapping:
            raise ValueError(
                "Fitted EBM contains disc_level interaction terms, but interaction_mapping is empty. "
                "This indicates an unexpected term-name mismatch."
            )

        # Optionally save the full contribution table (can be large on bigger datasets).
        if bool(explain_cfg.get("save_local_contrib", False)):
            contrib_all.to_csv(out_dir / "local_contrib_all.csv", index=False, encoding="utf-8")
            g_contrib_all.to_csv(out_dir / "g_contrib_all.csv", index=False, encoding="utf-8")

        seg_quantiles = list(explain_cfg.get("segment_quantiles", [0.1, 0.3, 0.5, 0.7, 0.9]))
        seg_n_segments = int(explain_cfg.get("segment_n_segments", 3))
        lambda_seg = float(explain_cfg.get("lambda_seg", 1e-2))
        segment_threshold_search(
            X_num=X_explain_num,
            tasks=loaded.tasks,
            task_names=loaded.task_names,
            contrib_df=g_contrib_all,
            feature_names_num=list(X_explain_num.columns),
            out_path=out_dir / "segment_thresholds.csv",
            segment_quantiles=seg_quantiles,
            segment_n_segments=seg_n_segments,
            lambda_seg=lambda_seg,
        )

        print("\n[Explain]")
        print(f"- all plots saved to: {out_dir}")

    write_report(
        out_path=run_dir / "report.md",
        cfg=cfg,
        cv_metrics=cv_df,
        best_fold=cv_res.best_fold,
        feature_importance=feature_imp_df,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
