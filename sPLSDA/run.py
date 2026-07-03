from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import sys

import pandas as pd

from lumbar_splsda.data import attach_pfirrmann_from_xlsx, load_model_input_csv
from lumbar_splsda.explain import explain_shape_functions
from lumbar_splsda.feature_selection import load_feature_selection_dataset
from lumbar_splsda.figures import generate_all_figures
from lumbar_splsda.train import train_group_kfold
from tools.calibrate_decision_thresholds import calibrate_and_apply


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_run_dir(base_dir: str, run_name: str | None) -> Path:
    if not run_name:
        run_name = _dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = Path(base_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _setup_utf8_stdio() -> None:
    # Make console output readable on Windows when printing Chinese / symbols.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _write_run_readme(run_dir: Path, cfg: dict) -> None:
    txt = f"""# sPLS-DA 对比方案（本次运行输出说明）

本目录包含一次完整的 sPLS-DA（含 segment-aligned 后处理）训练/验证/解释输出。
本实现默认按 `sPLSDA.md 4.3.3.3` 执行 **嵌套交叉验证调参**：
外层 GroupKFold 评估；内层 GroupKFold 选择 `psi_fit=(H, keepX)`，并在固定主体模型下选择
`psi_post=(version, dist, alpha)`。

## 1 如何快速看结果（推荐顺序）

1. `cv_metrics.csv`：先看整体指标（含 `mean` 与 `overall`；优先看 `overall`）。
2. `oof_predictions.csv`：全量 pooled OOF 预测（按 `orig_index` 排序，便于阈值校准与误差分析）。
3. `checkpoints/fold_*/val_predictions.csv`：每折验证集 OOF 预测（含 `p_gt_*`、`p_cls_*`）。
4. `decision_thresholds_calibrated.json` / `oof_predictions_calibrated.csv`：阈值校准产物与校准后的离散预测列 `y_pred_calibrated`。
5. `checkpoints/fold_*/preds_val.csv`：看每条样本的预测、连续量化 y_cont、各类概率/距离（便于与旧脚本兼容）。
6. `checkpoints/fold_*/confusion_val.png`：看混淆矩阵，定位“易混等级”。
6b. `confusion_matrix_oof.png` / `confusion_matrix_oof_norm.png`：**基于阈值校准后的预测等级**的 pooled OOF 混淆矩阵（与主方案对齐）。
6c. `ycont_scatter_oof.png`：连续量化散点图（按校准后误差着色）。
6d. `figures/`：连续量化一致性（Bland–Altman）、性能稳定性、概率校准、解释稳定性、全局重要性概览等对比图（与主方案对齐）。
7. `checkpoints/fold_*/inner_cv_grid.csv`：看该外层折内层调参网格得分（J_in）。
8. `checkpoints/fold_*/selected_hyperparams.json`：看该外层折最终选中的 (H, keepX, version, dist, alpha)。
9. `feature_selection_frequency.csv`：看跨折稳定被选中的特征（越接近 1 越稳定）。
10. `beta_axis_stability.csv`：看“退变轴（severity axis）”线性贡献 beta 的均值/方差（用于解释）。
11. `explain_all/shape_functions/`：每个特征的线性 shape function 图（beta * x）。

## 2 如何根据结果调参（手动经验）

- `tuning.keepX_candidates`（每成分保留特征数候选集合）：
  - 过拟合（外层 CV 指标波动大、feature_selection_frequency 分散）→ 减小 keepX。
  - 欠拟合（所有指标都很差、confusion 接近随机）→ 适度增大 keepX 或增加成分数。
- `tuning.H_candidates`：
  - 默认建议 1~2。H 太大容易在小样本下不稳定。
- `tuning.dist_candidates`：
  - `centroids.dist`：最稳健的默认。
  - `mahalanobis.dist`：当成分相关明显时可能更好；若数值不稳可增大 `cov_reg`。
  - `max.dist`：依赖 dummy 得分，可能更“硬”，建议作为备选对比。
- `tuning.alpha_candidates`：
  - 越大 → 概率更尖锐，y_cont 更接近整数；越小 → 更平滑。用于控制连续量化的“平滑度”。
- `tuning.version_candidates`：
  - `segment-aligned`（推荐）：与 ProtoNAM 的 task=节段设定更对齐。
  - `global`：不显式条件化节段，可做公平性/敏感性对比。

配置文件：`config.json`（本次运行副本见本目录下同名文件）。
"""
    (run_dir / "README.md").write_text(txt, encoding="utf-8")


def main() -> int:
    _setup_utf8_stdio()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root_dir / cfg_path
    cfg = _load_json(cfg_path)
    os.chdir(root_dir)

    data_cfg = cfg["data"]
    pooling_cfg = data_cfg.get("pooling", {}) or {}

    pooling_feature_types = pooling_cfg.get("feature_types", None)
    if pooling_feature_types is None:
        pooling_feature_types = None
    elif isinstance(pooling_feature_types, str):
        pooling_feature_types = [pooling_feature_types]
    else:
        pooling_feature_types = list(pooling_feature_types)

    feature_selection_dataset = None
    data_mode = str(data_cfg.get("mode", "single_csv")).strip().lower()
    if data_mode == "dual_csv":
        feature_selection_dataset = load_feature_selection_dataset(
            unperturbed_csv=data_cfg.get("unperturbed_csv_path", "未扰动.csv"),
            perturbed_csv=data_cfg.get("perturbed_csv_path", "扰动后.csv"),
            pfirrmann_csv=(cfg.get("labels", {}) or {}).get("xlsx_path", "pfirr_data.xlsx"),
            segment_levels=list(data_cfg.get("segment_levels", ["L3-L4", "L4-L5", "L5-S1"])),
            id_col=str(data_cfg.get("id_col", "case_id_层级")),
        )
        loaded = feature_selection_dataset
    else:
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
            segment_levels=list(data_cfg.get("segment_levels", ["L3-L4", "L4-L5", "L5-S1"])),
            add_segment_onehot=bool(data_cfg.get("add_segment_onehot", False)),
            pooling_stats=list(pooling_cfg.get("stats", []) or []),
            pooling_feature_types=pooling_feature_types,
            pooling_pyr_prefix=str(pooling_cfg.get("pyr_prefix", "PyRadiomics_")),
            pooling_deep_prefix=str(pooling_cfg.get("deep_prefix", "DeepPCA_")),
            pooling_tensor_prefix=str(pooling_cfg.get("tensor_prefix", "TensorPCA_")),
            pooling_out_prefix=str(pooling_cfg.get("out_prefix", "pool_")),
        )

    if data_mode != "dual_csv" and loaded.y is None:
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
                    {
                        "L3-L4": "L34",
                        "L4-L5": "L45",
                        "L5-S1": "L5S1",
                    },
                )
            ),
            on_missing=str(lab_cfg.get("on_missing", "error")),
        )

    print("[Data Summary]")
    print(f"- rows: {len(loaded.X)}")
    print(f"- cols (X): {loaded.X.shape[1]}")
    print(f"- id_col: {data_cfg['id_col']}")
    print(f"- label_col (csv): {data_cfg.get('label_col')}")
    print(f"- has_label: {loaded.y is not None}")
    print(f"- classic_features: {len(getattr(loaded, 'classic_feature_names', []))}")
    print(f"- tasks (segments): {len(set(loaded.tasks.tolist()))}/{len(loaded.task_names)}")

    run_dir = _make_run_dir(cfg["output"]["base_dir"], cfg["output"].get("run_name"))
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    (run_dir / "data_schema.json").write_text(
        json.dumps(
            {
                "csv_path": str(Path(data_cfg.get("csv_path", "")).resolve()) if data_cfg.get("csv_path") else None,
                "data_mode": data_mode,
                "unperturbed_csv_path": str(Path(data_cfg.get("unperturbed_csv_path", "")).resolve())
                if data_cfg.get("unperturbed_csv_path")
                else None,
                "perturbed_csv_path": str(Path(data_cfg.get("perturbed_csv_path", "")).resolve())
                if data_cfg.get("perturbed_csv_path")
                else None,
                "id_col": data_cfg["id_col"],
                "label_col_csv": data_cfg.get("label_col"),
                "label_xlsx_path": (cfg.get("labels", {}) or {}).get("xlsx_path"),
                "n_rows": int(len(loaded.X)),
                "feature_names": list(loaded.X.columns),
                "classic_feature_names": list(getattr(loaded, "classic_feature_names", [])),
                "task_names": loaded.task_names,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if loaded.y is None:
        raise ValueError("Label is required but still missing after attempting to load from pfirr_data.xlsx.")

    cv_res = train_group_kfold(
        X=loaded.X,
        y=loaded.y,
        tasks=loaded.tasks,
        task_names=loaded.task_names,
        groups=loaded.groups,
        classic_feature_names=list(getattr(loaded, "classic_feature_names", [])),
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
            X_raw=loaded.X,
            meta=loaded.meta,
            tasks=loaded.tasks,
            task_names=loaded.task_names,
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
    rows.append(
        {
            "fold": "overall",
            "n_train": "-",
            "n_val": "-",
            "mae": cv_res.overall_metrics.mae,
            "kappa_quadratic": cv_res.overall_metrics.kappa_quadratic,
            "spearman": cv_res.overall_metrics.spearman,
            "acc_pm1": cv_res.overall_metrics.acc_pm1,
            "acc": cv_res.overall_metrics.acc,
            "bacc": cv_res.overall_metrics.bacc,
            "macro_f1": cv_res.overall_metrics.macro_f1,
            "weighted_f1": cv_res.overall_metrics.weighted_f1,
            "ccc": cv_res.overall_metrics.ccc,
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
    pd.DataFrame(rows).to_csv(run_dir / "cv_metrics.csv", index=False, encoding="utf-8")

    _write_run_readme(run_dir, cfg)

    print("\n[CV Summary]")
    print(f"- best_fold (by kappa): {cv_res.best_fold}")
    print(f"- overall MAE: {cv_res.overall_metrics.mae:.4f}")
    print(f"- overall Kappa(q): {cv_res.overall_metrics.kappa_quadratic:.4f}")
    print(f"- overall Spearman: {cv_res.overall_metrics.spearman:.4f}")
    print(f"- overall Acc+/-1: {cv_res.overall_metrics.acc_pm1:.4f}")
    print(f"- overall Acc: {cv_res.overall_metrics.acc:.4f}")
    print(f"- overall BAcc: {cv_res.overall_metrics.bacc:.4f}")
    print(f"- overall Macro-F1: {cv_res.overall_metrics.macro_f1:.4f}")
    print(f"- overall Weighted-F1: {cv_res.overall_metrics.weighted_f1:.4f}")
    print(f"- overall CCC: {cv_res.overall_metrics.ccc:.4f}")
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
        if m_cal.get("acc") is not None:
            print(f"- OOF pooled Acc (calibrated): {float(m_cal.get('acc', 0.0)):.4f}")

    if bool(cfg.get("explain", {}).get("enabled", True)) and loaded.feature_names:
        fold_cfg = cfg["explain"].get("fold", "best")
        fold = cv_res.best_fold if fold_cfg == "best" else int(fold_cfg)

        fold_dir = run_dir / "checkpoints" / f"fold_{fold}"
        out_dir = run_dir / "explain_all"
        explain_shape_functions(
            X_raw=loaded.X,
            fold_dir=fold_dir,
            out_dir=out_dir,
            cfg=cfg,
        )
        print("\n[Explain] shape function plots saved to:")
        print(out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
