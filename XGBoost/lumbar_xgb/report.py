from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from lumbar_xgb.train import CVResult


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        if x is None:
            return "-"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.{nd}f}"
    except Exception:
        pass
    return str(x)


def _dump_json_one_line(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(obj)


def write_report(
    *,
    run_dir: Path,
    cfg: Dict[str, Any],
    cv_res: CVResult,
    data_summary: Dict[str, Any],
    explain_dir: Path | None,
) -> None:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data_cfg = cfg.get("data", {}) or {}
    preprocess_cfg = cfg.get("preprocess", {}) or {}
    weights_cfg = cfg.get("weights", {}) or {}
    xgb_cfg = cfg.get("xgb", {}) or {}
    coral_cfg = cfg.get("coral", {}) or {}
    ord_cfg = cfg.get("ordinal", {}) or {}
    coral_fit_cfg = coral_cfg.get("fit", {}) or {}
    coral_sel_cfg = coral_cfg.get("selection", {}) or {}
    explain_cfg = cfg.get("explain", {}) or {}

    # Load top features if available.
    top_features: list[str] = []
    if explain_dir is not None:
        p = Path(explain_dir) / "global_importance.csv"
        if p.exists():
            try:
                df = pd.read_csv(p, encoding="utf-8")
                top_features = df["feature"].astype(str).head(10).tolist()
            except Exception:
                top_features = []

    # CV table.
    cv_rows = []
    for fr in cv_res.fold_results:
        cv_rows.append(
            {
                "fold": fr.fold,
                "mae": fr.metrics.mae,
                "kappa_q": fr.metrics.kappa_quadratic,
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
    cv_rows.append(
        {
            "fold": "mean",
            "mae": cv_res.mean_metrics.mae,
            "kappa_q": cv_res.mean_metrics.kappa_quadratic,
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
    cv_df = pd.DataFrame(cv_rows)

    lines: list[str] = []
    lines.append("# XGBoost + CORAL 对比方案运行报告")
    lines.append("")
    lines.append(f"- 时间: {now}")
    lines.append(f"- 输出目录: `{run_dir}`")
    lines.append("")

    lines.append("## 1) 数据与配置摘要")
    lines.append("")
    lines.append(f"- CSV: `{data_cfg.get('csv_path')}`")
    lines.append(f"- ID 列: `{data_cfg.get('id_col')}`")
    lines.append(f"- 标签列（CSV 内，如有）: `{data_cfg.get('label_col')}`")
    lines.append(f"- 标签 Excel: `{(cfg.get('labels', {}) or {}).get('xlsx_path')}`")
    lines.append(f"- 样本数(椎间盘): {data_summary.get('n_rows')}")
    lines.append(f"- 特征维度: {data_summary.get('n_features')}")
    lines.append(f"- 病人数: {data_summary.get('n_patients')}")
    lines.append(f"- 节段(task): {_dump_json_one_line(data_summary.get('task_names'))}")
    lines.append(
        f"- 节段差异建模: {'实现A(节段one-hot入模)' if bool(data_cfg.get('add_segment_onehot', False)) else '关闭(仅作task统计)'}"
    )
    lines.append(f"- 统计池化(data.pooling): `{_dump_json_one_line(data_cfg.get('pooling', {}) or {})}`")
    lines.append(
        f"- 预处理: {'开启' if bool(preprocess_cfg.get('enabled', True)) else '关闭'}（RankGauss + ZScore，仅对连续特征；节段one-hot passthrough，仅训练折拟合）"
    )
    lines.append(f"- 权重: {_dump_json_one_line(weights_cfg)}")
    lines.append(f"- XGBoost: eval_metric={xgb_cfg.get('eval_metric')}, early_stopping_rounds={xgb_cfg.get('early_stopping_rounds')}")
    lines.append(f"- XGBoost params: `{_dump_json_one_line((xgb_cfg.get('params', {}) or {}))}`")
    lines.append(
        f"- Ordinal: n_classes={ord_cfg.get('n_classes', coral_cfg.get('n_classes'))}, decision_threshold={_dump_json_one_line(ord_cfg.get('decision_threshold'))}"
    )
    lines.append(f"- CORAL.fit.use_sample_weight: {bool(coral_fit_cfg.get('use_sample_weight', False))}")
    if bool(coral_sel_cfg.get("enabled", False)):
        lines.append(
            "- t* 选择（4.3.3.3）: enabled=true, metric="
            f"{coral_sel_cfg.get('metric')}, strategy={_dump_json_one_line((coral_sel_cfg.get('candidates', {}) or {}).get('strategy'))}"
        )
    lines.append(f"- Explain: {_dump_json_one_line(explain_cfg)}")
    lines.append("")

    lines.append("## 2) 交叉验证结果（按病人分组 GroupKFold）")
    lines.append("")
    lines.append(f"- best_fold（按 kappa_q 最大）: {cv_res.best_fold}")
    lines.append("")
    lines.append("```text")
    lines.append(cv_df.to_string(index=False))
    lines.append("```")
    lines.append("")

    if top_features:
        lines.append("## 3) 全局重要特征（TreeSHAP mean(|phi|) Top10）")
        lines.append("")
        for i, f in enumerate(top_features, start=1):
            lines.append(f"{i}. `{f}`")
        lines.append("")

    lines.append("## 4) 输出文件怎么读（建议顺序）")
    lines.append("")
    lines.append("1. **先看总体是否稳定**：`cv_metrics.csv` / 上表，关注各折波动（是否有某一折明显崩掉）。")
    lines.append("2. **定位错在什么等级**：进入 best_fold 的 `checkpoints/fold_k/`，查看：")
    lines.append("   - `confusion_matrix_val_norm.png`：常见错误是否集中在相邻等级（2↔3、3↔4）。")
    lines.append("     - 注意：该图默认使用 `decision_thresholds_calibrated.json` 对离散等级做阈值校准后的输出。")
    lines.append("   - `val_predictions_calibrated.csv`：逐样本核对（可按 `abs(y_true-y_pred_calibrated)` 排序找最差病例/节段）。")
    if bool(coral_sel_cfg.get("enabled", False)):
        lines.append("   - `coral_iteration_sweep.csv`：查看每个候选迭代轮 t 的 CORAL 后指标，并确认最终选择的 `selected=true` 行（即 t*）。")
    lines.append("3. **看连续量化是否合理**：`ycont_scatter_val.png`，理想情况应随 y_true 单调上升且离群点少。")
    lines.append("4. **查节段差异**：优先看 `val_metrics_by_task_calibrated.csv`（阈值校准后离散解码），再参考 `val_metrics_by_task.csv`。")
    lines.append("5. **一致性/稳定性/校准（对齐主方案 ProtoNAM 的评估图示口径）**：`figures/`：")
    lines.append("   - `bland_altman_disc.png` / `bland_altman_patient_mean.png`：连续量化一致性（Bland–Altman + CCC）。")
    lines.append("   - `performance_stability_by_fold_calibrated.png`：跨折性能稳定性（点图 + 95%CI）。")
    lines.append("   - `probability_calibration_reliability.png`：概率校准（reliability diagram + ECE/Brier）。")
    if explain_dir is not None:
        lines.append("6. **解释（单折 TreeSHAP）**：`explain_all/`：")
        lines.append("   - `global_importance.csv/png`：总体重要性排序")
        lines.append("   - `shape_functions/*.png`：每个特征的“类 shape function”（phi vs feature）")
        lines.append("   - `shape_functions_data/*.csv`：对应每张图的分箱均值数据（可直接做统计/写论文）")
        lines.append("   - `shap_contribs.csv`：逐样本贡献明细（用于复核/二次分析）")
        lines.append("7. **解释稳定性/全局概览（跨折）**：`figures/`：")
        lines.append("   - `feature_importance_overview.png`：全局重要性/贡献方差（mean±SD across folds）。")
        lines.append("   - `explain_stability/shape_stability_*.png`：Top 特征的“类 shape function”跨折稳定性（均值±SD 带）。")
    lines.append("")

    lines.append("## 5) 如何根据结果调参（经验菜单）")
    lines.append("")
    lines.append("### A. 过拟合（训练很好、验证差/折间波动大）")
    lines.append("- 降低模型复杂度：`max_depth` ↓，`min_child_weight` ↑，`gamma` ↑")
    lines.append("- 增强正则：`reg_lambda` ↑，必要时 `reg_alpha` ↑")
    lines.append("- 增强随机性：`subsample` ↓，`colsample_bytree` ↓")
    lines.append("- 学习率策略：`eta` ↓，同时 `num_boost_round` ↑（配合 early stopping）")
    lines.append("")

    lines.append("### B. 欠拟合（训练/验证都差，y_cont 散点像一条横线）")
    lines.append("- 增大容量：`max_depth` 小幅 ↑ 或降低 `min_child_weight`")
    lines.append("- 提高学习强度：`eta` ↑ 或 `num_boost_round` ↑")
    lines.append("- 检查预处理：若特征分布极端/长尾，保持 `preprocess.enabled=true` 通常更稳")
    lines.append("")

    lines.append("### C. 离散指标差但 Spearman 好（排序对了但阈值不理想）")
    lines.append("- 优先调 CORAL：`coral.fit.lr/epochs/patience`，或切换初始化 `init=quantile`（默认）")
    lines.append("- 若你希望偏“保守/激进”预测，可调 `ordinal.decision_threshold.default`（>0.5 更保守，<0.5 更激进）")
    lines.append("- 或使用训练后阈值标定输出：`decision_thresholds_calibrated.json` + `oof_metrics_calibrated.json`")
    lines.append("")

    lines.append("### D. 某个节段(task)明显崩")
    lines.append("- 先看 `val_metrics_by_task.csv` 确认是哪一节段")
    lines.append("- 选项1：开启 `data.add_segment_onehot=true`（让模型显式看到节段）")
    lines.append("- 选项2：开启 `weights.class_weight` 或调整病人权重策略（避免训练被少数等级/病人主导）")
    if explain_dir is not None:
        lines.append("- 选项3：开启 `explain.by_task=true`，对崩掉的节段单独看 shape 图是否异常（强交互/分布漂移）")
    lines.append("")

    lines.append("## 6) 复现要点（避免信息泄露）")
    lines.append("")
    lines.append("- 交叉验证按 `patient_id` 分组（GroupKFold），同一病人不会同时出现在 train/val。")
    lines.append("- 预处理（RankGauss + ZScore）仅在训练折拟合；若启用节段 one-hot，one-hot passthrough 不参与预处理。")
    lines.append("- CORAL 阈值仅用训练折的 `s_train` 拟合，并保存到 `coral_thresholds.json`。")
    if bool(coral_sel_cfg.get("enabled", False)):
        lines.append("- 最终输出使用 t*（见 `fold_info.json:xgb_selected_iteration` 与 `coral_iteration_sweep.csv`）。")
    lines.append("")

    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
