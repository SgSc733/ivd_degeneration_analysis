from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_run_report(*, run_dir: Path, cfg: dict[str, Any]) -> None:
    """
    Write a "how to read the outputs" report to run_dir/report.md.

    This is intentionally verbose because the user wants a step-by-step guide on:
    - how to analyze results
    - how to adjust parameters based on results
    """
    run_dir = Path(run_dir)
    out_path = run_dir / "report.md"

    cv_path = run_dir / "cv_metrics.csv"
    oof_path = run_dir / "oof_predictions.csv"
    oof_cal_path = run_dir / "oof_predictions_calibrated.csv"

    cv_df = pd.read_csv(cv_path) if cv_path.exists() else None
    oof_df = pd.read_csv(oof_path) if oof_path.exists() else None
    oof_cal_df = pd.read_csv(oof_cal_path) if oof_cal_path.exists() else None

    lines: list[str] = []
    lines.append("# Elastic Net 对比方案：训练/验证/解释运行报告")
    lines.append("")
    lines.append("本报告用于解释本次 `run.py` 运行输出的每个文件，以及如何据此调整 `config.json` 中的参数。")
    lines.append("")
    lines.append("## 1. 输出目录结构（run_dir）")
    lines.append("")
    lines.append("- `config.json`：本次运行使用的完整配置快照（可复现实验）。")
    lines.append("- `data_schema.json`：数据路径、列名、样本数等元信息快照。")
    lines.append("- `cv_metrics.csv`：外层 GroupKFold 的每折指标与均值。")
    lines.append("- `oof_predictions.csv`：外层每折验证集的拼接（OOF 预测），可用于画散点图/误差分布/分节段分析。")
    lines.append("- （阈值校准）`tmp_threshold_calibration.csv`：阈值校准明细（baseline / cross-fitted / median across folds）。")
    lines.append("- （阈值校准）`decision_thresholds_calibrated.json`：建议阈值（按节段 task）。")
    lines.append("- （阈值校准）`oof_predictions_calibrated.csv`：应用建议阈值后的 OOF 预测明细。")
    lines.append("- （阈值校准）`oof_metrics_calibrated.json`：校准后的 OOF pooled 指标（point estimate）。")
    lines.append("- （阈值校准）`oof_metrics_by_task_calibrated.csv`：校准后的按节段指标。")
    lines.append("- `confusion_matrix_oof*.png` / `ycont_scatter_oof.png`：基于阈值校准后的 OOF pooled 头图（对齐 ProtoNAM 口径）。")
    lines.append("- `figures/`：对齐主方案的评估+解释图示（连续量化一致性、性能稳定性、概率校准、解释稳定性、全局重要性/贡献概览、典型病例贡献分解等）。")
    lines.append("- `coefficients_all_folds.csv`：所有外层折的系数表拼接（用于稳定性统计与可视化）。")
    lines.append("- `coefficients_stability.csv`：跨外层折的系数稳定性摘要（mean/std/非零频率），用于筛选“稳定关键特征”。")
    lines.append("- `checkpoints/fold_k/`：每折的模型与解释性产物。")
    lines.append("  - `best_model.pkl`：本工作区的 `FittedElasticNet`（包含折内预处理 + 系数），支持 `.predict(DataFrame)`。")
    lines.append("  - `coefficients.csv`：该折的设计矩阵列系数（含每列的 penalty_factor=v_j 与 penalized 标记）。")
    lines.append("  - `val_predictions.csv`：该折验证集预测明细。")
    lines.append("  - `val_metrics_by_task.csv`：该折按节段（L1-L2...）分组的指标。")
    lines.append("  - `confusion_matrix_val*.png` / `ycont_scatter_val.png`：后处理阶段基于 `decision_thresholds_calibrated.json` 生成的折内可审计图示（对齐 ProtoNAM 口径）。")
    lines.append("  - `inner_cv_summary.csv`：内层 CV 的 (l1_ratio, lambda_min, lambda_1se, 选择策略) 汇总。")
    lines.append("  - `inner_cv_curve.csv`：内层 CV 的 lambda 路径曲线（mean±SE），用于核对 1SE 规则。")
    lines.append("- `explain/`：shape function 图示输出（若启用 explain）。")
    lines.append("")

    lines.append("## 2. 如何解读验证指标（cv_metrics.csv）")
    lines.append("")
    lines.append("建议优先看：`kappa_quadratic` 与 `acc_pm1`，其次看 `mae` 与 `spearman`。")
    lines.append("")
    lines.append("- `mae`：等级 MAE（由连续输出 `y_cont` 通过阈值解码得到 `y_pred` 后计算）。越低越好。")
    lines.append("- `kappa_quadratic`：二次加权 Kappa。越高越好，能更敏感地区分“差 1 档”和“差 3 档”。")
    lines.append("- `spearman`：真实等级与连续输出 `y_cont` 的秩相关。越高越好，反映连续量化排序能力。")
    lines.append("- `acc_pm1`：±1 档可接受率。越高越好。")
    lines.append("- `acc`：分类准确率（完全一致率）。越高越好。")
    lines.append("- `bacc`：Balanced Accuracy（各等级召回率的宏平均）。越高越好。")
    lines.append("- `macro_f1`：Macro-F1（各等级 F1 的宏平均）。越高越好。")
    lines.append("- `weighted_f1`：Weighted-F1（按样本数加权的 F1）。越高越好。")
    lines.append("- `ccc`：Lin 的一致性相关系数（y_true vs y_cont）。越高越好。")
    lines.append("")
    if cv_df is not None:
        lines.append("本次运行的外层 CV 均值（来自 `cv_metrics.csv` 的 `mean` 行）：")
        mean_row = cv_df[cv_df["fold"].astype(str) == "mean"]
        if not mean_row.empty:
            r = mean_row.iloc[0].to_dict()
            lines.append(f"- mean MAE: {r.get('mae')}")
            lines.append(f"- mean Kappa(q): {r.get('kappa_quadratic')}")
            lines.append(f"- mean Spearman: {r.get('spearman')}")
            lines.append(f"- mean Acc±1: {r.get('acc_pm1')}")
            lines.append(f"- mean Acc: {r.get('acc')}")
            lines.append(f"- mean BAcc: {r.get('bacc')}")
            lines.append(f"- mean Macro-F1: {r.get('macro_f1')}")
            lines.append(f"- mean Weighted-F1: {r.get('weighted_f1')}")
            lines.append(f"- mean CCC: {r.get('ccc')}")
        lines.append("")

    lines.append("## 3. 如何用 oof_predictions.csv 做误差分析（推荐）")
    lines.append("")
    lines.append("你可以用 `oof_predictions.csv` 做以下分析：")
    lines.append("")
    lines.append("1) 看误差分布：统计 `abs(y_true - y_pred)` 的直方图，判断是否主要集中在 0/1 档。")
    lines.append("2) 看连续输出：画 `y_true` vs `y_cont` 的散点图并拟合回归线，检查是否存在系统性偏差（例如整体偏高/偏低）。")
    lines.append("3) 分节段分析：按 `disc_level` 或 `task_name` 分组计算 MAE/Kappa，定位“哪个节段最难”。")
    lines.append("")
    if oof_df is not None:
        lines.append("OOF 误差快速摘要：")
        err = (oof_df["y_true"].astype(int) - oof_df["y_pred"].astype(int)).abs()
        lines.append(f"- OOF MAE (baseline decode): {float(err.mean()):.4f}")
        lines.append(f"- OOF Acc±1: {float((err <= 1).mean()):.4f}")
        lines.append(f"- OOF Acc: {float((err == 0).mean()):.4f}")
        if oof_cal_df is not None and "y_pred_calibrated" in oof_cal_df.columns:
            err2 = (oof_cal_df["y_true"].astype(int) - oof_cal_df["y_pred_calibrated"].astype(int)).abs()
            lines.append(f"- OOF MAE (calibrated): {float(err2.mean()):.4f}")
            lines.append(f"- OOF Acc±1 (calibrated): {float((err2 <= 1).mean()):.4f}")
            lines.append(f"- OOF Acc (calibrated): {float((err2 == 0).mean()):.4f}")
        lines.append("")

    lines.append("## 4. 如何解读 shape function 图（explain/）")
    lines.append("")
    lines.append("shape function 图的横轴是原始特征值，纵轴是连续输出 `y_cont` 的“中心化贡献”（只改变该特征，其余特征固定为中位数）。")
    lines.append("")
    lines.append("- `explain/shared_reference/`：以参考节段（默认 L3-L4）为基准的曲线。")
    lines.append("- `explain/by_segment/{L1-L2...}/`：若启用了节段交互项，则每个节段各自一套曲线。")
    lines.append("")
    lines.append(
        "提示：因为本实现使用严格 Rank→Phi^{-1} + Z-score（折内拟合），线性模型在“原始尺度”上可能呈现非线性曲线，这是单调预处理映射导致的。"
    )
    lines.append("")

    lines.append("## 4.1 如何用 coefficients_stability.csv 判断解释是否稳定")
    lines.append("")
    lines.append("推荐做法：按 `mean_abs` 和 `nonzero_freq` 排序，优先关注“绝对系数大且跨折经常非零”的特征。")
    lines.append("")
    lines.append(
        "- 若出现大量特征 `nonzero_freq` 很低（只在少数折出现），说明解释不稳定：可优先使用 `search.lambda_choice=lambda_1se`，并考虑增大 `model.interaction_penalty_factor`（更强抑制交互项）。"
    )
    lines.append("- 若稳定特征集中在少量特征且指标不差，说明 Elastic Net 基线解释更可靠，可作为强对照结论的一部分。")
    lines.append("")

    lines.append("## 5. 如何根据结果调参（config.json）")
    lines.append("")
    lines.append("下面给出“可操作”的调参路径（按优先级）：")
    lines.append("")
    lines.append("### 5.1 首先决定是否需要节段交互")
    lines.append("")
    lines.append("- 本实现默认 `model.segment_penalty_factor=0.0`，即节段主效应 γ（截距偏移）不参与惩罚，与方案口径一致。")
    lines.append("- 若 `val_metrics_by_task.csv` 显示不同节段表现差异大，建议保持：`model.include_segment_interactions=true`。")
    lines.append("- 若加入交互后整体 Kappa 下降、各折波动变大，尝试：")
    lines.append("  - 增大 `model.interaction_penalty_factor`（例如 2 -> 4），更强约束交互项；或")
    lines.append("  - 直接设 `model.include_segment_interactions=false` 只保留节段截距。")
    lines.append("")
    lines.append("### 5.2 调整稀疏程度（lambda 路径 / l1_ratio / 1SE 规则）")
    lines.append("")
    lines.append("- `search.l1_ratio` 越接近 1 越像 Lasso（更稀疏），越接近 0 越像 Ridge（更稳定）。")
    lines.append("- `search.lambda_choice=lambda_1se`：更偏向稳定/更稀疏（同 1SE 规则）。")
    lines.append("- `search.lambda_choice=lambda_min`：更偏向追求内层 CV 最优均值（可能更不稳定）。")
    lines.append("- `search.lambda_min_ratio` 与 `search.n_lambdas` 控制路径范围与分辨率。")
    lines.append("")
    lines.append("经验法：")
    lines.append("- 若验证 MAE 高且 Spearman 低：可能欠拟合 → 适当降低正则（例如使用 `lambda_min` 或减小 `interaction_penalty_factor`）。")
    lines.append("- 若验证指标波动大、某些折明显变差：可能过拟合 → 使用 `lambda_1se`，或增大 `interaction_penalty_factor`，或减少交互。")
    lines.append("")
    lines.append("### 5.3 预处理策略（严格 RankGauss + Z-score）")
    lines.append("")
    lines.append("- 本实现固定采用严格 rank→Phi^{-1} + z-score（折内拟合），以对齐方案 4.3.0 的数学定义并避免信息泄露。")
    lines.append("")

    lines.append("### 5.4 收敛性检查（convergence）")
    lines.append("")
    lines.append("- 每折会在 `checkpoints/fold_k/fold_info.json` 里记录 `converged/n_iter`。")
    lines.append("- 若出现 `converged=false`：优先增大 `model.max_iter`，其次适度放宽 `model.tol`；并避免过小的 `lambda`（可用 `lambda_1se`）。")
    lines.append("")

    lines.append("## 6. 复现实验与复用模型")
    lines.append("")
    lines.append("- 复现：直接用保存下来的 `run_dir/config.json` 重新运行即可。")
    lines.append("- 复用模型：加载 `checkpoints/fold_k/best_model.pkl`（`FittedElasticNet`），对同格式 DataFrame 调用 `.predict()`。")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "has_cv_metrics": bool(cv_path.exists()),
        "has_oof_predictions": bool(oof_path.exists()),
        "has_oof_predictions_calibrated": bool(oof_cal_path.exists()),
    }
    (run_dir / "report_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
