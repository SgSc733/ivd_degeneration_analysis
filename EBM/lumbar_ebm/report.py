from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lumbar_ebm.calibration import parse_decision_threshold_default


def write_report(
    *,
    out_path: Path,
    cfg: dict[str, Any],
    cv_metrics: pd.DataFrame,
    best_fold: int,
    feature_importance: pd.DataFrame | None,
) -> None:
    """Write a Chinese, step-by-step report to help interpret outputs and tune params."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ebm_cfg = cfg.get("ebm", {}) or {}
    ordinal_cfg = cfg.get("ordinal", {}) or {}
    calib_cfg = cfg.get("calibration", {}) or {}
    explain_cfg = cfg.get("explain", {}) or {}
    explain_enabled = bool(explain_cfg.get("enabled", True))
    decision_rule = str(ordinal_cfg.get("decision_rule", "threshold"))
    decision_threshold_default = parse_decision_threshold_default(
        cfg=cfg,
        n_classes=int(ordinal_cfg.get("n_classes", 5)),
    )

    lines: list[str] = []
    lines.append("# EBM 对比方案运行报告（InterpretML EBM + CORAL 校准）")
    lines.append("")
    lines.append("## 1. 你已经得到的核心输出文件（按优先级）")
    lines.append("")
    lines.append("- `cv_metrics.csv`：每折 + `mean_unweighted`（各折简单平均）+ `mean`（拼接所有验证集后计算，等价于按样本数加权）")
    lines.append(f"- `checkpoints/fold_{best_fold}/val_predictions.csv`：best fold 的逐样本预测（含 s_raw / y_cont / 概率）")
    lines.append(f"- `checkpoints/fold_{best_fold}/fold_info.json`：best fold 的阈值 theta 与 alpha（以及该折指标）")
    lines.append("- `confusion_matrix_oof.png`、`confusion_matrix_oof_norm.png`：OOF pooled 混淆矩阵（使用阈值校准后的离散等级）")
    lines.append("- `ycont_scatter_oof.png`：连续输出 y_cont vs y_true（按阈值校准后的等级误差着色）")
    lines.append("- `figures/`：一致性/稳定性/概率校准/解释稳定性/典型案例解释等图（均基于阈值校准后的离散等级）")
    if explain_enabled:
        lines.append("- `explain_all/term_main/*.png`：每个主效应项（shape function）")
        lines.append("- `explain_all/feature_importance.csv`：全局重要性排序（term importance）")
        lines.append("- `explain_all/segment_thresholds.csv`：节段内阈值化解释（按贡献 g(x) 做三段阈值搜索）")
    lines.append("")

    lines.append("## 2. 如何快速判断模型是否“可用”")
    lines.append("")
    lines.append("建议按顺序查看：")
    lines.append("1) `cv_metrics.csv` 的 `mean` 行（推荐，以样本为单位的汇总）：")
    lines.append("- **Kappa(q)**：作为主指标（越大越好）")
    lines.append("- **MAE**：等级误差（越小越好）")
    lines.append("- **Acc±1**：允许 ±1 档的可接受率（越大越好）")
    lines.append("- **Acc**：严格准确率（越大越好）")
    lines.append("- **BAcc**：Balanced Accuracy（类别不均衡下更稳健，越大越好）")
    lines.append("- **Macro/Weighted-F1**：综合分类质量（越大越好）")
    lines.append("- **Spearman**：连续量化一致性（越大越好）")
    lines.append("- **CCC**：连续量化一致性（补充指标，越大越好）")
    lines.append("2) 若各 fold 波动很大：说明数据量较小或模型不稳定 → 优先做“更平滑/更强 bagging” 的调参。")
    lines.append("")

    lines.append("## 3. 如何根据 `fold_info.json` 调整校准（alpha / theta）")
    lines.append("")
    lines.append("- 校准器形式：`p_gt_k = sigmoid(alpha * (s_raw - theta_k))`（k=1..K-1）")
    lines.append("- 当 `decision_threshold=0.5` 且 `alpha>0` 时：`p_gt_k>0.5 ⇔ s_raw>theta_k`，因此 `theta_k` 可直接视为 **s_raw 轴等级边界**；对应 margin 为 `s_raw-theta_k`。")
    if decision_rule.lower().strip() == "argmax":
        lines.append("- 注意：当前 `decision_rule=argmax`（MAP 解码），离散等级预测不是简单的“超过 0.5 的阈值计数”；但 theta 仍可用于解释各等级边界与 margin。")
    lines.append("- theta 严格递增（由参数化保证），重点关注 **alpha**：")
    lines.append("  - alpha 很小：说明 s_raw 幅度偏大或噪声较大，校准被迫“压扁” → 建议让 EBM 更平滑、更稳健（见第 4 节）")
    lines.append("  - alpha 很大：说明 s_raw 幅度偏小或 bins 太粗，难以形成清晰的阈值 → 可增大 max_bins / max_leaves 或降低 smoothing_rounds")
    lines.append("")

    lines.append("## 4. 如何根据 shape function 调参（让曲线更临床可读）")
    lines.append("")
    lines.append("你主要会调 EBM 这些超参（都在 `config.json` 的 `ebm` 块里）：")
    lines.append("- 曲线太“锯齿/尖峰”（过拟合）：")
    lines.append("  - 降低 `max_bins`（例如 128→64→32）")
    lines.append("  - 增大 `smoothing_rounds`（例如 0→100→200）")
    lines.append("  - 增大 `outer_bags`（例如 4→8→14）")
    lines.append("  - 降低 `max_leaves`（例如 3→2）")
    lines.append("- 曲线几乎全平（欠拟合/过强平滑）：")
    lines.append("  - 增大 `max_bins` / `max_leaves`")
    lines.append("  - 减少 `smoothing_rounds`")
    lines.append("")
    lines.append("此外：默认 `interactions=0`（只做主效应，解释最清晰）。若某个节段明显更差，可考虑只增加少量交互，优先：`disc_level × Top特征`。")
    lines.append("")
    lines.append("提示：在 Windows 上若你把 `ebm.n_jobs` 调到非 1，有些环境可能会触发 joblib 多进程权限错误（如 WinError 5）。遇到这种情况请把 `ebm.n_jobs` 改回 1。")
    lines.append("")

    lines.append("## 5. 如何使用 `segment_thresholds.csv` 写出“阈值化解释句”")
    lines.append("")
    lines.append("该文件每行是 (节段×特征) 的三段阈值搜索结果：")
    lines.append("- `bounds`：两个阈值，把特征分成 3 段")
    lines.append("- `means`：每段贡献均值（贡献=该特征 term 对总分的加性贡献）")
    lines.append("- `vars`：每段贡献方差（越小越稳定）")
    lines.append("- `ns`：每段样本数（过小会导致不稳定，写论文时应回避）")
    lines.append("")
    lines.append("建议筛选标准（经验）：每段 ns 都足够大（例如 >=10），且 means 段间差异明显且方向符合临床认知。")
    lines.append("")

    lines.append("## 6. 本次运行的关键配置摘要（便于复现）")
    lines.append("")
    lines.append(f"- 序数解码: decision_rule={decision_rule}, decision_threshold.default={decision_threshold_default}")
    lines.append(f"- EBM: max_bins={ebm_cfg.get('max_bins')}, smoothing_rounds={ebm_cfg.get('smoothing_rounds')}, "
                 f"outer_bags={ebm_cfg.get('outer_bags')}, max_rounds={ebm_cfg.get('max_rounds')}, "
                 f"max_leaves={ebm_cfg.get('max_leaves')}, interactions={ebm_cfg.get('interactions')}")
    lines.append(f"- 校准: learn_alpha={calib_cfg.get('learn_alpha')}, lr={calib_cfg.get('lr')}, "
                 f"max_epoch={calib_cfg.get('max_epoch')}, patience={calib_cfg.get('patience')}")
    if explain_enabled:
        lines.append(
            f"- 解释: fold={explain_cfg.get('fold')}, dpi={explain_cfg.get('dpi')}, "
            f"segment_n_segments={explain_cfg.get('segment_n_segments')}, lambda_seg={explain_cfg.get('lambda_seg')}"
        )
    lines.append("")

    if feature_importance is not None and not feature_importance.empty:
        lines.append("## 7. Top-10 全局重要性（term importance）")
        lines.append("")
        top = feature_importance.head(10)
        for _, r in top.iterrows():
            lines.append(f"- {r['term']}: {float(r['importance']):.6f}")
        lines.append("")

    lines.append("## 8. 下一步推荐调参路线（从最省事到最有效）")
    lines.append("")
    lines.append("1) 先固定 interactions=0，把曲线调到可读：优先动 `max_bins` 与 `smoothing_rounds`。")
    lines.append("2) 若整体性能不足但曲线已稳定：提高 `max_leaves` 或略增 `max_bins`。")
    lines.append("3) 若某个节段持续最差：考虑开启少量交互（建议只做 `disc_level × Top特征`）。")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
