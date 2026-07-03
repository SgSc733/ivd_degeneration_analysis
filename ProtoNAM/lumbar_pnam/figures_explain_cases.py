from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lumbar_pnam.figures_explain_types import FoldContrib


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _orig_index_to_fold(run_dir: Path) -> dict[int, int]:
    m: dict[int, int] = {}
    for fold_dir in sorted([p for p in (run_dir / "checkpoints").glob("fold_*") if p.is_dir()]):
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except Exception:
            continue
        p = fold_dir / "val_predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, encoding="utf-8", usecols=["orig_index"])
        for idx in df["orig_index"].to_numpy(dtype=int).tolist():
            m[int(idx)] = int(fold)
    return m


def _pick_typical_cases(oof_df: pd.DataFrame, *, n_cases: int = 3) -> pd.DataFrame:
    df = oof_df.copy()
    if "y_pred_calibrated" not in df.columns:
        raise ValueError("oof_df missing y_pred_calibrated")
    df["abs_err"] = (df["y_true"].astype(int) - df["y_pred_calibrated"].astype(int)).abs()

    picked: list[pd.Series] = []
    severe = df[(df["y_true"] == df["y_true"].max()) & (df["abs_err"] == 0)]
    if not severe.empty:
        picked.append(severe.sort_values("y_cont", ascending=False).iloc[0])
    mild = df[(df["y_true"] == df["y_true"].min()) & (df["abs_err"] == 0)]
    if not mild.empty:
        picked.append(mild.sort_values("y_cont", ascending=True).iloc[0])
    worst = df.sort_values(["abs_err", "y_cont"], ascending=[False, False])
    if not worst.empty:
        picked.append(worst.iloc[0])

    uniq: dict[int, pd.Series] = {}
    for s in picked:
        uniq[int(s["orig_index"])] = s
    rows = list(uniq.values())[: int(n_cases)]
    if not rows:
        rows = [df.iloc[0]]
    return pd.DataFrame(rows).reset_index(drop=True)


def save_case_contribution_plots(
    *,
    run_dir: str | Path,
    folds: list[FoldContrib],
    tasks: np.ndarray,
    out_dir: Path,
    top_k_features: int = 15,
) -> None:
    """Typical cases: additive decomposition bar plots (task-weighted contributions)."""
    run_dir = Path(run_dir)
    _ensure_dir(out_dir)

    oof_path = run_dir / "oof_predictions_calibrated.csv"
    oof_df = pd.read_csv(oof_path, encoding="utf-8")
    cases = _pick_typical_cases(oof_df, n_cases=3)
    idx_to_fold = _orig_index_to_fold(run_dir)
    fold_map = {fd.fold: fd for fd in folds}

    tasks = np.asarray(tasks).astype(int)
    rows_out: list[dict[str, Any]] = []
    for i_case in range(int(cases.shape[0])):
        row = cases.iloc[i_case]
        orig_idx = int(row["orig_index"])
        fold = int(idx_to_fold.get(orig_idx, folds[0].fold))
        fd = fold_map[int(fold)]
        t = int(tasks[orig_idx])
        feat_names = fd.feature_names

        w = fd.w_last[t, :].astype(float)
        b = float(fd.b_last[t])
        f_mean = fd.f_mean.astype(float)
        f_i = fd.f_raw[orig_idx, :].astype(float)

        baseline = b + float(np.dot(w, f_mean))
        contrib = w * (f_i - f_mean)
        total = baseline + float(contrib.sum())

        order = np.argsort(np.abs(contrib))[::-1]
        top_idx = order[: int(top_k_features)]
        other_sum = float(contrib[order[int(top_k_features) :]].sum()) if order.size > int(top_k_features) else 0.0

        labels = [feat_names[j] for j in top_idx] + ["<others>"]
        vals = [float(contrib[j]) for j in top_idx] + [other_sum]
        colors = ["tab:red" if v > 0 else "tab:blue" for v in vals]

        fig, ax = plt.subplots(figsize=(8, max(4.0, 0.35 * len(labels))))
        y = np.arange(len(labels), dtype=float)
        ax.barh(y, vals, color=colors, alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Contribution to score s (centered; task-weighted)")
        title = (
            f"Case {i_case+1}: fold={fold}, task={t}, id={row.get('case_id_层级', orig_idx)}\n"
            f"y_true={int(row['y_true'])}, y_pred_calib={int(row['y_pred_calibrated'])}, y_cont={float(row['y_cont']):.3f}"
        )
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="x", linewidth=0.3, alpha=0.5)
        ax.text(
            0.02,
            0.98,
            f"baseline={baseline:.3f}\nscore(s)={total:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
        )
        fig.tight_layout()

        case_id = str(row.get("case_id_层级", f"idx{orig_idx}")).replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"case_{i_case+1}_{case_id}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        rows_out.append(
            {
                "case_rank": int(i_case + 1),
                "orig_index": int(orig_idx),
                "fold": int(fold),
                "task": int(t),
                "case_id": case_id,
                "patient_id": row.get("patient_id"),
                "disc_level": row.get("disc_level"),
                "y_true": int(row["y_true"]),
                "y_pred_calibrated": int(row["y_pred_calibrated"]),
                "y_cont": float(row["y_cont"]),
                "baseline": float(baseline),
                "score_s": float(total),
                "top_features": json.dumps(labels, ensure_ascii=False),
                "top_contrib": json.dumps(vals, ensure_ascii=False),
                "plot": str(out_path),
            }
        )

    pd.DataFrame(rows_out).to_csv(out_dir / "case_contribution_summary.csv", index=False, encoding="utf-8")
