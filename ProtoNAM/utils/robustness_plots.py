from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

if TYPE_CHECKING:  # pragma: no cover
    from utils.robustness_analysis import RobustnessAnalysisResult

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _feature_type_token(name: str) -> Optional[str]:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    s = str(name).strip()
    pos = s.find("_")
    if pos <= 0:
        return None
    return s[:pos].strip().lower()


def _save_figure(fig: Figure, path: Path, *, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvas(fig)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")


def _kde_curve(values: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3 or float(np.nanstd(v)) <= 0.0:
        return None
    try:
        kde = stats.gaussian_kde(v)
    except Exception:
        return None
    xs = np.linspace(0.0, 1.0, 256)
    ys = kde(xs)
    return xs, ys


def save_icc_hist_kde(
    icc_values: Sequence[float],
    *,
    icc_threshold: float,
    output_path: str | Path,
    title: str,
    bins: int = 50,
) -> None:
    v = np.asarray(list(icc_values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return

    fig = Figure(figsize=(7.6, 4.6))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.hist(v, bins=int(bins), color="#4c78a8", alpha=0.75, edgecolor="white")

    kde = _kde_curve(v)
    if kde is not None:
        xs, ys = kde
        ax2 = ax.twinx()
        ax2.plot(xs, ys, color="#f58518", linewidth=2.0, alpha=0.95)
        ax2.set_ylabel("KDE density")

    thr = float(icc_threshold)
    if np.isfinite(thr):
        ax.axvline(thr, color="#e45756", linestyle="--", linewidth=2.0, label=f"T_ICC={thr:g}")
        ax.legend(loc="upper left")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("ICC(2,1)")
    ax.set_ylabel("Feature count")
    ax.set_title(title)
    fig.tight_layout()
    _save_figure(fig, Path(output_path))


def save_retained_vs_threshold(
    icc_values: Sequence[float],
    *,
    icc_threshold: float,
    output_path: str | Path,
    title: str,
    step: float = 0.01,
) -> None:
    v = np.asarray(list(icc_values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return

    step = float(step) if float(step) > 0 else 0.01
    n_steps = int(round(1.0 / step)) + 1
    ts = np.linspace(0.0, 1.0, n_steps)
    counts = np.asarray([(v >= t).sum() for t in ts], dtype=np.int64)

    fig = Figure(figsize=(7.6, 4.6))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(ts, counts, color="#4c78a8", linewidth=2.2)
    ax.set_xlabel("ICC threshold")
    ax.set_ylabel("Retained feature count")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)

    thr = float(icc_threshold)
    if np.isfinite(thr):
        ax.axvline(thr, color="#e45756", linestyle="--", linewidth=2.0)
        ax.scatter([thr], [int((v >= thr).sum())], color="#e45756", zorder=5)

    fig.tight_layout()
    _save_figure(fig, Path(output_path))


def _cluster_order_from_corr(corr: pd.DataFrame) -> List[int]:
    n = int(corr.shape[0])
    if n < 3:
        return list(range(n))

    mat = corr.to_numpy(dtype=np.float64, copy=False)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    abs_corr = np.abs(mat)
    np.fill_diagonal(abs_corr, 1.0)

    dist = 1.0 - abs_corr
    dist = np.clip(dist, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    return leaves_list(linkage(condensed, method="average")).tolist()


def save_corr_heatmap(
    corr: Optional[pd.DataFrame],
    *,
    output_path: str | Path,
    title: str,
    reorder: bool = True,
    max_labels: int = 40,
) -> None:
    if corr is None or corr.empty:
        return

    c = corr.copy()
    if reorder:
        order = _cluster_order_from_corr(c)
        c = c.iloc[order, order]

    n = int(c.shape[0])
    base = max(6.0, min(20.0, 0.22 * n))
    fig = Figure(figsize=(base, base))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    mat = c.to_numpy(dtype=np.float64, copy=False)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    im = ax.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title)

    if n <= int(max_labels):
        labels = [str(x) for x in c.columns.tolist()]
        ax.set_xticks(list(range(n)))
        ax.set_yticks(list(range(n)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{n} features (labels hidden)")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
    fig.tight_layout()
    _save_figure(fig, Path(output_path))


def _find_case_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["case_id", "image_id", "source_case_id", "序号", "编号", "ID", "id"]:
        if c in df.columns:
            return c
    return None


def save_pfirrmann_distribution(
    grades_long: pd.DataFrame,
    *,
    segments_order: Sequence[str],
    output_path: str | Path,
) -> None:
    df = grades_long.copy()
    for col in ["case_id", "segment", "pfirrmann"]:
        if col not in df.columns:
            return
    df = df.dropna(subset=["pfirrmann"])
    if df.empty:
        return

    df["pfirrmann"] = pd.to_numeric(df["pfirrmann"], errors="coerce")
    df = df.dropna(subset=["pfirrmann"])
    df["pfirrmann"] = df["pfirrmann"].astype(int)

    grades = [1, 2, 3, 4, 5]
    segs = list(segments_order)
    df = df.loc[df["segment"].isin(segs)]

    overall = df["pfirrmann"].value_counts().reindex(grades, fill_value=0)
    by_seg = (
        df.groupby(["segment", "pfirrmann"]).size().unstack(fill_value=0).reindex(index=segs, columns=grades, fill_value=0)
    )

    colors = {1: "#4c78a8", 2: "#72b7b2", 3: "#f58518", 4: "#e45756", 5: "#54a24b"}

    fig = Figure(figsize=(12.6, 4.8))
    FigureCanvas(fig)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.bar([str(g) for g in grades], overall.to_numpy(), color=[colors[g] for g in grades])
    ax1.set_title("Pfirrmann Grade Distribution (Overall)")
    ax1.set_xlabel("Grade")
    ax1.set_ylabel("Disc count")

    bottoms = np.zeros(len(segs), dtype=np.float64)
    for g in grades:
        vals = by_seg[g].to_numpy(dtype=np.float64, copy=False)
        ax2.bar(segs, vals, bottom=bottoms, color=colors[g], label=str(g))
        bottoms = bottoms + vals

    ax2.set_title("Pfirrmann Grade Distribution (By Segment)")
    ax2.set_xlabel("Segment")
    ax2.set_ylabel("Disc count")
    ax2.legend(title="Grade", fontsize=8, title_fontsize=9, loc="upper right")
    ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    _save_figure(fig, Path(output_path))


def _top_value_counts(s: pd.Series, *, top_n: int = 15) -> pd.Series:
    s2 = s.dropna().astype(str).str.strip()
    if s2.empty:
        return pd.Series(dtype=np.int64)
    vc = s2.value_counts()
    if int(vc.shape[0]) <= int(top_n):
        return vc
    head = vc.head(int(top_n))
    other = int(vc.iloc[int(top_n) :].sum())
    return pd.concat([head, pd.Series({"Other": other})])


def save_scanner_batch_distribution(
    statistics_df: pd.DataFrame,
    *,
    output_path: str | Path,
) -> None:
    df = statistics_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    fig = Figure(figsize=(10.8, 4.6))
    FigureCanvas(fig)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    if "manufacturer" in df.columns:
        vc = _top_value_counts(df["manufacturer"], top_n=12)
        ax1.barh(vc.index[::-1], vc.to_numpy()[::-1], color="#72b7b2")
        ax1.set_title("Manufacturer Distribution")
    else:
        ax1.text(0.5, 0.5, "Missing 'manufacturer' column", ha="center", va="center")
        ax1.set_axis_off()

    if "batch_scanner" in df.columns:
        vc = _top_value_counts(df["batch_scanner"], top_n=12)
        ax2.barh(vc.index[::-1], vc.to_numpy()[::-1], color="#f58518")
        ax2.set_title("Batch/Scanner Distribution")
    else:
        ax2.text(0.5, 0.5, "Missing 'batch_scanner' column", ha="center", va="center")
        ax2.set_axis_off()

    fig.tight_layout()
    _save_figure(fig, Path(output_path))


def save_robustness_figures(
    result: "RobustnessAnalysisResult",
    *,
    output_dir: str | Path,
    grades_long: Optional[pd.DataFrame],
    statistics_df: Optional[pd.DataFrame],
) -> List[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    icc_s = result.icc.dropna()
    rad_feats = [f for f in icc_s.index.tolist() if _feature_type_token(f) == "pyradiomics"]
    if rad_feats:
        icc_v = icc_s.loc[rad_feats].to_numpy(dtype=np.float64, copy=False)
        suffix = "pyradiomics"
        title_suffix = "(PyRadiomics)"
    else:
        icc_v = icc_s.to_numpy(dtype=np.float64, copy=False)
        suffix = "all"
        title_suffix = "(All selected features)"

    p1 = out / f"icc_hist_kde_{suffix}.png"
    save_icc_hist_kde(
        icc_v,
        icc_threshold=float(result.icc_threshold),
        output_path=p1,
        title=f"ICC(2,1) Distribution: Histogram + KDE {title_suffix}",
        bins=50,
    )
    if p1.exists():
        saved.append(p1.name)

    p2 = out / f"icc_retained_vs_threshold_{suffix}.png"
    save_retained_vs_threshold(
        icc_v,
        icc_threshold=float(result.icc_threshold),
        output_path=p2,
        title=f"Retained Features vs ICC Threshold {title_suffix}",
        step=0.01,
    )
    if p2.exists():
        saved.append(p2.name)

    p3 = out / "corr_heatmap_pre_dedup.png"
    save_corr_heatmap(
        result.corr_pre_dedup,
        output_path=p3,
        title=f"Spearman Correlation Matrix (Pre-dedup, top-{len(result.corr_pre_dedup_features)} by score)",
        reorder=True,
        max_labels=35,
    )
    if p3.exists():
        saved.append(p3.name)

    p4 = out / "corr_heatmap_post_dedup.png"
    save_corr_heatmap(
        result.corr_post_dedup,
        output_path=p4,
        title=f"Spearman Correlation Matrix (Post-dedup, top-{len(result.corr_post_dedup_features)} by score)",
        reorder=True,
        max_labels=35,
    )
    if p4.exists():
        saved.append(p4.name)

    if grades_long is not None and statistics_df is not None:
        gdf = grades_long.copy()
        sdf = statistics_df.copy()
        sdf.columns = [str(c).strip() for c in sdf.columns]

        cid_col = _find_case_id_col(sdf)
        if cid_col is not None:
            case_ids = set(sdf[cid_col].astype(str).str.strip().tolist())
            gdf = gdf.loc[gdf["case_id"].astype(str).str.strip().isin(case_ids)].copy()

        p5 = out / "cohort_pfirrmann_distribution.png"
        save_pfirrmann_distribution(gdf, segments_order=result.segments, output_path=p5)
        if p5.exists():
            saved.append(p5.name)

        p6 = out / "cohort_scanner_batch_distribution.png"
        save_scanner_batch_distribution(sdf, output_path=p6)
        if p6.exists():
            saved.append(p6.name)

    return saved


__all__ = [
    "save_corr_heatmap",
    "save_icc_hist_kde",
    "save_pfirrmann_distribution",
    "save_retained_vs_threshold",
    "save_robustness_figures",
    "save_scanner_batch_distribution",
]
