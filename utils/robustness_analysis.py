from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy import stats


SEGMENTS: List[str] = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]


@dataclass(frozen=True)
class RobustnessAnalysisResult:

    conditions: List[str]
    n_samples: int
    segments: List[str]

    spearman_rho: Optional[pd.Series]
    spearman_p: Optional[pd.Series]
    spearman_alpha: Optional[float]

    icc: pd.Series
    icc_cluster: pd.Series
    robust_features: List[str]

    final_features: List[str]


def extract_base_case_id(case_id: str) -> Tuple[str, str]:

    if case_id is None or (isinstance(case_id, float) and np.isnan(case_id)):
        return "", "gold"
    s = str(case_id).strip()
    pos = s.find("_")
    if pos < 0:
        return s, "gold"
    base_id = s[:pos]
    perturb = s[pos + 1 :]
    for suffix in ("_image", "_mask"):
        if perturb.endswith(suffix):
            perturb = perturb[: -len(suffix)]
    return base_id, perturb


def _read_feature_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "case_id" not in df.columns:
        raise ValueError("CSV 缺少必需列: case_id")
    df["case_id"] = df["case_id"].astype(str).str.strip()
    return df


def wide_to_disc_long(
    df_wide: pd.DataFrame,
    *,
    segments: Sequence[str] = SEGMENTS,
) -> pd.DataFrame:
    df_wide = df_wide.copy()
    df_wide = df_wide.replace([np.inf, -np.inf], np.nan)
    df_wide = df_wide.set_index("case_id")

    frames: List[pd.DataFrame] = []
    for seg in segments:
        seg_cols = [c for c in df_wide.columns if f"_{seg}_" in c]
        if not seg_cols:
            continue
        seg_df = df_wide[seg_cols].copy()
        seg_df.columns = [c.replace(f"_{seg}_", "_", 1) for c in seg_cols]
        seg_df["segment"] = seg
        seg_df = seg_df.set_index("segment", append=True)
        frames.append(seg_df)

    if not frames:
        raise ValueError("未在特征列名中识别到任何节段（如 _L1-L2_）。")

    long_df = pd.concat(frames, axis=0)
    long_df.index = long_df.index.set_names(["case_id", "segment"])
    long_df.sort_index(inplace=True)
    return long_df


def _load_conditions(
    unperturbed_csv: str | Path,
    perturbed_csv: str | Path,
    *,
    segments: Sequence[str] = SEGMENTS,
) -> Dict[str, pd.DataFrame]:
    un_df = _read_feature_csv(unperturbed_csv)
    gold_long = wide_to_disc_long(un_df, segments=segments)

    pert_df = _read_feature_csv(perturbed_csv)
    base_and_cond = pert_df["case_id"].apply(extract_base_case_id)
    pert_df["base_case_id"] = base_and_cond.apply(lambda x: x[0])
    pert_df["condition"] = base_and_cond.apply(lambda x: x[1])

    conditions: Dict[str, pd.DataFrame] = {"gold": gold_long}

    for cond in sorted([c for c in pert_df["condition"].unique().tolist() if c != "gold"]):
        sub = pert_df.loc[pert_df["condition"] == cond].copy()
        sub = sub.drop(columns=["case_id", "condition"])
        sub["base_case_id"] = sub["base_case_id"].astype(str).str.strip()
        sub = sub.groupby("base_case_id", as_index=False).first()
        sub = sub.rename(columns={"base_case_id": "case_id"})
        sub_long = wide_to_disc_long(sub, segments=segments)
        conditions[cond] = sub_long

    if len(conditions) < 2:
        raise ValueError("扰动后文件中未识别到任何扰动类型（需要至少 1 种扰动）。")

    return conditions


def _align_and_clean_conditions(
    conditions: Dict[str, pd.DataFrame],
    *,
    drop_any_nan_feature: bool = True,
    drop_constant_feature_eps: float = 1e-12,
) -> Dict[str, pd.DataFrame]:
    common_index = None
    common_cols = None
    for df in conditions.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
        common_cols = df.columns if common_cols is None else common_cols.intersection(df.columns)

    if common_index is None or common_cols is None:
        raise ValueError("输入为空，无法分析。")

    aligned = {k: v.loc[common_index, common_cols].copy() for k, v in conditions.items()}

    for df in aligned.values():
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if drop_any_nan_feature:
        bad_cols = pd.Index([])
        for df in aligned.values():
            bad_cols = bad_cols.union(df.columns[df.isna().any(axis=0)])
        if len(bad_cols) > 0:
            for k in list(aligned.keys()):
                aligned[k].drop(columns=bad_cols, inplace=True)

    gold = aligned["gold"]
    variances = gold.var(axis=0, ddof=1)
    const_cols = variances[variances <= drop_constant_feature_eps].index
    if len(const_cols) > 0:
        for k in list(aligned.keys()):
            aligned[k].drop(columns=const_cols, inplace=True)

    if aligned["gold"].shape[1] == 0:
        raise ValueError("清洗后无可用特征（可能存在大量缺失/常量特征）。")

    return aligned


def _roman_to_int(x) -> Optional[int]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
    if s in roman:
        return roman[s]
    try:
        return int(float(s))
    except Exception:
        return None


def _read_pfirrmann_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


def _segment_aliases(seg: str) -> List[str]:
    aliases = {seg, seg.replace("-", ""), seg.replace("-", "/")}
    if "-" in seg:
        left, right = seg.split("-", 1)
        if right.startswith("L"):
            aliases.add(left + right[1:])
        else:
            aliases.add(left + right)
    return [a.upper() for a in aliases]


def load_pfirrmann_grades(
    pfirrmann_csv: str | Path,
    *,
    segments: Sequence[str] = SEGMENTS,
) -> pd.DataFrame:

    df = _read_pfirrmann_table(pfirrmann_csv)
    df.columns = [str(c).strip() for c in df.columns]

    if {"case_id", "segment", "pfirrmann"}.issubset(df.columns):
        df["case_id"] = df["case_id"].astype(str).str.strip()
        df["segment"] = df["segment"].astype(str).str.strip()
        df["pfirrmann"] = df["pfirrmann"].apply(_roman_to_int)
        return df

    if "case_id" not in df.columns:
        for alt in ["序号", "编号", "ID", "id"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "case_id"})
                break

    if "case_id" not in df.columns:
        raise ValueError("Pfirrmann 文件缺少必需列: case_id（或 '序号'）")

    df["case_id"] = df["case_id"].astype(str).str.strip()
    df = df.set_index("case_id")

    seg_col_map: Dict[str, str] = {}
    for seg in segments:
        aliases = _segment_aliases(seg)
        exact = [c for c in df.columns if str(c).strip().upper() in aliases]
        if exact:
            seg_col_map[seg] = exact[0]
            continue
        sub = [c for c in df.columns if any(a in str(c).strip().upper() for a in aliases)]
        if not sub:
            raise ValueError(f"Pfirrmann 文件未找到可匹配节段 '{seg}' 的列（支持: {aliases}）。")
        seg_col_map[seg] = sub[0]

    rows: List[dict] = []
    for case_id, row in df.iterrows():
        for seg in segments:
            rows.append(
                {
                    "case_id": case_id,
                    "segment": seg,
                    "pfirrmann": _roman_to_int(row[seg_col_map[seg]]),
                }
            )
    return pd.DataFrame(rows)


def _build_grade_series(
    grade_long_df: pd.DataFrame,
    sample_index: pd.MultiIndex,
) -> pd.Series:
    if not {"case_id", "segment", "pfirrmann"}.issubset(grade_long_df.columns):
        raise ValueError("grade_long_df 需要包含 case_id/segment/pfirrmann 三列。")
    lut = (
        grade_long_df.dropna(subset=["pfirrmann"])
        .set_index(["case_id", "segment"])["pfirrmann"]
        .astype(float)
    )
    y = lut.reindex(sample_index)
    if y.isna().any():
        missing = int(y.isna().sum())
        raise ValueError(f"Pfirrmann 分级缺失：有 {missing} 个样本未匹配到分级。")
    return y


def _spearman_filter(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    alpha: float,
) -> Tuple[pd.Series, pd.Series, List[str]]:
    rhos: List[float] = []
    ps: List[float] = []
    cols = X.columns.tolist()
    yv = y.to_numpy()
    for c in cols:
        xv = X[c].to_numpy()
        rho, p = stats.spearmanr(xv, yv)
        rhos.append(rho)
        ps.append(p)
    rho_s = pd.Series(rhos, index=cols, name="spearman_rho")
    p_s = pd.Series(ps, index=cols, name="spearman_p")

    keep = rho_s.index[(~rho_s.isna()) & (~p_s.isna()) & (p_s < alpha)].tolist()
    return rho_s, p_s, keep


def compute_icc_21(
    condition_matrices: Sequence[pd.DataFrame],
    *,
    condition_names: Sequence[str],
) -> pd.Series:
    
    if len(condition_matrices) != len(condition_names):
        raise ValueError("condition_matrices 与 condition_names 长度不一致。")

    k = len(condition_matrices)
    if k < 2:
        raise ValueError("ICC(2,1) 需要至少 2 个条件（gold + >=1 扰动）。")

    stacked = np.stack([m.to_numpy(dtype=np.float64) for m in condition_matrices], axis=1)
    n, k2, p = stacked.shape
    if k2 != k:
        raise RuntimeError("内部维度异常。")
    if n < 2:
        raise ValueError("样本数不足（<2），无法计算 ICC。")

    grand_mean = stacked.mean(axis=(0, 1))
    row_means = stacked.mean(axis=1)
    col_means = stacked.mean(axis=0)

    ms_subj = (k / (n - 1.0)) * ((row_means - grand_mean) ** 2).sum(axis=0)
    ms_rater = (n / (k - 1.0)) * ((col_means - grand_mean) ** 2).sum(axis=0)

    resid = stacked - row_means[:, None, :] - col_means[None, :, :] + grand_mean[None, None, :]
    ms_err = (resid**2).sum(axis=(0, 1)) / ((n - 1.0) * (k - 1.0))

    denom = ms_subj + (k - 1.0) * ms_err + (k * (ms_rater - ms_err) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        icc = (ms_subj - ms_err) / denom
    icc = np.where(np.isfinite(icc), icc, np.nan)
    icc = np.clip(icc, -1.0, 1.0)

    return pd.Series(icc, index=condition_matrices[0].columns, name="icc_21")


def cluster_high_icc_features(icc: pd.Series, *, n_clusters: int) -> Tuple[pd.Series, List[str]]:
    icc = icc.dropna()
    if icc.empty:
        raise ValueError("ICC 结果为空，无法聚类。")

    if n_clusters < 2 or n_clusters > len(icc):
        n_clusters = min(max(2, n_clusters), len(icc))

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(icc.to_numpy().reshape(-1, 1))
    clusters = pd.Series(labels, index=icc.index, name="icc_cluster")

    means = clusters.groupby(clusters).apply(lambda s: float(icc.loc[s.index].mean()))
    best = int(means.idxmax())
    keep = clusters.index[clusters == best].tolist()
    return clusters, keep


def remove_redundant_features_by_spearman(
    X: pd.DataFrame,
    *,
    threshold: float,
    remove_lower_variance: bool = True,
) -> List[str]:
    if X.shape[1] == 0:
        return []
    if X.shape[1] == 1:
        return X.columns.tolist()

    corr = X.corr(method="spearman")
    abs_corr = corr.abs().to_numpy()
    iu = np.triu_indices_from(abs_corr, k=1)
    hit = abs_corr[iu] > threshold
    if not np.any(hit):
        return X.columns.tolist()

    i_idx = iu[0][hit]
    j_idx = iu[1][hit]
    cols = X.columns.tolist()
    var = X.var(axis=0, ddof=1)

    to_remove: set[str] = set()
    for i, j in zip(i_idx.tolist(), j_idx.tolist()):
        f1 = cols[i]
        f2 = cols[j]
        if f1 in to_remove or f2 in to_remove:
            continue
        v1 = float(var.loc[f1])
        v2 = float(var.loc[f2])
        if remove_lower_variance:
            to_remove.add(f1 if v1 < v2 else f2)
        else:
            to_remove.add(f1 if v1 > v2 else f2)

    return [c for c in cols if c not in to_remove]


def analyze_robustness(
    *,
    unperturbed_csv: str | Path,
    perturbed_csv: str | Path,
    pfirrmann_csv: Optional[str | Path] = None,
    enable_pfirrmann_filter: bool = True,
    alpha: float = 0.05,
    icc_cluster_count: int = 3,
    dup_corr_threshold: float = 0.99,
    segments: Sequence[str] = SEGMENTS,
) -> RobustnessAnalysisResult:
    conditions_raw = _load_conditions(unperturbed_csv, perturbed_csv, segments=segments)
    conditions = _align_and_clean_conditions(conditions_raw)

    cond_names = ["gold"] + [c for c in conditions.keys() if c != "gold"]
    mats = [conditions[c] for c in cond_names]

    rho_s: Optional[pd.Series] = None
    p_s: Optional[pd.Series] = None
    selected_by_p = conditions["gold"].columns.tolist()

    if enable_pfirrmann_filter:
        if pfirrmann_csv is None:
            raise ValueError("已启用 Pfirrmann 相关性预筛，但未提供 pfirrmann_csv。")
        grade_df = load_pfirrmann_grades(pfirrmann_csv, segments=segments)
        y = _build_grade_series(grade_df, conditions["gold"].index)
        rho_s, p_s, selected_by_p = _spearman_filter(conditions["gold"], y, alpha=alpha)
        if len(selected_by_p) == 0:
            raise ValueError("Spearman 相关性预筛后无特征保留（请检查分级文件或放宽 alpha）。")

        for c in cond_names:
            conditions[c] = conditions[c][selected_by_p]
        mats = [conditions[c] for c in cond_names]

    icc = compute_icc_21(mats, condition_names=cond_names)
    icc_cluster, robust = cluster_high_icc_features(icc, n_clusters=icc_cluster_count)

    gold_X = conditions["gold"]
    robust_X = gold_X[robust]
    final = remove_redundant_features_by_spearman(
        robust_X, threshold=dup_corr_threshold, remove_lower_variance=True
    )

    return RobustnessAnalysisResult(
        conditions=cond_names,
        n_samples=int(gold_X.shape[0]),
        segments=list(segments),
        spearman_rho=rho_s,
        spearman_p=p_s,
        spearman_alpha=(float(alpha) if enable_pfirrmann_filter else None),
        icc=icc,
        icc_cluster=icc_cluster,
        robust_features=robust,
        final_features=final,
    )


def save_analysis_result(
    result: RobustnessAnalysisResult,
    *,
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    final_df = pd.DataFrame({"feature": result.final_features})
    final_df["icc_21"] = [result.icc.get(f, np.nan) for f in result.final_features]
    if result.spearman_rho is not None and result.spearman_p is not None:
        final_df["spearman_rho"] = [result.spearman_rho.get(f, np.nan) for f in result.final_features]
        final_df["spearman_p"] = [result.spearman_p.get(f, np.nan) for f in result.final_features]
    final_df.to_csv(out / "final_robust_features.csv", index=False)

    icc_df = result.icc.reset_index()
    icc_df.columns = ["feature", "icc_21"]
    icc_df["cluster"] = icc_df["feature"].map(result.icc_cluster.to_dict())
    icc_df.to_csv(out / "icc_values.csv", index=False)

    if result.spearman_rho is not None and result.spearman_p is not None:
        sp = pd.concat([result.spearman_rho, result.spearman_p], axis=1).reset_index()
        sp.columns = ["feature", "spearman_rho", "spearman_p"]
        sp.to_csv(out / "spearman_with_pfirrmann.csv", index=False)

    report = out / "analysis_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("椎间盘特征稳健性相关性分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"条件数 k: {len(result.conditions)}\n")
        f.write(f"条件列表: {', '.join(result.conditions)}\n")
        f.write(f"样本数 N_disc: {result.n_samples}\n")
        f.write(f"节段: {', '.join(result.segments)}\n\n")

        if result.spearman_rho is not None and result.spearman_p is not None:
            f.write("步骤1：Pfirrmann 相关性预筛（Spearman）\n")
            alpha = 0.05 if result.spearman_alpha is None else float(result.spearman_alpha)
            alpha_str = f"{alpha:g}"
            keep_n = int(
                ((~result.spearman_rho.isna()) & (~result.spearman_p.isna()) & (result.spearman_p < alpha)).sum()
            )
            f.write(f"  保留特征数: {keep_n} (p<{alpha_str})\n\n")
        else:
            f.write("步骤1：Pfirrmann 相关性预筛（Spearman）\n")
            f.write("  已跳过（未提供分级文件或未启用）\n\n")

        f.write("步骤2：扰动鲁棒性（ICC(2,1) + 分层聚类）\n")
        f.write(f"  ICC 计算特征数: {len(result.icc)}\n")
        f.write(f"  鲁棒簇特征数: {len(result.robust_features)}\n\n")

        f.write("步骤3：去冗余（Spearman 相关阈值）\n")
        f.write(f"  最终特征数: {len(result.final_features)}\n\n")
