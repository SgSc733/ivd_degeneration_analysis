from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


SEGMENTS: List[str] = ["L3-L4", "L4-L5", "L5-S1"]


@dataclass(frozen=True)
class RobustnessAnalysisResult:

    conditions: List[str]
    n_samples: int
    segments: List[str]

    feature_types: List[str]
    stability_selection_enabled: bool
    force_include_features: List[str]
    force_included_added: List[str]

    n_features_initial: int
    n_features_after_cleaning: int
    dropped_nan_features: List[str]
    dropped_constant_features: List[str]

    pca_enabled: bool
    pca_eta: float
    pca_m_cap: int
    pca_deep_info: Optional[pd.DataFrame]
    pca_tensor_info: Optional[pd.DataFrame]
    deep_feature_prefixes: List[str]
    tensor_feature_prefixes: List[str]

    spearman_rho: pd.Series
    spearman_p: pd.Series
    spearman_q: pd.Series
    spearman_alpha_fdr: float
    spearman_rho_min: float
    selected_by_spearman: List[str]

    icc: pd.Series
    icc_threshold: float
    robust_features: List[str]

    composite_score: pd.Series
    dup_corr_threshold: float

    enet_l1_ratio: float
    lambda_mode: str
    lambda_value: float
    lambda_cv_folds: Optional[int]
    lambda_cv_n_alphas: Optional[int]
    lambda_cv_epsilon: Optional[float]
    lambda_cv_use_1se: Optional[bool]
    lambda_cv_table: Optional[pd.DataFrame]
    lambda_cv_selected: Optional[float]
    lambda_size_tuning_enabled: bool
    lambda_size_tuning_table: Optional[pd.DataFrame]
    bootstrap_B: int
    stability_delta: float
    stability_tau: float
    stable_pi: pd.Series
    stable_mean_abs_beta: pd.Series

    k_max: Optional[int]
    final_features: List[str]
    final_model_input: pd.DataFrame

    pass_dedup: pd.Series
    pass_topk: pd.Series
    dedup_removed_by: pd.Series
    dedup_removed_corr: pd.Series
    dedup_removed_abs_corr: pd.Series
    dedup_removed_order: pd.Series
    dedup_max_abs_corr: pd.Series
    dedup_n_corr_over_threshold: pd.Series
    dedup_decisions: Optional[pd.DataFrame]

    # Small correlation matrices (typically top-N by composite_score) for visualization/reporting.
    corr_pre_dedup_features: List[str]
    corr_post_dedup_features: List[str]
    corr_pre_dedup: Optional[pd.DataFrame]
    corr_post_dedup: Optional[pd.DataFrame]


_ALLOWED_FEATURE_TYPES: Tuple[str, ...] = ("classic", "pyradiomics", "deep", "tensor")


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _feature_type_token(name: str) -> Optional[str]:

    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    s = str(name).strip()
    pos = s.find("_")
    if pos <= 0:
        return None
    return s[:pos].strip().lower()


def _feature_type_category(name: str) -> Optional[str]:

    tok = _feature_type_token(name)
    if tok is None:
        return None
    if tok == "classic":
        return "classic"
    if tok == "pyradiomics":
        return "pyradiomics"
    if tok == "deep":
        return "deep"
    if tok in ("tensor", "tucker"):
        return "tensor"
    return None


def _normalize_feature_types(feature_types: Sequence[str]) -> List[str]:
    selected: set[str] = set()
    for raw in feature_types:
        if raw is None:
            continue
        t = str(raw).strip().lower()
        if not t:
            continue
        if t == "tucker":
            t = "tensor"
        selected.add(t)

    unknown = sorted([t for t in selected if t not in _ALLOWED_FEATURE_TYPES])
    if unknown:
        raise ValueError(f"未知的 feature_types: {unknown}（支持: {list(_ALLOWED_FEATURE_TYPES)}）")

    out = [t for t in _ALLOWED_FEATURE_TYPES if t in selected]
    if not out:
        raise ValueError("feature_types 不能为空（请至少选择一种特征类型）。")
    return out


def _normalize_force_include_features(
    features: Sequence[str],
    *,
    segments: Sequence[str],
) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in features:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue

        for seg in segments:
            token = f"_{seg}_"
            if token in s:
                s = s.replace(token, "_", 1)
                break

        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _pick_feature_groups(
    columns: Sequence[str],
    *,
    deep_prefixes: Sequence[str],
    tensor_prefixes: Sequence[str],
) -> Tuple[List[str], List[str], List[str]]:
    deep_cols: List[str] = []
    tensor_cols: List[str] = []
    other_cols: List[str] = []
    for c in columns:
        cs = str(c)
        if any(cs.startswith(p) for p in deep_prefixes):
            deep_cols.append(cs)
        elif any(cs.startswith(p) for p in tensor_prefixes):
            tensor_cols.append(cs)
        else:
            other_cols.append(cs)
    return deep_cols, tensor_cols, other_cols


def _fit_pca(
    X: pd.DataFrame,
    *,
    eta: float,
    m_cap: int,
) -> Tuple[Optional[StandardScaler], Optional[PCA], Optional[pd.DataFrame]]:
    if X.shape[1] == 0:
        return None, None, None
    n_samples = int(X.shape[0])
    if n_samples < 2:
        raise ValueError("PCA 需要至少 2 个样本。")

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X.to_numpy(dtype=np.float64, copy=False))

    pca_full = PCA(svd_solver="full")
    pca_full.fit(Xs)
    evr = pca_full.explained_variance_ratio_
    if evr is None or len(evr) == 0:
        return scaler, None, None

    cum = np.cumsum(evr)
    m = int(np.searchsorted(cum, float(eta), side="left") + 1)
    m = min(m, int(m_cap), int(n_samples - 1), int(X.shape[1]))
    m = max(m, 1)

    pca = PCA(n_components=m, svd_solver="full")
    pca.fit(Xs)
    info = pd.DataFrame(
        {
            "component": np.arange(1, m + 1, dtype=int),
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(float),
        }
    )
    info["cumulative_explained"] = info["explained_variance_ratio"].cumsum()
    return scaler, pca, info


def _apply_pca(
    df: pd.DataFrame,
    *,
    cols: Sequence[str],
    scaler: StandardScaler,
    pca: PCA,
    out_prefix: str,
) -> pd.DataFrame:
    if len(cols) == 0:
        return df
    X = df.loc[:, list(cols)].to_numpy(dtype=np.float64, copy=False)
    Xs = scaler.transform(X)
    Z = pca.transform(Xs)
    out_cols = [f"{out_prefix}{i:03d}" for i in range(1, Z.shape[1] + 1)]
    out_df = pd.DataFrame(Z, index=df.index, columns=out_cols)
    df2 = df.drop(columns=list(cols))
    return pd.concat([df2, out_df], axis=1)


def _patient_weights(sample_index: pd.MultiIndex) -> pd.Series:
    case_id = pd.Series(sample_index.get_level_values(0).astype(str), index=sample_index, name="case_id")
    counts = case_id.value_counts()
    w = case_id.map(lambda cid: 1.0 / float(counts[cid]))
    return w.astype(np.float64)


def _segment_residualize_xy(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    segments: pd.Series,
    sample_weight: pd.Series,
    train_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    if train_mask is None:
        train_mask = np.ones(len(y), dtype=bool)

    seg_all = segments.astype(str).to_numpy()
    w_all = sample_weight.to_numpy(dtype=np.float64, copy=False)

    Xv = X.to_numpy(dtype=np.float64, copy=False)
    yv = y.to_numpy(dtype=np.float64, copy=False)

    means_y: Dict[str, float] = {}
    means_X: Dict[str, np.ndarray] = {}

    seg_train = seg_all[train_mask]
    for seg in np.unique(seg_train).tolist():
        seg_s = str(seg)
        m = train_mask & (seg_all == seg_s)
        ww = w_all[m]
        denom = float(ww.sum())
        if denom <= 0:
            means_y[seg_s] = float(np.nan)
            means_X[seg_s] = np.full((Xv.shape[1],), np.nan, dtype=np.float64)
            continue
        means_y[seg_s] = float((ww * yv[m]).sum() / denom)
        means_X[seg_s] = (ww.reshape(-1, 1) * Xv[m]).sum(axis=0) / denom

    X_res = Xv.copy()
    y_res = yv.copy()
    for seg_s, mu_y in means_y.items():
        idx = seg_all == seg_s
        if not bool(idx.any()):
            continue
        if np.isfinite(mu_y):
            y_res[idx] = y_res[idx] - float(mu_y)
        mu_x = means_X.get(seg_s)
        if mu_x is not None and np.isfinite(mu_x).all():
            X_res[idx, :] = X_res[idx, :] - mu_x.reshape(1, -1)

    X_df = pd.DataFrame(X_res, index=X.index, columns=X.columns)
    y_s = pd.Series(y_res, index=y.index, name="y_segment_residual")
    return X_df, y_s


def _elasticnet_cv_lambda(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    groups: pd.Series,
    segments: pd.Series,
    sample_weight: pd.Series,
    l1_ratio: float,
    n_folds: int,
    n_alphas: int,
    epsilon: float,
    use_1se: bool,
    random_state: int = 0,
    max_iter: int = 5000,
) -> Tuple[float, pd.DataFrame]:
    if n_folds < 2:
        raise ValueError("K_lambda 必须 >= 2。")
    if n_alphas < 2:
        raise ValueError("lambda 路径长度 L 必须 >= 2。")
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon 必须在 (0, 1) 内。")

    X_res_full, y_res_full = _segment_residualize_xy(
        X,
        y,
        segments=segments,
        sample_weight=sample_weight,
        train_mask=None,
    )

    Xv = X_res_full.to_numpy(dtype=np.float64, copy=False)
    scaler0 = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler0.fit_transform(Xv)

    yv = y_res_full.to_numpy(dtype=np.float64, copy=False)
    wv = sample_weight.to_numpy(dtype=np.float64, copy=False)
    mu = float(np.average(yv, weights=wv)) if float(wv.sum()) > 0 else float(np.nanmean(yv))
    y0 = yv - mu

    denom = float(wv.sum()) * float(max(l1_ratio, 1e-12))
    grad = Xs.T @ (wv * y0)
    alpha_max = float(np.max(np.abs(grad)) / denom) if denom > 0 else 1.0
    alpha_max = max(alpha_max, 1e-6)

    exps = np.arange(n_alphas, dtype=np.float64) / float(n_alphas - 1)
    alphas = (alpha_max * (epsilon**exps)).astype(np.float64)

    gkf = GroupKFold(n_splits=int(n_folds))
    fold_data: List[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        tr_mask = np.zeros(len(y), dtype=bool)
        tr_mask[tr_idx] = True

        X_res, y_res = _segment_residualize_xy(
            X,
            y,
            segments=segments,
            sample_weight=sample_weight,
            train_mask=tr_mask,
        )

        X_tr = X_res.iloc[tr_idx]
        X_va = X_res.iloc[va_idx]
        y_tr = y_res.iloc[tr_idx]
        y_va = y_res.iloc[va_idx]
        w_tr = sample_weight.iloc[tr_idx]
        w_va = sample_weight.iloc[va_idx]

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr_s = scaler.fit_transform(X_tr.to_numpy(dtype=np.float64, copy=False))
        X_va_s = scaler.transform(X_va.to_numpy(dtype=np.float64, copy=False))

        fold_data.append(
            (
                X_tr_s,
                X_va_s,
                y_tr.to_numpy(dtype=np.float64, copy=False),
                y_va.to_numpy(dtype=np.float64, copy=False),
                w_tr.to_numpy(dtype=np.float64, copy=False),
                w_va.to_numpy(dtype=np.float64, copy=False),
            )
        )

    mean_err: List[float] = []
    se_err: List[float] = []

    for a in alphas.tolist():
        fold_errs: List[float] = []
        for X_tr_s, X_va_s, y_tr_v, y_va_v, w_tr_v, w_va_v in fold_data:
            model = ElasticNet(
                alpha=float(a),
                l1_ratio=float(l1_ratio),
                fit_intercept=False,
                max_iter=int(max_iter),
                random_state=int(random_state),
            )
            model.fit(X_tr_s, y_tr_v, sample_weight=w_tr_v)
            pred = model.predict(X_va_s).astype(np.float64)
            err = (pred - y_va_v) ** 2
            denom_w = float(w_va_v.sum())
            fold_errs.append(float((w_va_v * err).sum() / denom_w) if denom_w > 0 else float(np.mean(err)))

        m = float(np.mean(fold_errs))
        se = float(np.std(fold_errs, ddof=1) / np.sqrt(len(fold_errs))) if len(fold_errs) > 1 else 0.0
        mean_err.append(m)
        se_err.append(se)

    df = pd.DataFrame({"alpha": alphas.astype(float), "cv_mean_mse": mean_err, "cv_se": se_err})
    idx_min = int(df["cv_mean_mse"].astype(float).idxmin())
    mse_min = float(df.loc[idx_min, "cv_mean_mse"])
    se_min = float(df.loc[idx_min, "cv_se"])

    if bool(use_1se):
        thresh = mse_min + se_min
        cand = df.loc[df["cv_mean_mse"] <= thresh, "alpha"].astype(float)
        alpha_star = float(cand.max())
        df["selected"] = df["alpha"].astype(float) == alpha_star
    else:
        alpha_star = float(df.loc[idx_min, "alpha"])
        df["selected"] = df.index == idx_min

    return alpha_star, df


def _bootstrap_stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    groups: pd.Series,
    segments: pd.Series,
    sample_weight: pd.Series,
    alpha: float,
    l1_ratio: float,
    B: int,
    delta: float,
    tau: float,
    random_state: int = 0,
    max_iter: int = 5000,
) -> Tuple[pd.Series, pd.Series, List[str]]:
    if B <= 0:
        raise ValueError("bootstrap_B 必须 > 0。")

    rng = np.random.default_rng(int(random_state))
    patient_ids = groups.astype(str).unique().tolist()
    n_patients = len(patient_ids)
    if n_patients < 2:
        raise ValueError("病人数不足（<2），无法做 bootstrap 稳定选择。")

    features = X.columns.tolist()
    sel = np.zeros((int(B), len(features)), dtype=bool)
    abs_beta = np.zeros((int(B), len(features)), dtype=np.float64)

    g_all = groups.astype(str).to_numpy()
    idx_by_patient: Dict[str, np.ndarray] = {}
    for pid in patient_ids:
        idx_by_patient[str(pid)] = np.where(g_all == str(pid))[0]

    for b in range(int(B)):
        sampled = rng.choice(patient_ids, size=n_patients, replace=True).tolist()
        boot_indices = np.concatenate([idx_by_patient[str(pid)] for pid in sampled], axis=0)

        X_b = X.iloc[boot_indices]
        y_b = y.iloc[boot_indices]
        seg_b = segments.iloc[boot_indices]
        w_b = sample_weight.iloc[boot_indices]

        X_res_b, y_res = _segment_residualize_xy(
            X_b,
            y_b,
            segments=seg_b,
            sample_weight=w_b,
            train_mask=None,
        )

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_s = scaler.fit_transform(X_res_b.to_numpy(dtype=np.float64, copy=False))

        model = ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            fit_intercept=False,
            max_iter=int(max_iter),
            random_state=int(random_state),
        )
        model.fit(X_s, y_res.to_numpy(dtype=np.float64, copy=False), sample_weight=w_b.to_numpy())

        coef = model.coef_.astype(np.float64, copy=False)
        absb = np.abs(coef)
        abs_beta[b, :] = absb
        sel[b, :] = absb > float(delta)

    pi = sel.mean(axis=0).astype(np.float64)
    mean_abs = abs_beta.mean(axis=0).astype(np.float64)

    pi_s = pd.Series(pi, index=features, name="pi")
    mean_abs_s = pd.Series(mean_abs, index=features, name="mean_abs_beta")

    stable = pi_s.index[pi_s.to_numpy(dtype=np.float64, copy=False) >= float(tau)].tolist()
    r_s = (pi_s.astype(float) * mean_abs_s.astype(float)).rename("rank_score")
    stable_sorted = sorted(
        stable,
        key=lambda f: (
            -float(r_s.loc[f]),
            -float(pi_s.loc[f]),
            -float(mean_abs_s.loc[f]),
            f,
        ),
    )
    return pi_s, mean_abs_s, stable_sorted


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
) -> Tuple[Dict[str, pd.DataFrame], List[str], List[str], int]:
    common_index = None
    common_cols = None
    for df in conditions.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
        common_cols = df.columns if common_cols is None else common_cols.intersection(df.columns)

    if common_index is None or common_cols is None:
        raise ValueError("输入为空，无法分析。")

    n_features_initial = int(len(common_cols))
    aligned = {k: v.loc[common_index, common_cols].copy() for k, v in conditions.items()}

    for df in aligned.values():
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    dropped_nan_features: List[str] = []
    if drop_any_nan_feature:
        bad_cols = pd.Index([])
        for df in aligned.values():
            bad_cols = bad_cols.union(df.columns[df.isna().any(axis=0)])
        if len(bad_cols) > 0:
            dropped_nan_features = bad_cols.tolist()
            for k in list(aligned.keys()):
                aligned[k].drop(columns=bad_cols, inplace=True)

    gold = aligned["gold"]
    variances = gold.var(axis=0, ddof=1)
    const_cols = variances[variances <= drop_constant_feature_eps].index
    dropped_constant_features = const_cols.tolist()
    if len(const_cols) > 0:
        for k in list(aligned.keys()):
            aligned[k].drop(columns=const_cols, inplace=True)

    if aligned["gold"].shape[1] == 0:
        raise ValueError("清洗后无可用特征（可能存在大量缺失/常量特征）。")

    return aligned, dropped_nan_features, dropped_constant_features, n_features_initial


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
    alpha_fdr: float,
    rho_min: float,
) -> Tuple[pd.Series, pd.Series, pd.Series, List[str]]:
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

    q_s = _benjamini_hochberg(p_s)
    keep = rho_s.index[
        (~rho_s.isna()) & (~p_s.isna()) & (~q_s.isna()) & (q_s < alpha_fdr) & (rho_s.abs() >= rho_min)
    ].tolist()
    return rho_s, p_s, q_s, keep


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


def _benjamini_hochberg(p_s: pd.Series) -> pd.Series:
    p = p_s.to_numpy(dtype=np.float64, copy=False)
    q = np.full(p.shape, np.nan, dtype=np.float64)

    mask = np.isfinite(p)
    if int(mask.sum()) == 0:
        return pd.Series(q, index=p_s.index, name="spearman_q")

    pv = p[mask]
    order = np.argsort(pv)
    pv_sorted = pv[order]

    n = float(pv_sorted.size)
    q_sorted = np.empty_like(pv_sorted)
    prev = 1.0
    for i in range(pv_sorted.size - 1, -1, -1):
        rank = float(i + 1)
        val = float(pv_sorted[i]) * n / rank
        prev = min(prev, val)
        q_sorted[i] = prev

    qv = np.clip(q_sorted, 0.0, 1.0)
    q[mask] = qv[np.argsort(order)]
    return pd.Series(q, index=p_s.index, name="spearman_q")


def _pav_isotonic(values: np.ndarray, weights: np.ndarray, *, increasing: bool) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    if v.size != w.size:
        raise ValueError("values 与 weights 长度不一致。")
    if v.size == 0:
        return v

    if not increasing:
        v = -v

    starts: List[int] = []
    ends: List[int] = []
    vals: List[float] = []
    ws: List[float] = []
    for i in range(v.size):
        starts.append(i)
        ends.append(i + 1)
        vals.append(float(v[i]))
        ws.append(float(w[i]))

        while len(vals) >= 2 and vals[-2] > vals[-1]:
            w_new = ws[-2] + ws[-1]
            if w_new <= 0:
                v_new = 0.5 * (vals[-2] + vals[-1])
            else:
                v_new = (vals[-2] * ws[-2] + vals[-1] * ws[-1]) / w_new

            starts[-2] = starts[-2]
            ends[-2] = ends[-1]
            vals[-2] = float(v_new)
            ws[-2] = float(w_new)

            starts.pop()
            ends.pop()
            vals.pop()
            ws.pop()

    fitted = np.empty(v.size, dtype=np.float64)
    for s, e, val in zip(starts, ends, vals):
        fitted[s:e] = float(val)

    if not increasing:
        fitted = -fitted
    return fitted


def _monotonicity_filter(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sign: pd.Series,
    tau_mono: float,
) -> Tuple[pd.Series, List[str]]:
    if X.shape[1] == 0:
        return pd.Series(dtype=np.float64, name="monotonic_delta"), []

    grade = y.astype(int)
    grouped = X.groupby(grade, sort=True)
    med = grouped.median(numeric_only=True)
    n_g = grouped.size().to_numpy(dtype=np.float64, copy=False)
    grade_levels = med.index.to_numpy()

    deltas: List[float] = []
    cols = X.columns.tolist()

    for c in cols:
        m = med[c].to_numpy(dtype=np.float64, copy=False)
        keep_mask = n_g > 0
        m2 = m[keep_mask]
        w2 = n_g[keep_mask]

        sgn = float(sign.get(c, np.nan))
        increasing = bool(sgn >= 0)
        m_hat = _pav_isotonic(m2, w2, increasing=increasing)

        numer = float(np.sum(w2 * (m2 - m_hat) ** 2))
        mean = float(np.sum(w2 * m2) / np.sum(w2))
        denom = float(np.sum(w2 * (m2 - mean) ** 2))
        if denom <= 0.0:
            deltas.append(0.0)
        else:
            deltas.append(float(numer / denom))

    delta_s = pd.Series(deltas, index=cols, name="monotonic_delta")
    keep = delta_s.index[(~delta_s.isna()) & (delta_s <= float(tau_mono))].tolist()
    return delta_s, keep


@dataclass(frozen=True)
class _DedupAudit:
    kept_features: List[str]
    pass_dedup: pd.Series
    removed_by: pd.Series
    removed_corr: pd.Series
    removed_abs_corr: pd.Series
    removed_order: pd.Series
    max_abs_corr: pd.Series
    n_corr_over_threshold: pd.Series
    decisions: pd.DataFrame


def _dedup_by_spearman_with_score(
    X: pd.DataFrame,
    *,
    score: pd.Series,
    threshold: float,
) -> _DedupAudit:
    cols = X.columns.tolist()
    if X.shape[1] == 0:
        empty = pd.Series(dtype="boolean", name="pass_dedup")
        return _DedupAudit(
            kept_features=[],
            pass_dedup=empty,
            removed_by=pd.Series(dtype="object", name="dedup_removed_by"),
            removed_corr=pd.Series(dtype=np.float64, name="dedup_removed_corr"),
            removed_abs_corr=pd.Series(dtype=np.float64, name="dedup_removed_abs_corr"),
            removed_order=pd.Series(dtype=np.float64, name="dedup_removed_order"),
            max_abs_corr=pd.Series(dtype=np.float64, name="dedup_max_abs_corr"),
            n_corr_over_threshold=pd.Series(dtype=np.int64, name="dedup_n_corr_over_threshold"),
            decisions=pd.DataFrame(
                columns=[
                    "feature1",
                    "feature2",
                    "spearman_rho",
                    "abs_rho",
                    "score1",
                    "score2",
                    "removed",
                    "kept",
                    "removed_score",
                    "kept_score",
                    "removed_order",
                ]
            ),
        )
    if X.shape[1] == 1:
        one = cols[0]
        pass_dedup = pd.Series([True], index=[one], dtype="boolean", name="pass_dedup")
        removed_by = pd.Series([pd.NA], index=[one], dtype="object", name="dedup_removed_by")
        removed_corr = pd.Series([np.nan], index=[one], name="dedup_removed_corr")
        removed_abs = pd.Series([np.nan], index=[one], name="dedup_removed_abs_corr")
        removed_order = pd.Series([pd.NA], index=[one], dtype="Int64", name="dedup_removed_order")
        max_abs = pd.Series([0.0], index=[one], name="dedup_max_abs_corr")
        n_over = pd.Series([0], index=[one], name="dedup_n_corr_over_threshold")
        return _DedupAudit(
            kept_features=[one],
            pass_dedup=pass_dedup,
            removed_by=removed_by,
            removed_corr=removed_corr,
            removed_abs_corr=removed_abs,
            removed_order=removed_order,
            max_abs_corr=max_abs,
            n_corr_over_threshold=n_over,
            decisions=pd.DataFrame(
                columns=[
                    "feature1",
                    "feature2",
                    "spearman_rho",
                    "abs_rho",
                    "score1",
                    "score2",
                    "removed",
                    "kept",
                    "removed_score",
                    "kept_score",
                    "removed_order",
                ]
            ),
        )

    corr = X.corr(method="spearman")
    abs_corr_df = corr.abs()

    abs_no_diag = abs_corr_df.copy()
    np.fill_diagonal(abs_no_diag.values, 0.0)
    max_abs_corr = abs_no_diag.max(axis=1).rename("dedup_max_abs_corr")
    n_over = (abs_no_diag > float(threshold)).sum(axis=1).astype(int).rename("dedup_n_corr_over_threshold")

    removed_by = pd.Series([pd.NA] * len(cols), index=cols, dtype="object", name="dedup_removed_by")
    removed_corr = pd.Series([np.nan] * len(cols), index=cols, name="dedup_removed_corr")
    removed_abs = pd.Series([np.nan] * len(cols), index=cols, name="dedup_removed_abs_corr")
    removed_order = pd.Series([pd.NA] * len(cols), index=cols, dtype="Int64", name="dedup_removed_order")

    def _safe_score(feat: str) -> float:
        v = float(score.get(feat, np.nan))
        return v if np.isfinite(v) else float("-inf")

    corr_v = corr.to_numpy(dtype=np.float64, copy=False)
    abs_v = np.abs(corr_v)
    np.fill_diagonal(abs_v, 0.0)

    pairs: List[tuple[float, str, str]] = []
    thr = float(threshold)
    for i, fi in enumerate(cols[:-1]):
        row_abs = abs_v[i, i + 1 :]
        if row_abs.size == 0:
            continue
        mask = row_abs > thr
        if not bool(mask.any()):
            continue
        js = (np.nonzero(mask)[0] + (i + 1)).tolist()
        for j in js:
            fj = cols[j]
            pairs.append((float(abs_v[i, j]), fi, fj))

    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

    remaining: set[str] = set(cols)
    decisions: List[dict] = []
    order = 1

    idx = 0
    while idx < len(pairs):
        abs_rho, f1, f2 = pairs[idx]
        idx += 1
        if f1 not in remaining or f2 not in remaining:
            continue

        s1 = _safe_score(f1)
        s2 = _safe_score(f2)

        if s1 < s2:
            removed, kept = f1, f2
            removed_score, kept_score = s1, s2
        elif s2 < s1:
            removed, kept = f2, f1
            removed_score, kept_score = s2, s1
        else:
            if f1 <= f2:
                kept, removed = f1, f2
                kept_score, removed_score = s1, s2
            else:
                kept, removed = f2, f1
                kept_score, removed_score = s2, s1

        remaining.remove(removed)

        rho = float(corr.loc[removed, kept])
        removed_by.loc[removed] = kept
        removed_corr.loc[removed] = rho
        removed_abs.loc[removed] = float(abs(rho))
        removed_order.loc[removed] = int(order)

        decisions.append(
            {
                "feature1": removed,
                "feature2": kept,
                "spearman_rho": float(rho),
                "abs_rho": float(abs(rho)),
                "score1": float(removed_score),
                "score2": float(kept_score),
                "removed": removed,
                "kept": kept,
                "removed_score": float(removed_score),
                "kept_score": float(kept_score),
                "removed_order": int(order),
            }
        )
        order += 1

    kept_features = sorted([c for c in cols if c in remaining], key=lambda c: (-_safe_score(c), c))
    kept_set = set(kept_features)
    pass_dedup = pd.Series([c in kept_set for c in cols], index=cols, dtype="boolean", name="pass_dedup")
    decisions_df = pd.DataFrame(
        decisions,
        columns=[
            "feature1",
            "feature2",
            "spearman_rho",
            "abs_rho",
            "score1",
            "score2",
            "removed",
            "kept",
            "removed_score",
            "kept_score",
            "removed_order",
        ],
    )
    return _DedupAudit(
        kept_features=kept_features,
        pass_dedup=pass_dedup,
        removed_by=removed_by,
        removed_corr=removed_corr,
        removed_abs_corr=removed_abs,
        removed_order=removed_order,
        max_abs_corr=max_abs_corr,
        n_corr_over_threshold=n_over,
        decisions=decisions_df,
    )


def _pick_top_features_for_plot(
    features: Sequence[str],
    *,
    score: pd.Series,
    max_features: int,
) -> List[str]:
    """Pick a small, stable subset of features for visualization.

    When the feature set is large, full correlation heatmaps become unreadable and slow.
    We therefore take top-N by composite score (descending), tie-broken by name.
    """
    feats = list(features)
    if len(feats) <= int(max_features):
        return feats

    def _safe_score(name: str) -> float:
        v = float(score.get(name, np.nan))
        return v if np.isfinite(v) else float("-inf")

    feats_sorted = sorted(feats, key=lambda f: (-_safe_score(f), f))
    return feats_sorted[: int(max_features)]


def _spearman_corr_for_plot(
    X: pd.DataFrame,
    *,
    features: Sequence[str],
    score: pd.Series,
    max_features: int,
) -> Tuple[List[str], Optional[pd.DataFrame]]:
    sel = _pick_top_features_for_plot(features, score=score, max_features=max_features)
    if len(sel) < 2:
        return sel, None
    corr = X.loc[:, sel].corr(method="spearman")
    return sel, corr


def analyze_robustness(
    *,
    unperturbed_csv: str | Path,
    perturbed_csv: str | Path,
    pfirrmann_csv: str | Path,
    feature_types: Sequence[str] = _ALLOWED_FEATURE_TYPES,
    enable_stability_selection: bool = True,
    force_include_features: Sequence[str] = (),
    enable_pca: bool = True,
    pca_eta: float = 0.95,
    pca_m_cap: int = 50,
    deep_feature_prefixes: Sequence[str] = ("Deep_",),
    tensor_feature_prefixes: Sequence[str] = ("Tucker_", "Tensor_", "tensor_", "TENSOR_"),
    icc_threshold: float = 0.80,
    alpha_fdr: float = 0.05,
    rho_min: float = 0.20,
    dup_corr_threshold: float = 0.95,
    enet_l1_ratio: float = 0.8,
    enable_auto_lambda_cv: bool = True,
    lambda_value: float = 0.01,
    lambda_cv_folds: int = 5,
    lambda_cv_n_alphas: int = 30,
    lambda_cv_epsilon: float = 0.01,
    lambda_cv_use_1se: bool = True,
    enable_lambda_size_tuning: bool = False,
    bootstrap_B: int = 100,
    stability_delta: float = 1e-4,
    stability_tau: float = 0.5,
    k_max: Optional[int] = 100,
    segments: Sequence[str] = SEGMENTS,
) -> RobustnessAnalysisResult:
    feature_types_norm = _normalize_feature_types(feature_types)
    types_set = set(feature_types_norm)
    force_norm = _normalize_force_include_features(force_include_features, segments=segments)

    conditions_raw = _load_conditions(unperturbed_csv, perturbed_csv, segments=segments)
    gold_full = conditions_raw["gold"]

    conditions_proc_raw: Dict[str, pd.DataFrame] = {}
    for cond, df in conditions_raw.items():
        keep_cols = [c for c in df.columns if _feature_type_category(c) in types_set]
        conditions_proc_raw[cond] = df.loc[:, keep_cols].copy()

    conditions, dropped_nan, dropped_const, n_features_initial = _align_and_clean_conditions(conditions_proc_raw)
    n_features_after_cleaning = int(conditions["gold"].shape[1])
    if n_features_after_cleaning <= 0:
        raise ValueError("按 feature_types 过滤并清洗后无可用特征列（请检查输入CSV或调整类型选择）。")

    cond_names = ["gold"] + [c for c in conditions.keys() if c != "gold"]

    deep_info: Optional[pd.DataFrame] = None
    tensor_info: Optional[pd.DataFrame] = None

    if bool(enable_pca):
        deep_cols, tensor_cols, _other_cols = _pick_feature_groups(
            conditions["gold"].columns.tolist(),
            deep_prefixes=deep_feature_prefixes,
            tensor_prefixes=tensor_feature_prefixes,
        )

        deep_scaler, deep_pca, deep_info = _fit_pca(
            conditions["gold"].loc[:, deep_cols], eta=float(pca_eta), m_cap=int(pca_m_cap)
        )
        tensor_scaler, tensor_pca, tensor_info = _fit_pca(
            conditions["gold"].loc[:, tensor_cols], eta=float(pca_eta), m_cap=int(pca_m_cap)
        )

        for c in cond_names:
            df = conditions[c]
            if deep_scaler is not None and deep_pca is not None and deep_cols:
                df = _apply_pca(df, cols=deep_cols, scaler=deep_scaler, pca=deep_pca, out_prefix="DeepPCA_")
            if tensor_scaler is not None and tensor_pca is not None and tensor_cols:
                df = _apply_pca(df, cols=tensor_cols, scaler=tensor_scaler, pca=tensor_pca, out_prefix="TensorPCA_")
            conditions[c] = df

        common_cols = None
        for df in conditions.values():
            common_cols = df.columns if common_cols is None else common_cols.intersection(df.columns)
        if common_cols is None or len(common_cols) == 0:
            raise RuntimeError("PCA 后未得到任何可用特征列。")
        for c in cond_names:
            conditions[c] = conditions[c].loc[:, common_cols]

    grade_df = load_pfirrmann_grades(pfirrmann_csv, segments=segments)
    y = _build_grade_series(grade_df, conditions["gold"].index)

    mats = [conditions[c] for c in cond_names]
    icc = compute_icc_21(mats, condition_names=cond_names)
    icc_keep = icc.index[(~icc.isna()) & (icc >= float(icc_threshold))].tolist()
    if len(icc_keep) == 0:
        raise ValueError("ICC 鲁棒性筛选后无特征保留（请放宽 T_ICC 或检查输入文件）。")

    rho_s, p_s, q_s, keep_by_spearman = _spearman_filter(
        conditions["gold"], y, alpha_fdr=float(alpha_fdr), rho_min=float(rho_min)
    )
    if len(keep_by_spearman) == 0:
        raise ValueError("Spearman+FDR 相关性预筛后无特征保留（请检查分级文件或放宽阈值）。")

    icc_set = set(icc_keep)
    s_rel = [f for f in keep_by_spearman if f in icc_set]
    if len(s_rel) == 0:
        raise ValueError("ICC 与 Spearman 交集后无特征保留（请放宽阈值或检查数据）。")

    gold_X = conditions["gold"]
    rel_X = gold_X.loc[:, s_rel]
    score = (rho_s.abs() * icc).rename("composite_score")
    dedup_audit = _dedup_by_spearman_with_score(rel_X, score=score, threshold=float(dup_corr_threshold))
    s_nr = dedup_audit.kept_features
    if len(s_nr) == 0:
        raise ValueError("去冗余后无特征保留（请放宽 T_dup）。")

    # Prepare small correlation matrices for visualization (top-N by composite score).
    corr_pre_feats, corr_pre = _spearman_corr_for_plot(rel_X, features=s_rel, score=score, max_features=80)
    corr_post_feats, corr_post = _spearman_corr_for_plot(rel_X, features=s_nr, score=score, max_features=80)

    lambda_table: Optional[pd.DataFrame] = None
    lambda_cv_selected: Optional[float] = None
    lambda_size_tuning_table: Optional[pd.DataFrame] = None

    if bool(enable_stability_selection):
        groups = pd.Series(gold_X.index.get_level_values(0).astype(str), index=gold_X.index, name="case_id")
        seg_s = pd.Series(gold_X.index.get_level_values(1).astype(str), index=gold_X.index, name="segment")
        w = _patient_weights(gold_X.index)
        X_sel = gold_X.loc[:, s_nr]

        if bool(enable_auto_lambda_cv):
            alpha_star, lambda_table = _elasticnet_cv_lambda(
                X_sel,
                y,
                groups=groups,
                segments=seg_s,
                sample_weight=w,
                l1_ratio=float(enet_l1_ratio),
                n_folds=int(lambda_cv_folds),
                n_alphas=int(lambda_cv_n_alphas),
                epsilon=float(lambda_cv_epsilon),
                use_1se=bool(lambda_cv_use_1se),
            )
            lambda_mode = "auto_cv"
            lambda_cv_selected = float(alpha_star)
            lambda_used = float(alpha_star)
            cv_folds: Optional[int] = int(lambda_cv_folds)
            cv_n_alphas: Optional[int] = int(lambda_cv_n_alphas)
            cv_eps: Optional[float] = float(lambda_cv_epsilon)
            cv_1se: Optional[bool] = bool(lambda_cv_use_1se)
        else:
            lambda_mode = "manual"
            lambda_used = float(lambda_value)
            cv_folds = None
            cv_n_alphas = None
            cv_eps = None
            cv_1se = None

        lambda_size_tuning_enabled = bool(enable_lambda_size_tuning) and (lambda_mode == "auto_cv")

        pi_s, mean_abs_s, stable_sorted = _bootstrap_stability_selection(
            X_sel,
            y,
            groups=groups,
            segments=seg_s,
            sample_weight=w,
            alpha=float(lambda_used),
            l1_ratio=float(enet_l1_ratio),
            B=int(bootstrap_B),
            delta=float(stability_delta),
            tau=float(stability_tau),
        )

        k_target = int(k_max) if (k_max is not None and int(k_max) > 0) else None
        if (
            bool(lambda_size_tuning_enabled)
            and k_target is not None
            and lambda_table is not None
            and not lambda_table.empty
        ):
            alpha_grid = lambda_table["alpha"].astype(float).to_numpy()
            if "selected" in lambda_table.columns and bool(lambda_table["selected"].astype(bool).any()):
                idx_star = int(lambda_table.index[lambda_table["selected"].astype(bool)].to_list()[0])
            else:
                idx_star = int(np.argmin(np.abs(alpha_grid - float(lambda_used))))

            trace: List[dict] = []

            def _record(alpha: float, n_stable: int, note: str) -> None:
                trace.append({"alpha": float(alpha), "n_stable": int(n_stable), "note": str(note)})

            _record(float(lambda_used), int(len(stable_sorted)), "start")

            best_alpha = float(lambda_used)
            best_pi = pi_s
            best_mean_abs = mean_abs_s
            best_stable = stable_sorted

            if len(best_stable) > int(k_target):
                for idx in range(int(idx_star) - 1, -1, -1):
                    a = float(alpha_grid[idx])
                    pi_t, mean_abs_t, stable_t = _bootstrap_stability_selection(
                        X_sel,
                        y,
                        groups=groups,
                        segments=seg_s,
                        sample_weight=w,
                        alpha=a,
                        l1_ratio=float(enet_l1_ratio),
                        B=int(bootstrap_B),
                        delta=float(stability_delta),
                        tau=float(stability_tau),
                    )
                    _record(a, int(len(stable_t)), "increase_lambda")
                    if len(stable_t) <= int(k_target):
                        best_alpha, best_pi, best_mean_abs, best_stable = a, pi_t, mean_abs_t, stable_t
                        break
            elif len(best_stable) == 0:
                for idx in range(int(idx_star) + 1, int(len(alpha_grid))):
                    a = float(alpha_grid[idx])
                    pi_t, mean_abs_t, stable_t = _bootstrap_stability_selection(
                        X_sel,
                        y,
                        groups=groups,
                        segments=seg_s,
                        sample_weight=w,
                        alpha=a,
                        l1_ratio=float(enet_l1_ratio),
                        B=int(bootstrap_B),
                        delta=float(stability_delta),
                        tau=float(stability_tau),
                    )
                    _record(a, int(len(stable_t)), "decrease_lambda")
                    if len(stable_t) > 0:
                        best_alpha, best_pi, best_mean_abs, best_stable = a, pi_t, mean_abs_t, stable_t
                        break

            lambda_used = float(best_alpha)
            pi_s, mean_abs_s, stable_sorted = best_pi, best_mean_abs, best_stable
            lambda_size_tuning_table = pd.DataFrame(trace, columns=["alpha", "n_stable", "note"])

        final_proc = stable_sorted
        if k_max is not None and int(k_max) > 0 and len(final_proc) > int(k_max):
            final_proc = final_proc[: int(k_max)]

        stable_pi = pi_s
        stable_mean_abs = mean_abs_s
    else:
        lambda_mode = "skipped"
        lambda_used = float("nan")
        cv_folds = None
        cv_n_alphas = None
        cv_eps = None
        cv_1se = None
        lambda_size_tuning_enabled = False
        stable_pi = pd.Series(dtype=float)
        stable_mean_abs = pd.Series(dtype=float)
        final_proc = list(s_nr)

    gold_full_aligned = gold_full.reindex(gold_X.index)
    gold_out = gold_X
    force_to_add = [f for f in force_norm if f not in gold_out.columns and f in gold_full_aligned.columns]
    if force_to_add:
        gold_out = pd.concat([gold_out, gold_full_aligned.loc[:, force_to_add]], axis=1)

    final_all = list(final_proc)
    force_added: List[str] = []
    for f in force_norm:
        if f in final_all:
            continue
        if f in gold_out.columns:
            final_all.append(f)
            force_added.append(f)

    X_out = gold_out.loc[:, final_all]
    case_id = gold_out.index.get_level_values(0).astype(str)
    segment = gold_out.index.get_level_values(1).astype(str)
    case_level = pd.Series(case_id + "_" + segment, index=gold_out.index, name="case_id_层级")
    final_model_input = pd.concat([case_level, X_out], axis=1).reset_index(drop=True)

    pass_topk = pd.Series([pd.NA] * len(dedup_audit.pass_dedup), index=dedup_audit.pass_dedup.index, dtype="boolean")
    final_set = set(final_all)
    for f in s_nr:
        pass_topk.loc[f] = f in final_set

    return RobustnessAnalysisResult(
        conditions=cond_names,
        n_samples=int(gold_out.shape[0]),
        segments=list(segments),
        feature_types=list(feature_types_norm),
        stability_selection_enabled=bool(enable_stability_selection),
        force_include_features=list(force_norm),
        force_included_added=list(force_added),
        n_features_initial=int(n_features_initial),
        n_features_after_cleaning=int(n_features_after_cleaning),
        dropped_nan_features=dropped_nan,
        dropped_constant_features=dropped_const,
        pca_enabled=bool(enable_pca),
        pca_eta=float(pca_eta),
        pca_m_cap=int(pca_m_cap),
        pca_deep_info=deep_info,
        pca_tensor_info=tensor_info,
        deep_feature_prefixes=list(map(str, deep_feature_prefixes)),
        tensor_feature_prefixes=list(map(str, tensor_feature_prefixes)),
        spearman_rho=rho_s,
        spearman_p=p_s,
        spearman_q=q_s,
        spearman_alpha_fdr=float(alpha_fdr),
        spearman_rho_min=float(rho_min),
        selected_by_spearman=keep_by_spearman,
        icc=icc,
        icc_threshold=float(icc_threshold),
        robust_features=icc_keep,
        composite_score=score,
        dup_corr_threshold=float(dup_corr_threshold),
        enet_l1_ratio=float(enet_l1_ratio),
        lambda_mode=lambda_mode,
        lambda_value=float(lambda_used),
        lambda_cv_folds=cv_folds,
        lambda_cv_n_alphas=cv_n_alphas,
        lambda_cv_epsilon=cv_eps,
        lambda_cv_use_1se=cv_1se,
        lambda_cv_table=lambda_table,
        lambda_cv_selected=(float(lambda_cv_selected) if lambda_cv_selected is not None else None),
        lambda_size_tuning_enabled=bool(lambda_size_tuning_enabled),
        lambda_size_tuning_table=lambda_size_tuning_table,
        bootstrap_B=int(bootstrap_B),
        stability_delta=float(stability_delta),
        stability_tau=float(stability_tau),
        stable_pi=stable_pi,
        stable_mean_abs_beta=stable_mean_abs,
        k_max=(int(k_max) if k_max is not None else None),
        final_features=final_all,
        final_model_input=final_model_input,
        pass_dedup=dedup_audit.pass_dedup,
        pass_topk=pass_topk,
        dedup_removed_by=dedup_audit.removed_by,
        dedup_removed_corr=dedup_audit.removed_corr,
        dedup_removed_abs_corr=dedup_audit.removed_abs_corr,
        dedup_removed_order=dedup_audit.removed_order,
        dedup_max_abs_corr=dedup_audit.max_abs_corr,
        dedup_n_corr_over_threshold=dedup_audit.n_corr_over_threshold,
        dedup_decisions=dedup_audit.decisions,
        corr_pre_dedup_features=corr_pre_feats,
        corr_post_dedup_features=corr_post_feats,
        corr_pre_dedup=corr_pre,
        corr_post_dedup=corr_post,
    )


def save_analysis_result(
    result: RobustnessAnalysisResult,
    *,
    output_dir: str | Path,
    pfirrmann_csv: str | Path | None = None,
    statistics_csv: str | Path | None = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if result.pca_deep_info is not None and not result.pca_deep_info.empty:
        result.pca_deep_info.to_csv(out / "pca_deep_info.csv", index=False)
    if result.pca_tensor_info is not None and not result.pca_tensor_info.empty:
        result.pca_tensor_info.to_csv(out / "pca_tensor_info.csv", index=False)
    if result.lambda_cv_table is not None and not result.lambda_cv_table.empty:
        result.lambda_cv_table.to_csv(out / "lambda_cv.csv", index=False)
    if result.lambda_size_tuning_table is not None and not result.lambda_size_tuning_table.empty:
        result.lambda_size_tuning_table.to_csv(out / "lambda_size_tuning.csv", index=False)

    sel_icc = set(result.robust_features)
    sel_spearman = set(result.selected_by_spearman)
    sel_final = set(result.final_features)

    def _build_feature_table(features: Sequence[str]) -> pd.DataFrame:
        feat_list = list(features)
        df = pd.DataFrame({"feature": feat_list})

        df["icc_21"] = result.icc.reindex(feat_list).to_numpy()
        pass_icc = df["feature"].isin(sel_icc)
        df["pass_icc"] = pass_icc.where(~df["icc_21"].isna(), other=pd.NA)

        df["spearman_rho"] = result.spearman_rho.reindex(feat_list).to_numpy()
        df["spearman_p"] = result.spearman_p.reindex(feat_list).to_numpy()
        df["spearman_q"] = result.spearman_q.reindex(feat_list).to_numpy()
        df["pass_spearman"] = df["feature"].isin(sel_spearman)

        df["composite_score"] = result.composite_score.reindex(feat_list).to_numpy()

        df["pass_dedup"] = result.pass_dedup.reindex(feat_list).to_numpy()
        df["pass_topk"] = result.pass_topk.reindex(feat_list).to_numpy()
        df["dedup_removed_by"] = result.dedup_removed_by.reindex(feat_list).to_numpy()
        df["dedup_removed_corr"] = result.dedup_removed_corr.reindex(feat_list).to_numpy()
        df["dedup_removed_abs_corr"] = result.dedup_removed_abs_corr.reindex(feat_list).to_numpy()
        df["dedup_removed_order"] = result.dedup_removed_order.reindex(feat_list).to_numpy()
        df["dedup_max_abs_corr"] = result.dedup_max_abs_corr.reindex(feat_list).to_numpy()
        df["dedup_n_corr_over_threshold"] = result.dedup_n_corr_over_threshold.reindex(feat_list).to_numpy()

        df["pi"] = result.stable_pi.reindex(feat_list).to_numpy()
        df["mean_abs_beta"] = result.stable_mean_abs_beta.reindex(feat_list).to_numpy()
        df["rank_score"] = df["pi"].astype(float) * df["mean_abs_beta"].astype(float)
        pi_v = pd.to_numeric(df["pi"], errors="coerce")
        pass_stability = pd.Series(pd.NA, index=df.index, dtype="boolean")
        mask = ~pi_v.isna()
        if bool(mask.any()):
            pass_stability.loc[mask] = pi_v.loc[mask] >= float(result.stability_tau)
        df["pass_stability"] = pass_stability

        df["final_selected"] = df["feature"].isin(sel_final)
        return df

    _build_feature_table(result.spearman_rho.index.tolist()).to_csv(out / "spearman_with_pfirrmann.csv", index=False)

    _build_feature_table(result.icc.index.tolist()).to_csv(out / "icc_values.csv", index=False)

    if result.robust_features:
        _build_feature_table(result.robust_features).to_csv(out / "robust_features_by_icc.csv", index=False)

    st_in = result.pass_dedup.index[result.pass_dedup.astype(bool)].tolist()
    if st_in:
        _build_feature_table(st_in).to_csv(out / "stability_selection.csv", index=False)

    _build_feature_table(result.final_features).to_csv(out / "final_robust_features.csv", index=False)

    result.final_model_input.to_csv(out / "最终模型输入.csv", index=False, encoding="utf-8-sig")

    if result.dedup_decisions is not None:
        result.dedup_decisions.to_csv(out / "dedup_pairs.csv", index=False)

    report = out / "analysis_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("椎间盘特征稳健性相关性分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"条件数 k: {len(result.conditions)}\n")
        f.write(f"条件列表: {', '.join(result.conditions)}\n")
        f.write(f"样本数 N_disc: {result.n_samples}\n")
        f.write(f"节段: {', '.join(result.segments)}\n")
        f.write(f"特征处理对象: {', '.join(result.feature_types)}\n")
        f.write(f"步骤5启用: {bool(result.stability_selection_enabled)}\n")
        if result.force_include_features:
            f.write(f"强制纳入候选: {', '.join(result.force_include_features)}\n")
            f.write(
                f"强制纳入新增: "
                f"{', '.join(result.force_included_added) if result.force_included_added else '无'}\n"
            )
        f.write("\n")

        f.write("步骤0：对齐与清洗\n")
        f.write(f"  初始共有特征数（取各条件交集）: {result.n_features_initial}\n")
        f.write(f"  剔除缺失/无穷导致的特征数: {len(result.dropped_nan_features)}\n")
        f.write(f"  剔除常量特征数: {len(result.dropped_constant_features)}\n")
        f.write(f"  清洗后特征数: {result.n_features_after_cleaning}\n\n")

        f.write("步骤1：深度/张量特征 PCA 预降维\n")
        f.write(f"  enable_pca: {bool(result.pca_enabled)}\n")
        if not bool(result.pca_enabled):
            f.write("  跳过（用户关闭；使用原始 deep/tensor 特征，不产生 DeepPCA_/TensorPCA_）\n")
        else:
            f.write(f"  eta_pca: {float(result.pca_eta):g}\n")
            f.write(f"  m_cap: {int(result.pca_m_cap)}\n")
            if result.pca_deep_info is not None:
                f.write(f"  Deep PCA 维度: {int(result.pca_deep_info.shape[0])}\n")
            else:
                f.write("  Deep PCA: 跳过（未检测到 Deep_ 特征）\n")
            if result.pca_tensor_info is not None:
                f.write(f"  Tensor PCA 维度: {int(result.pca_tensor_info.shape[0])}\n")
            else:
                f.write("  Tensor PCA: 跳过（未检测到 Tucker_/Tensor_ 特征）\n")
        f.write("\n")

        f.write("步骤2：扰动鲁棒性筛选（ICC(2,1) 阈值）\n")
        f.write(f"  T_ICC: {float(result.icc_threshold):g}\n")
        f.write(f"  ICC 计算特征数: {len(result.icc)}\n")
        f.write(f"  保留特征数: {len(result.robust_features)}\n\n")

        f.write("步骤3：Pfirrmann 相关性预筛（Spearman + BH-FDR）\n")
        f.write(f"  alpha_FDR: {float(result.spearman_alpha_fdr):g}\n")
        f.write(f"  rho_min: {float(result.spearman_rho_min):g}\n")
        f.write(f"  Spearman 通过特征数: {len(result.selected_by_spearman)}\n\n")

        f.write("步骤4：特征间相关性去冗余（Spearman + 综合得分）\n")
        f.write(f"  T_dup: {float(result.dup_corr_threshold):g}\n")
        f.write(f"  去冗余后特征数: {int(result.pass_dedup.astype(bool).sum())}\n\n")

        f.write("步骤5：病人级 bootstrap + ElasticNet 稳定选择\n")
        if not bool(result.stability_selection_enabled):
            f.write("  跳过（未启用）\n")
            if result.force_include_features:
                f.write(
                    f"  强制纳入新增: {len(result.force_included_added)}/{len(result.force_include_features)}\n"
                )
            f.write(f"  最终特征数: {len(result.final_features)}\n\n")
        else:
            f.write(f"  enet_l1_ratio: {float(result.enet_l1_ratio):g}\n")
            f.write("  segment_covariate: one-hot (unpenalized)\n")
            f.write(f"  lambda_mode: {result.lambda_mode}\n")
            if result.lambda_cv_selected is not None:
                f.write(f"  lambda_cv_selected(alpha*): {float(result.lambda_cv_selected):g}\n")
            f.write(f"  lambda(alpha): {float(result.lambda_value):g}\n")
            if result.lambda_cv_table is not None:
                f.write(
                    f"  lambda_cv: folds={int(result.lambda_cv_folds or 0)} "
                    f"L={int(result.lambda_cv_n_alphas or 0)} epsilon={float(result.lambda_cv_epsilon or 0):g} "
                    f"1se={bool(result.lambda_cv_use_1se)}\n"
                )
            if bool(result.lambda_size_tuning_enabled) and result.lambda_size_tuning_table is not None:
                tried = int(result.lambda_size_tuning_table.shape[0])
                f.write(f"  lambda_size_tuning: enabled=True tried={tried}\n")
            f.write(f"  bootstrap_B: {int(result.bootstrap_B)}\n")
            f.write(f"  delta: {float(result.stability_delta):g}\n")
            f.write(f"  tau: {float(result.stability_tau):g}\n")
            if result.k_max is not None:
                f.write(f"  K_max: {int(result.k_max)}\n")
            f.write(f"  最终特征数: {len(result.final_features)}\n\n")
        f.write(
            f"  最终模型输入.csv: rows={int(result.final_model_input.shape[0])} cols={int(result.final_model_input.shape[1])}\n\n"
        )

    # Figures are best-effort: they should never prevent producing the core CSV outputs.
    try:
        from lumbar_enet.robustness_plots import save_robustness_figures

        grades_long = None
        stats_df = None
        if statistics_csv is not None and str(statistics_csv).strip():
            stats_path = Path(str(statistics_csv).strip())
            if stats_path.exists():
                stats_df = pd.read_csv(stats_path)
                stats_df.columns = [str(c).strip() for c in stats_df.columns]

        if stats_df is not None and pfirrmann_csv is not None and str(pfirrmann_csv).strip():
            grades_long = load_pfirrmann_grades(pfirrmann_csv, segments=result.segments)

        saved = save_robustness_figures(
            result,
            output_dir=out,
            grades_long=grades_long,
            statistics_df=stats_df,
        )
        if saved:
            with report.open("a", encoding="utf-8") as f:
                f.write("附：统计图示输出\n")
                for name in saved:
                    f.write(f"  - {name}\n")
                f.write("\n")
    except Exception as e:
        with report.open("a", encoding="utf-8") as f:
            f.write("附：统计图示输出（失败）\n")
            f.write(f"  - error: {type(e).__name__}: {e}\n\n")
