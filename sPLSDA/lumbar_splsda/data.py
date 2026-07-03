from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumbar_splsda.xlsx_reader import read_xlsx_sheet_to_df

DEFAULT_SEGMENT_LEVELS = ["L3-L4", "L4-L5", "L5-S1"]
DEFAULT_LEVEL_TO_COL = {
    "L3-L4": "L34",
    "L4-L5": "L45",
    "L5-S1": "L5S1",
}


@dataclass(frozen=True)
class LoadedTabular:
    df_raw: pd.DataFrame
    meta: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray | None
    tasks: np.ndarray
    groups: np.ndarray
    feature_names: list[str]
    classic_feature_names: list[str]
    task_names: list[str]


def _safe_onehot_col_name(level: str) -> str:
    # Windows-friendly filename/column name: avoid '/', '-', etc.
    return "seg_" + (
        level.replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _safe_meta_onehot_token(token: str) -> str:
    """
    Make a token safe for use in a Windows filename.

    NOTE: explain.py uses feature names directly as filenames, so we must avoid
    invalid characters like <>:"/\\|?*.
    """

    s = str(token).strip()
    # Preserve a bit of semantics for common inequality markers.
    s = (
        s.replace(">=", "ge_")
        .replace("<=", "le_")
        .replace(">", "gt_")
        .replace("<", "lt_")
        .replace("=", "eq_")
    )
    s = re.sub(r"[\\/:*?\"<>|\s-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "empty"


def _safe_case_meta_onehot_col_name(prefix: str, col: str, level: str) -> str:
    return f"{prefix}{_safe_meta_onehot_token(col)}__{_safe_meta_onehot_token(level)}"


def _normalize_pooling_feature_types(feature_types: list[str] | None) -> list[str]:
    """
    Normalize pooling feature-group names to canonical ids.

    Canonical groups:
      - classic
      - pyr        (PyRadiomics)
      - deep       (DeepPCA)
      - tensor     (TensorPCA)

    If feature_types is None, defaults to the historical behavior: ["classic", "pyr"].
    """

    if feature_types is None:
        return ["classic", "pyr"]
    out: list[str] = []
    for t in feature_types:
        key = str(t).strip().lower().replace("-", "_")
        if not key:
            continue
        if key in {"all", "*"}:
            return ["classic", "pyr", "deep", "tensor"]
        if key in {"classic"}:
            norm = "classic"
        elif key in {"pyr", "pyradiomics", "radiomics"}:
            norm = "pyr"
        elif key in {"deep", "deeppca"}:
            norm = "deep"
        elif key in {"tensor", "tensorpca"}:
            norm = "tensor"
        else:
            raise ValueError(
                "data.pooling.feature_types items must be one of: classic, pyr/pyradiomics, deep, tensor, all. "
                f"Got: {t!r}"
            )
        if norm not in out:
            out.append(norm)
    return out


def load_model_input_csv(
    csv_path: str | Path,
    *,
    id_col: str,
    label_col: str | None,
    drop_cols: list[str] | None = None,
    drop_patterns: list[str] | None = None,
    feature_order: str = "csv",  # "csv" | "sorted"
    classic_prefix: str = "classic_",
    patient_id_from_id_col: bool = True,
    patient_id_sep: str = "_",
    level_from_id_col: bool = True,
    level_sep: str = "_",
    segment_levels: list[str] | None = None,
    add_segment_onehot: bool = True,
    pooling_stats: list[str] | None = None,
    pooling_pyr_prefix: str = "PyRadiomics_",
    pooling_out_prefix: str = "pool_",
    pooling_feature_types: list[str] | None = None,
    pooling_deep_prefix: str = "DeepPCA_",
    pooling_tensor_prefix: str = "TensorPCA_",
    add_case_meta_onehot: bool = False,
    case_meta_csv_path: str | Path | None = None,
    case_meta_case_id_col: str = "image_id",
    case_meta_cols: list[str] | None = None,
    case_meta_onehot_prefix: str = "meta_",
    case_meta_onehot_drop_first: bool = False,
    case_meta_on_missing: str = "error",  # "error" | "unknown"
) -> LoadedTabular:
    """
    Load the final model input CSV and build:
    - X: numeric features (+ optional segment one-hot + optional case-meta one-hot)
    - y: labels (optional; can be missing for inference-only runs)
    - groups: patient-level grouping id for GroupKFold
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in CSV columns: {list(df.columns)[:10]} ...")

    drop_cols = drop_cols or []
    drop_patterns = drop_patterns or []
    drop_set: set[str] = set(drop_cols)

    reserved = {id_col}
    if label_col:
        reserved.add(label_col)

    for pat in drop_patterns:
        r = re.compile(str(pat))
        for c in df.columns:
            if c in reserved:
                continue
            if r.search(c):
                drop_set.add(c)

    drop_cols_eff = sorted([c for c in drop_set if c in df.columns and c not in reserved])
    if drop_cols_eff:
        df = df.drop(columns=drop_cols_eff)

    # --- meta columns ---
    id_series = df[id_col].astype(str)
    patient_ids: list[str] = []
    levels: list[str | None] = []
    for v in id_series.tolist():
        parts = v.split(patient_id_sep) if patient_id_from_id_col else [v]
        patient_ids.append(parts[0] if len(parts) > 0 else v)

        if level_from_id_col:
            parts2 = v.split(level_sep)
            levels.append(parts2[1] if len(parts2) > 1 else None)
        else:
            levels.append(None)

    segment_levels = list(dict.fromkeys(segment_levels or DEFAULT_SEGMENT_LEVELS))
    keep_mask = pd.Series(levels, dtype="object").astype(str).isin(segment_levels).to_numpy()
    if not keep_mask.any():
        found = sorted({str(v) for v in levels if v is not None})
        raise ValueError(
            "No rows matched data.segment_levels. "
            f"Expected one of: {segment_levels}. Found levels: {found}"
        )
    if not keep_mask.all():
        df = df.loc[keep_mask].reset_index(drop=True)
        id_series = id_series.loc[keep_mask].reset_index(drop=True)
        patient_ids = [pid for pid, keep in zip(patient_ids, keep_mask.tolist()) if keep]
        levels = [lvl for lvl, keep in zip(levels, keep_mask.tolist()) if keep]

    meta = pd.DataFrame(
        {
            id_col: id_series,
            "patient_id": patient_ids,
            "disc_level": levels,
        }
    )
    if meta[id_col].duplicated().any():
        dup = meta.loc[meta[id_col].duplicated(), id_col].astype(str).tolist()[:5]
        raise ValueError(f"id_col values must be unique to ensure safe alignment. Duplicates examples: {dup}")
    # Use a stable, human-readable index to make downstream alignment (e.g. harmonization)
    # fail-fast if someone accidentally resets/rebuilds DataFrames.
    meta = meta.set_index(id_col, drop=False)

    # --- y ---
    y: np.ndarray | None = None
    if label_col and label_col in df.columns:
        y = pd.to_numeric(df[label_col], errors="coerce").to_numpy()
        if np.isnan(y).any():
            raise ValueError(f"label_col '{label_col}' contains NaN after numeric conversion.")
    elif label_col:
        # label is optional; allow inference-only runs.
        y = None

    # --- X ---
    exclude_cols = {id_col}
    if label_col and label_col in df.columns:
        exclude_cols.add(label_col)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_order = str(feature_order).strip().lower()
    if feature_order == "sorted":
        feature_cols = sorted(feature_cols)
    elif feature_order == "csv":
        pass
    else:
        raise ValueError(f"feature_order must be one of: csv, sorted (got {feature_order!r})")

    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Missing-value imputation must be fit on the training fold only to avoid leakage.
    # Therefore we do NOT fill NaNs here; fold-inner imputation is handled in train.py.

    # IMPORTANT: align X with meta by the stable sample-id index *before* any ops that
    # rely on pandas' index alignment (e.g., one-hot, intra-patient pooling).
    X.index = meta.index

    if add_case_meta_onehot:
        case_meta_cols = list(case_meta_cols or [])
        if not case_meta_cols:
            raise ValueError("add_case_meta_onehot=True requires a non-empty case_meta_cols.")

        stats_path = Path(case_meta_csv_path or "statistics.csv")
        if not stats_path.exists():
            raise FileNotFoundError(f"case_meta_csv_path not found: {stats_path}")

        stats_df = pd.read_csv(stats_path, encoding="utf-8")
        case_id_col_stats = str(case_meta_case_id_col)
        if case_id_col_stats not in stats_df.columns:
            raise ValueError(
                f"case_meta_case_id_col '{case_id_col_stats}' not found in statistics CSV columns: {list(stats_df.columns)}"
            )

        missing_cols = [c for c in case_meta_cols if c not in stats_df.columns]
        if missing_cols:
            raise ValueError(
                "statistics CSV missing required columns for case_meta_onehot: "
                f"{missing_cols}. Available columns: {list(stats_df.columns)}"
            )

        df_stats = stats_df[[case_id_col_stats] + case_meta_cols].copy()
        df_stats[case_id_col_stats] = df_stats[case_id_col_stats].map(_norm_id)
        df_stats = df_stats.dropna(subset=[case_id_col_stats])
        df_stats = df_stats.drop_duplicates(subset=[case_id_col_stats], keep="first").reset_index(drop=True)
        df_stats = df_stats.set_index(case_id_col_stats, drop=True)

        on_missing = str(case_meta_on_missing).strip().lower()

        case_ids = meta["patient_id"].map(_norm_id)
        missing = case_ids[~case_ids.isin(df_stats.index)]
        if len(missing) > 0:
            miss_ex = sorted({str(x) for x in missing.tolist()})[:10]
            msg = f"statistics CSV does not cover all cases referenced by patient_id. Missing examples: {miss_ex}"
            if on_missing == "error":
                raise ValueError(msg)
            if on_missing != "unknown":
                raise ValueError(f"Unsupported case_meta_on_missing policy: {case_meta_on_missing!r}. {msg}")

        # Attach raw case-level meta columns onto meta (useful for debugging/analysis).
        for c in case_meta_cols:
            if c in meta.columns:
                raise ValueError(f"case_meta column name conflicts with existing meta column: {c!r}")
            meta[c] = case_ids.map(df_stats[c])

        prefix = str(case_meta_onehot_prefix)
        drop_first = bool(case_meta_onehot_drop_first)

        # One-hot encode the requested meta columns and append to X.
        for c in case_meta_cols:
            raw = meta[c]
            if raw.isna().any():
                if on_missing == "error":
                    bad = case_ids[raw.isna()].tolist()[:10]
                    raise ValueError(f"case_meta column {c!r} has missing values after join. Missing examples: {bad}")
                if on_missing == "unknown":
                    raw = raw.astype(object).where(raw.notna(), other="UNKNOWN")
                else:
                    raise ValueError(f"Unsupported case_meta_on_missing policy: {case_meta_on_missing!r}")

            s = raw.astype(str).str.strip()
            levels = sorted(s.unique().tolist())
            if drop_first and len(levels) > 0:
                levels = levels[1:]

            seen_cols: set[str] = set()
            for lvl in levels:
                col_name = _safe_case_meta_onehot_col_name(prefix, c, lvl)
                if col_name in X.columns or col_name in seen_cols:
                    raise ValueError(
                        "case_meta_onehot generated a duplicate/overlapping feature name. "
                        f"Please change case_meta_onehot_prefix or inspect levels. Duplicate: {col_name!r}"
                    )
                seen_cols.add(col_name)
                X[col_name] = (s == lvl).astype(np.float32)

    if add_segment_onehot:
        level_series = meta["disc_level"].astype(str)
        for lvl in segment_levels:
            col = _safe_onehot_col_name(lvl)
            X[col] = (level_series == lvl).astype(np.float32)

    # --- optional intra-patient statistical pooling (方案1.md 4.2) ---
    stats_norm = _normalize_pooling_stats(pooling_stats)
    if stats_norm:
        types_norm = _normalize_pooling_feature_types(pooling_feature_types)
        types_set = set(types_norm)
        groups_cols: dict[str, list[str]] = {}
        # Keep a stable order for reproducibility (independent of config list order).
        if "classic" in types_set:
            groups_cols["classic"] = [c for c in X.columns if c.startswith(str(classic_prefix))]
        if "pyr" in types_set:
            groups_cols["pyr"] = [c for c in X.columns if c.startswith(str(pooling_pyr_prefix))]
        if "deep" in types_set:
            groups_cols["deep"] = [c for c in X.columns if c.startswith(str(pooling_deep_prefix))]
        if "tensor" in types_set:
            groups_cols["tensor"] = [c for c in X.columns if c.startswith(str(pooling_tensor_prefix))]

        X = _attach_intra_patient_pooling(
            X,
            meta=meta,
            patient_id_col="patient_id",
            groups_cols=groups_cols,
            stats=stats_norm,
            out_prefix=str(pooling_out_prefix),
        )

    # Keep X aligned with meta by the stable sample id index.
    X.index = meta.index

    feature_names = list(X.columns)
    classic_feature_names = [c for c in feature_names if c.startswith(classic_prefix)]

    # Segment task index (used for multi-task weighting; not part of X by default).
    level_to_task = {lvl: i for i, lvl in enumerate(segment_levels)}
    disc_levels = meta["disc_level"].astype(str).tolist()
    tasks = np.asarray([level_to_task.get(lvl, -1) for lvl in disc_levels], dtype=int)
    if (tasks < 0).any():
        bad = sorted({disc_levels[i] for i in np.where(tasks < 0)[0][:10].tolist()})
        raise ValueError(
            "Unknown disc_level values (cannot map to task index). "
            f"Examples: {bad}. Expected one of: {segment_levels}"
        )

    groups = meta["patient_id"].to_numpy()

    return LoadedTabular(
        df_raw=df,
        meta=meta,
        X=X,
        y=y,
        tasks=tasks,
        groups=groups,
        feature_names=feature_names,
        classic_feature_names=classic_feature_names,
        task_names=segment_levels,
    )


def _normalize_pooling_stats(stats: list[str] | None) -> list[str]:
    if not stats:
        return []
    out: list[str] = []
    for s in stats:
        key = str(s).strip().lower()
        if not key:
            continue
        if key == "mean":
            norm = "mean"
        elif key == "delta":
            norm = "delta"
        elif key in {"mean_minus_d", "mean_-d", "mean-d"}:
            norm = "mean_minus_d"
        else:
            raise ValueError(
                "data.pooling.stats items must be one of: mean, delta, mean_minus_d (or alias mean_-d). "
                f"Got: {s!r}"
            )
        if norm not in out:
            out.append(norm)
    return out


def _attach_intra_patient_pooling(
    X: pd.DataFrame,
    *,
    meta: pd.DataFrame,
    patient_id_col: str,
    groups_cols: dict[str, list[str]],
    stats: list[str],
    out_prefix: str,
) -> pd.DataFrame:
    """
    Intra-patient statistical pooling (方案1.md 4.2).

    For a feature group tau, and for each patient i with discs d in D_i:
      mean:        x̄_i = (1/D_i) Σ_d x_{i,d}
      delta:       Δx_{i,d} = x_{i,d} - x̄_i
      mean_minus_d: x̄_{i,-d} = (D_i * x̄_i - x_{i,d}) / (D_i - 1)   (strictly excludes current disc d)

    Notes:
    - Pooling is computed *within patient* only, so it does not introduce cross-fold leakage.
    - Disabled stats are NOT created (no zero placeholders).
    """
    if X.empty:
        return X

    pid = meta[patient_id_col].astype(str)
    if pid.shape[0] != X.shape[0]:
        raise ValueError(f"meta.{patient_id_col} length mismatch: meta={pid.shape[0]} vs X={X.shape[0]}")

    out_prefix = str(out_prefix or "pool_")

    parts: list[pd.DataFrame] = [X]
    for group_name, cols in (groups_cols or {}).items():
        if not cols:
            continue
        parts.append(
            _pool_group(
                X,
                pid=pid,
                cols=cols,
                group_name=str(group_name),
                stats=stats,
                out_prefix=out_prefix,
            )
        )
    if len(parts) == 1:
        return X
    return pd.concat(parts, axis=1)


def _pool_group(
    X: pd.DataFrame,
    *,
    pid: pd.Series,
    cols: list[str],
    group_name: str,
    stats: list[str],
    out_prefix: str,
) -> pd.DataFrame:
    if not cols or not stats:
        return pd.DataFrame(index=X.index)

    xg = X[cols]
    mean_df = xg.groupby(pid, sort=False).transform("mean")
    out_dfs: list[pd.DataFrame] = []

    if "mean" in stats:
        df = mean_df.copy()
        df.columns = [f"{out_prefix}{group_name}_mean__{c}" for c in cols]
        out_dfs.append(df)

    if "delta" in stats:
        df = xg - mean_df
        df.columns = [f"{out_prefix}{group_name}_delta__{c}" for c in cols]
        out_dfs.append(df)

    if "mean_minus_d" in stats:
        D = pid.groupby(pid, sort=False).transform("size").to_numpy(dtype=np.float64)
        if (D < 2).any():
            bad = pid[np.where(D < 2)[0][:5]].tolist()
            raise ValueError(
                "mean_minus_d pooling requires each patient to have at least 2 discs. "
                f"Found patients with D_i<2, examples: {bad}"
            )
        x_np = xg.to_numpy(dtype=np.float64, copy=False)
        mean_np = mean_df.to_numpy(dtype=np.float64, copy=False)
        mean_minus = (D[:, None] * mean_np - x_np) / (D[:, None] - 1.0)
        df = pd.DataFrame(
            mean_minus,
            index=xg.index,
            columns=[f"{out_prefix}{group_name}_mean_minus_d__{c}" for c in cols],
        )
        out_dfs.append(df)

    if not out_dfs:
        return pd.DataFrame(index=X.index)
    return pd.concat(out_dfs, axis=1)


def _norm_id(v: object) -> str:
    s = str(v).strip()
    # Handle Excel numeric like "1.0"
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def attach_pfirrmann_from_xlsx(
    loaded: LoadedTabular,
    *,
    xlsx_path: str | Path,
    sheet_name: str | None,
    patient_id_col: str,
    level_to_col: dict[str, str],
    on_missing: str = "error",  # "error" | "drop"
) -> LoadedTabular:
    """
    Attach pfirrmann label vector y from a wide Excel table:
        patient_id_col, L34, L45, L5S1

    The join key uses loaded.meta["patient_id"] and loaded.meta["disc_level"].
    """
    df_lab = read_xlsx_sheet_to_df(xlsx_path, sheet_name=sheet_name)
    if df_lab.empty:
        raise ValueError(f"Empty label sheet in: {xlsx_path}")
    if patient_id_col not in df_lab.columns:
        raise ValueError(f"patient_id_col '{patient_id_col}' not found in label sheet columns: {list(df_lab.columns)}")

    # Build mapping: patient_id -> row (dict)
    lab_map: dict[str, dict[str, object]] = {}
    for _, row in df_lab.iterrows():
        pid_raw = row.get(patient_id_col)
        if pid_raw is None or str(pid_raw).strip() == "":
            continue
        pid = _norm_id(pid_raw)
        lab_map[pid] = row.to_dict()

    y_list: list[float] = []
    missing_ids: list[str] = []

    for i in range(len(loaded.meta)):
        # Use iloc (position-based) so meta can have a stable non-integer index.
        pid = _norm_id(loaded.meta.iloc[i]["patient_id"])
        lvl = loaded.meta.iloc[i]["disc_level"]
        lvl = str(lvl) if lvl is not None else ""
        col = level_to_col.get(lvl)
        if not col:
            missing_ids.append(str(loaded.meta.iloc[i][loaded.meta.columns[0]]))
            y_list.append(float("nan"))
            continue
        row_dict = lab_map.get(pid, {})
        v = row_dict.get(col)
        if v is None or str(v).strip() == "":
            missing_ids.append(str(loaded.meta.iloc[i][loaded.meta.columns[0]]))
            y_list.append(float("nan"))
            continue
        try:
            y_list.append(float(v))
        except Exception:
            missing_ids.append(str(loaded.meta.iloc[i][loaded.meta.columns[0]]))
            y_list.append(float("nan"))

    y = np.asarray(y_list, dtype=float)
    ok = ~np.isnan(y)

    if not ok.all():
        if on_missing == "drop":
            # Preserve index to keep alignment deterministic for downstream steps
            # (e.g., harmonization uses DataFrame.index for subsetting).
            meta2 = loaded.meta.loc[ok].copy()
            X2 = loaded.X.loc[ok].copy()
            tasks2 = loaded.tasks[ok]
            groups2 = meta2["patient_id"].to_numpy()
            y2 = y[ok]
            return LoadedTabular(
                df_raw=loaded.df_raw,
                meta=meta2,
                X=X2,
                y=y2,
                tasks=tasks2,
                groups=groups2,
                feature_names=loaded.feature_names,
                classic_feature_names=loaded.classic_feature_names,
                task_names=loaded.task_names,
            )

        # error
        sample = ", ".join(missing_ids[:10])
        raise ValueError(
            f"Missing pfirrmann labels for {int((~ok).sum())} rows. "
            f"Examples (case_id_层级): {sample}"
        )

    return LoadedTabular(
        df_raw=loaded.df_raw,
        meta=loaded.meta,
        X=loaded.X,
        y=y.astype(int),
        tasks=loaded.tasks,
        groups=loaded.groups,
        feature_names=loaded.feature_names,
        classic_feature_names=loaded.classic_feature_names,
        task_names=loaded.task_names,
    )
