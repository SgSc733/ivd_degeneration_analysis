from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

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
    X_num: pd.DataFrame
    X_ebm: pd.DataFrame
    y: np.ndarray | None
    tasks: np.ndarray
    groups: np.ndarray
    feature_names_num: list[str]
    feature_names_ebm: list[str]
    classic_feature_names: list[str]
    task_names: list[str]


def _normalize_col_name(s: object) -> str:
    # Be tolerant to BOM and accidental whitespace.
    return str(s).strip().lstrip("\ufeff")


def _resolve_column(df: pd.DataFrame, name: str) -> str:
    """Resolve a user-provided column name against a DataFrame.

    This is defensive against BOM/whitespace and Windows console display issues.
    """
    name_n = _normalize_col_name(name)
    cols = list(df.columns)
    norm_to_real: dict[str, str] = {_normalize_col_name(c): str(c) for c in cols}
    if name in df.columns:
        return name
    if name_n in norm_to_real:
        return norm_to_real[name_n]

    # Light heuristic fallback for the common case: the ID column starts with "case_id_".
    for c in cols:
        c_n = _normalize_col_name(c)
        if c_n.startswith("case_id_") and name_n.startswith("case_id_"):
            return str(c)

    raise ValueError(f"Column '{name}' not found. Available columns (first 10): {cols[:10]} ...")


def load_model_input_csv(
    csv_path: str | Path,
    *,
    id_col: str,
    label_col: str | None,
    drop_cols: list[str] | None = None,
    keep_feature_patterns: list[str] | None = None,
    classic_prefix: str = "classic_",
    patient_id_from_id_col: bool = True,
    patient_id_sep: str = "_",
    level_from_id_col: bool = True,
    level_sep: str = "_",
    segment_levels: list[str] | None = None,
    disc_level_feature_name: str = "disc_level",
    pooling_stats: list[str] | None = None,
    pooling_pyr_prefix: str = "PyRadiomics_",
    pooling_out_prefix: str = "pool_",
    pooling_feature_types: list[str] | None = None,
    pooling_deep_prefix: str = "DeepPCA_",
    pooling_tensor_prefix: str = "TensorPCA_",
) -> LoadedTabular:
    """Load the final model input CSV.

    The modeling unit is disc-level rows. We parse:
    - patient_id: used for GroupKFold grouping
    - disc_level: used both as a categorical EBM input, and as "task" for per-level metrics
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Use utf-8; tolerate BOM by normalizing col names afterwards.
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df.rename(columns={c: _normalize_col_name(c) for c in df.columns})

    id_col_real = _resolve_column(df, id_col)

    drop_cols = drop_cols or []
    drop_cols_real = []
    for c in drop_cols:
        try:
            drop_cols_real.append(_resolve_column(df, c))
        except Exception:
            continue
    if drop_cols_real:
        df = df.drop(columns=drop_cols_real)

    # --- meta ---
    id_series = df[id_col_real].astype(str)
    patient_ids: list[str] = []
    disc_levels: list[str | None] = []

    for v in id_series.tolist():
        if patient_id_from_id_col:
            parts = v.split(patient_id_sep)
            patient_ids.append(parts[0] if len(parts) > 0 else v)
        else:
            patient_ids.append(v)

        if level_from_id_col:
            parts2 = v.split(level_sep)
            disc_levels.append(parts2[1] if len(parts2) > 1 else None)
        else:
            disc_levels.append(None)

    meta = pd.DataFrame(
        {
            id_col: id_series,
            "patient_id": patient_ids,
            "disc_level": disc_levels,
        }
    )
    segment_levels = segment_levels or DEFAULT_SEGMENT_LEVELS
    keep_mask = meta["disc_level"].astype(str).isin(segment_levels).to_numpy()
    if not keep_mask.any():
        found = sorted({str(v) for v in meta["disc_level"].dropna().tolist()})
        raise ValueError(
            "No rows matched data.segment_levels. "
            f"Expected one of: {segment_levels}. Found levels: {found}"
        )
    if not keep_mask.all():
        df = df.loc[keep_mask].reset_index(drop=True)
        meta = meta.loc[keep_mask].reset_index(drop=True)

    # --- y (optional in CSV) ---
    y: np.ndarray | None = None
    if label_col:
        label_col_real = _resolve_column(df, label_col) if label_col in df.columns else None
        if label_col_real and label_col_real in df.columns:
            y = pd.to_numeric(df[label_col_real], errors="coerce").to_numpy()
            if np.isnan(y).any():
                raise ValueError(f"label_col '{label_col}' contains NaN after numeric conversion.")

    # --- X_num ---
    exclude_cols = {id_col_real}
    if label_col and label_col in df.columns:
        exclude_cols.add(label_col)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    keep_feature_patterns = keep_feature_patterns or []
    if keep_feature_patterns:
        compiled_patterns = [re.compile(str(pat)) for pat in keep_feature_patterns]
        feature_cols = [
            c for c in feature_cols if any(pat.search(str(c)) for pat in compiled_patterns)
        ]
        if not feature_cols:
            raise ValueError("keep_feature_patterns did not match any feature columns.")


    X_num = df[feature_cols].copy()
    for c in X_num.columns:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce")

    if X_num.isna().any().any():
        med = X_num.median(numeric_only=True)
        X_num = X_num.fillna(med)

    # --- optional intra-patient statistical pooling (方案1.md 4.2; ProtoNAM data.pooling) ---
    stats_norm = _normalize_pooling_stats(pooling_stats)
    if stats_norm:
        types_norm = _normalize_pooling_feature_types(pooling_feature_types)
        types_set = set(types_norm)
        groups_cols: dict[str, list[str]] = {}
        # Keep a stable order for reproducibility (independent of config list order).
        if "classic" in types_set:
            groups_cols["classic"] = [c for c in X_num.columns if c.startswith(str(classic_prefix))]
        if "pyr" in types_set:
            groups_cols["pyr"] = [c for c in X_num.columns if c.startswith(str(pooling_pyr_prefix))]
        if "deep" in types_set:
            groups_cols["deep"] = [c for c in X_num.columns if c.startswith(str(pooling_deep_prefix))]
        if "tensor" in types_set:
            groups_cols["tensor"] = [c for c in X_num.columns if c.startswith(str(pooling_tensor_prefix))]

        X_num = _attach_intra_patient_pooling(
            X_num,
            meta=meta,
            patient_id_col="patient_id",
            groups_cols=groups_cols,
            stats=stats_norm,
            out_prefix=str(pooling_out_prefix),
        )

    # --- disc_level categorical feature (EBM input) ---
    if disc_level_feature_name in X_num.columns:
        raise ValueError(
            f"disc_level_feature_name '{disc_level_feature_name}' conflicts with an existing CSV feature column."
        )
    X_ebm = X_num.copy()
    X_ebm[disc_level_feature_name] = meta["disc_level"].astype(str)

    feature_names_num = list(X_num.columns)
    feature_names_ebm = list(X_ebm.columns)
    classic_feature_names = [c for c in feature_names_num if c.startswith(classic_prefix)]

    # --- tasks + groups ---
    level_to_task = {lvl: i for i, lvl in enumerate(segment_levels)}
    disc_levels_s = meta["disc_level"].astype(str).tolist()
    tasks = np.asarray([level_to_task.get(lvl, -1) for lvl in disc_levels_s], dtype=int)
    if (tasks < 0).any():
        bad = sorted({disc_levels_s[i] for i in np.where(tasks < 0)[0][:10].tolist()})
        raise ValueError(
            "Unknown disc_level values (cannot map to task index). "
            f"Examples: {bad}. Expected one of: {segment_levels}"
        )

    groups = meta["patient_id"].to_numpy()

    return LoadedTabular(
        df_raw=df,
        meta=meta,
        X_num=X_num,
        X_ebm=X_ebm,
        y=y,
        tasks=tasks,
        groups=groups,
        feature_names_num=feature_names_num,
        feature_names_ebm=feature_names_ebm,
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
    - Pooling is computed *within patient* only. With GroupKFold by patient, this does not introduce cross-fold leakage.
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


def _read_xlsx_sheet_to_df(xlsx_path: str | Path, sheet_name: str | None) -> pd.DataFrame:
    # Prefer pandas+openpyxl (available in the target env), but keep a fallback.
    try:
        return pd.read_excel(xlsx_path, sheet_name=sheet_name or 0, engine="openpyxl")
    except Exception:
        from lumbar_ebm.xlsx_reader import read_xlsx_sheet_to_df

        return read_xlsx_sheet_to_df(xlsx_path, sheet_name=sheet_name)


def attach_pfirrmann_from_xlsx(
    loaded: LoadedTabular,
    *,
    xlsx_path: str | Path,
    sheet_name: str | None,
    patient_id_col: str,
    level_to_col: dict[str, str],
    on_missing: str = "error",  # "error" | "drop"
) -> LoadedTabular:
    """Attach Pfirrmann labels from a wide Excel table.

    Expected columns in Excel:
        patient_id_col, L12, L23, L34, L45, L5S1
    """
    df_lab = _read_xlsx_sheet_to_df(xlsx_path, sheet_name=sheet_name)
    if df_lab.empty:
        raise ValueError(f"Empty label sheet in: {xlsx_path}")

    df_lab = df_lab.rename(columns={c: _normalize_col_name(c) for c in df_lab.columns})
    patient_id_col_real = _resolve_column(df_lab, patient_id_col)

    lab_map: dict[str, dict[str, object]] = {}
    for _, row in df_lab.iterrows():
        pid_raw = row.get(patient_id_col_real)
        if pid_raw is None or str(pid_raw).strip() == "":
            continue
        pid = _norm_id(pid_raw)
        lab_map[pid] = row.to_dict()

    y_list: list[float] = []
    missing_ids: list[str] = []

    for i in range(len(loaded.meta)):
        pid = _norm_id(loaded.meta.loc[i, "patient_id"])
        lvl = str(loaded.meta.loc[i, "disc_level"])
        col = level_to_col.get(lvl)
        if not col:
            missing_ids.append(str(loaded.meta.loc[i, loaded.meta.columns[0]]))
            y_list.append(float("nan"))
            continue
        row_dict = lab_map.get(pid, {})
        v = row_dict.get(col)
        if v is None or str(v).strip() == "":
            missing_ids.append(str(loaded.meta.loc[i, loaded.meta.columns[0]]))
            y_list.append(float("nan"))
            continue
        try:
            y_list.append(float(v))
        except Exception:
            missing_ids.append(str(loaded.meta.loc[i, loaded.meta.columns[0]]))
            y_list.append(float("nan"))

    y = np.asarray(y_list, dtype=float)
    ok = ~np.isnan(y)

    if not ok.all():
        if on_missing == "drop":
            meta2 = loaded.meta.loc[ok].reset_index(drop=True)
            X_num2 = loaded.X_num.loc[ok].reset_index(drop=True)
            X_ebm2 = loaded.X_ebm.loc[ok].reset_index(drop=True)
            tasks2 = loaded.tasks[ok]
            groups2 = meta2["patient_id"].to_numpy()
            y2 = y[ok]
            return LoadedTabular(
                df_raw=loaded.df_raw,
                meta=meta2,
                X_num=X_num2,
                X_ebm=X_ebm2,
                y=y2.astype(int),
                tasks=tasks2,
                groups=groups2,
                feature_names_num=loaded.feature_names_num,
                feature_names_ebm=loaded.feature_names_ebm,
                classic_feature_names=loaded.classic_feature_names,
                task_names=loaded.task_names,
            )

        sample = ", ".join(missing_ids[:10])
        raise ValueError(
            f"Missing pfirrmann labels for {int((~ok).sum())} rows. "
            f"Examples (case_id_层级): {sample}"
        )

    return LoadedTabular(
        df_raw=loaded.df_raw,
        meta=loaded.meta,
        X_num=loaded.X_num,
        X_ebm=loaded.X_ebm,
        y=y.astype(int),
        tasks=loaded.tasks,
        groups=loaded.groups,
        feature_names_num=loaded.feature_names_num,
        feature_names_ebm=loaded.feature_names_ebm,
        classic_feature_names=loaded.classic_feature_names,
        task_names=loaded.task_names,
    )
