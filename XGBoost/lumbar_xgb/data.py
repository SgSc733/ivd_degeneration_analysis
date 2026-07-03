from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from lumbar_xgb.xlsx_reader import read_xlsx_sheet_to_df


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
    return "seg_" + (level.replace("/", "_").replace("\\", "_").replace("-", "_").replace(" ", "_"))


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
    add_segment_onehot: bool = False,
    pooling_stats: list[str] | None = None,
    pooling_pyr_prefix: str = "PyRadiomics_",
    pooling_out_prefix: str = "pool_",
    pooling_feature_types: list[str] | None = None,
    pooling_deep_prefix: str = "DeepPCA_",
    pooling_tensor_prefix: str = "TensorPCA_",
) -> LoadedTabular:
    """
    Load the final model input CSV and build:
    - X: numeric features (+ optional segment one-hot)
    - y: labels (optional; can be missing for inference-only runs)
    - groups: patient-level grouping id for GroupKFold
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # NOTE: input CSVs are typically saved as UTF-8 with BOM; pandas handles it under utf-8.
    df = pd.read_csv(csv_path, encoding="utf-8")
    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in CSV columns: {list(df.columns)[:10]} ...")

    drop_cols = drop_cols or []
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

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

    meta = pd.DataFrame(
        {
            id_col: id_series,
            "patient_id": patient_ids,
            "disc_level": levels,
        }
    )

    # --- y ---
    y: np.ndarray | None = None
    if label_col and label_col in df.columns:
        y = pd.to_numeric(df[label_col], errors="coerce").to_numpy()
        if np.isnan(y).any():
            raise ValueError(f"label_col '{label_col}' contains NaN after numeric conversion.")
    elif label_col:
        y = None

    # --- X ---
    exclude_cols = {id_col}
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


    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if X.isna().any().any():
        med = X.median(numeric_only=True)
        X = X.fillna(med)

    # IMPORTANT: keep X aligned with meta before any ops that rely on index alignment
    # (e.g., intra-patient statistical pooling).
    X.index = meta.index

    if add_segment_onehot:
        segment_levels = segment_levels or ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
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

    # Keep X aligned with meta.
    X.index = meta.index

    feature_names = list(X.columns)
    classic_feature_names = [c for c in feature_names if c.startswith(classic_prefix)]

    # Segment task index (used for per-task metrics and optional explanations).
    segment_levels = segment_levels or ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
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


def _norm_id(v: object) -> str:
    s = str(v).strip()
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
    Attach Pfirrmann label vector y from a wide Excel table:
        patient_id_col, L12, L23, L34, L45, L5S1
    """
    df_lab = read_xlsx_sheet_to_df(xlsx_path, sheet_name=sheet_name)
    if df_lab.empty:
        raise ValueError(f"Empty label sheet in: {xlsx_path}")
    if patient_id_col not in df_lab.columns:
        raise ValueError(f"patient_id_col '{patient_id_col}' not found in label sheet columns: {list(df_lab.columns)}")

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
        pid = _norm_id(loaded.meta.loc[i, "patient_id"])
        lvl = loaded.meta.loc[i, "disc_level"]
        lvl = str(lvl) if lvl is not None else ""
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
            X2 = loaded.X.loc[ok].reset_index(drop=True)
            tasks2 = loaded.tasks[ok]
            groups2 = meta2["patient_id"].to_numpy()
            y2 = y[ok]
            return LoadedTabular(
                df_raw=loaded.df_raw,
                meta=meta2,
                X=X2,
                y=y2.astype(int),
                tasks=tasks2,
                groups=groups2,
                feature_names=loaded.feature_names,
                classic_feature_names=loaded.classic_feature_names,
                task_names=loaded.task_names,
            )

        sample = ", ".join(missing_ids[:10])
        raise ValueError(
            f"Missing pfirrmann labels for {int((~ok).sum())} rows. "
            f"Examples ({loaded.meta.columns[0]}): {sample}"
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
