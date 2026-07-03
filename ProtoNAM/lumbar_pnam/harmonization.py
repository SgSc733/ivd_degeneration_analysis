from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd


def _safe_loc(obj: pd.Series | pd.DataFrame, idx: pd.Index, *, obj_name: str) -> pd.Series | pd.DataFrame:
    """
    Safer .loc with an explicit error message for index misalignment.

    Harmonization relies on stable sample IDs (DataFrame.index) to align X with batch/covariates.
    If someone resets/rebuilds X.index, .loc will either KeyError or (worse) silently misalign.
    """
    try:
        return obj.loc[idx]
    except KeyError as e:
        idx2 = pd.Index(idx)
        missing = idx2[~idx2.isin(obj.index)]
        examples = [str(x) for x in missing.tolist()[:5]]
        raise KeyError(
            f"{obj_name}.loc[X.index] failed due to index mismatch. "
            "Requirement: X.index must equal the meta/index used to build harmonization batch/covariates. "
            "Do NOT reset_index(drop=True) or rebuild DataFrames before harmonization. "
            f"Missing examples: {examples}"
        ) from e


def _norm_id(v: object) -> str:
    """
    Normalize an ID to a stable string key.

    Notes:
    - Handles common "Excel numeric" artifacts like "1.0" -> "1".
    - Does NOT strip leading zeros (kept as-is to avoid unintended collisions).
    """

    s = str(v).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def _read_statistics_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"statistics.csv not found: {path}")
    return pd.read_csv(path, encoding="utf-8")


def _build_case_batch_map(
    *,
    statistics_df: pd.DataFrame,
    case_id_col: str,
    batch_col: str,
    small_batch_cfg: dict[str, Any] | None,
) -> pd.Series:
    """
    Build a case-level mapping: case_id -> batch_label, with optional small-batch merging.
    """
    if case_id_col not in statistics_df.columns:
        raise ValueError(f"statistics.csv missing case_id_col={case_id_col!r}")
    if batch_col not in statistics_df.columns:
        raise ValueError(f"statistics.csv missing batch_col={batch_col!r}")

    df = statistics_df.copy()
    df[case_id_col] = df[case_id_col].map(_norm_id)
    df[batch_col] = df[batch_col].astype(str).str.strip()

    # Ensure one row per case (take the first if duplicates exist).
    df = df.dropna(subset=[case_id_col])
    df = df.drop_duplicates(subset=[case_id_col], keep="first").reset_index(drop=True)

    case_to_batch = pd.Series(df[batch_col].to_numpy(), index=df[case_id_col].to_numpy(), dtype=object)

    sb = small_batch_cfg or {}
    min_cases = int(sb.get("min_cases", 0) or 0)
    strategy = str(sb.get("strategy", "") or "").strip().lower()

    if min_cases > 0 and strategy:
        counts = case_to_batch.value_counts(dropna=False)
        small_batches = set(counts[counts < min_cases].index.astype(str).tolist())

        if small_batches:
            if strategy == "manufacturer__magnetic_field_strength":
                need_cols = ["manufacturer", "magnetic_field_strength"]
                miss = [c for c in need_cols if c not in df.columns]
                if miss:
                    raise ValueError(
                        "small_batch.strategy requires columns in statistics.csv: "
                        f"{miss}. Available columns: {list(df.columns)}"
                    )

                # Compute replacement for small batches at the case level.
                rep = (
                    df["manufacturer"].astype(str).str.strip()
                    + "__"
                    + df["magnetic_field_strength"].astype(str).str.strip()
                )
                rep.index = df[case_id_col].to_numpy()

                for cid, b in case_to_batch.items():
                    if str(b) in small_batches:
                        case_to_batch.loc[cid] = rep.loc[cid]
            else:
                raise ValueError(f"Unsupported small_batch.strategy: {strategy!r}")

    return case_to_batch


def _build_sample_batch_series(
    *,
    meta: pd.DataFrame,
    case_to_batch: pd.Series,
    on_missing: str = "error",  # "error"
) -> pd.Series:
    if "patient_id" not in meta.columns:
        raise ValueError("meta missing required column: 'patient_id'")

    case_ids = meta["patient_id"].map(_norm_id)
    batch = case_ids.map(case_to_batch)

    if batch.isna().any():
        missing = sorted({str(x) for x in case_ids[batch.isna()].tolist()})[:10]
        msg = f"statistics.csv does not cover all cases. Missing examples: {missing}"
        if str(on_missing).strip().lower() == "error":
            raise ValueError(msg)
        raise ValueError(f"Unsupported on_missing policy: {on_missing!r}. {msg}")

    return pd.Series(batch.to_numpy(), index=meta.index, name="batch", dtype=object)


def _build_discrete_covariates(
    *,
    meta: pd.DataFrame,
    use_task: bool,
) -> pd.DataFrame | None:
    if not use_task:
        return None
    if "disc_level" not in meta.columns:
        raise ValueError("meta missing required column for covariates.use_task: 'disc_level'")
    # Use string (categorical) covariate; safer than int-coded tasks for ComBat design-matrix.
    return pd.DataFrame({"task": meta["disc_level"].astype(str)}, index=meta.index)


@dataclass(frozen=True)
class HarmonizationContext:
    enabled: bool
    method: str  # "combat" | "meanvar" | "none"

    batch_all: pd.Series | None
    discrete_covariates_all: pd.DataFrame | None
    continuous_covariates_all: pd.DataFrame | None

    meanvar_eps: float

    combat_variant: str
    combat_parametric: bool
    combat_mean_only: bool
    combat_eps: float
    combat_ref_batch: str | None
    combat_covbat_cov_thresh: float
    combat_min_samples_per_batch_train: int


def build_harmonization_context(*, cfg: dict[str, Any], meta: pd.DataFrame | None) -> HarmonizationContext:
    harm_cfg = cfg.get("harmonization", {}) or {}
    enabled = bool(harm_cfg.get("enabled", False))

    method = str(harm_cfg.get("method", "combat")).strip().lower().replace("-", "_")
    if method in {"off", "disabled"}:
        method = "none"
    if method not in {"combat", "meanvar", "none"}:
        raise ValueError(f"harmonization.method must be one of: combat, meanvar, none (got {method!r})")

    if not enabled or method == "none":
        return HarmonizationContext(
            enabled=False,
            method="none",
            batch_all=None,
            discrete_covariates_all=None,
            continuous_covariates_all=None,
            meanvar_eps=float((harm_cfg.get("meanvar", {}) or {}).get("eps", 1e-6)),
            combat_variant=str((harm_cfg.get("combat", {}) or {}).get("variant", "fortin")),
            combat_parametric=bool((harm_cfg.get("combat", {}) or {}).get("parametric", True)),
            combat_mean_only=bool((harm_cfg.get("combat", {}) or {}).get("mean_only", False)),
            combat_eps=float((harm_cfg.get("combat", {}) or {}).get("eps", 1e-8)),
            combat_ref_batch=(harm_cfg.get("combat", {}) or {}).get("ref_batch"),
            combat_covbat_cov_thresh=float((harm_cfg.get("combat", {}) or {}).get("covbat_cov_thresh", 0.9)),
            combat_min_samples_per_batch_train=int((harm_cfg.get("combat", {}) or {}).get("min_samples_per_batch_train", 2)),
        )

    if meta is None:
        raise ValueError("harmonization requires meta (with patient_id/disc_level), but meta=None was provided.")

    statistics_csv_path = str(harm_cfg.get("statistics_csv_path", "statistics.csv"))
    case_id_col = str(harm_cfg.get("case_id_col", "image_id"))
    batch_col = str(harm_cfg.get("batch_col", "batch_scanner"))

    stats_df = _read_statistics_csv(statistics_csv_path)
    case_to_batch = _build_case_batch_map(
        statistics_df=stats_df,
        case_id_col=case_id_col,
        batch_col=batch_col,
        small_batch_cfg=(harm_cfg.get("small_batch", {}) or {}),
    )
    batch_all = _build_sample_batch_series(meta=meta, case_to_batch=case_to_batch, on_missing="error")

    cov_cfg = harm_cfg.get("covariates", {}) or {}
    discrete_cov = _build_discrete_covariates(meta=meta, use_task=bool(cov_cfg.get("use_task", True)))
    continuous_cov = None

    meanvar_cfg = harm_cfg.get("meanvar", {}) or {}
    combat_cfg = harm_cfg.get("combat", {}) or {}

    return HarmonizationContext(
        enabled=True,
        method=method,
        batch_all=batch_all,
        discrete_covariates_all=discrete_cov,
        continuous_covariates_all=continuous_cov,
        meanvar_eps=float(meanvar_cfg.get("eps", 1e-6)),
        combat_variant=str(combat_cfg.get("variant", "fortin")),
        combat_parametric=bool(combat_cfg.get("parametric", True)),
        combat_mean_only=bool(combat_cfg.get("mean_only", False)),
        combat_eps=float(combat_cfg.get("eps", 1e-8)),
        combat_ref_batch=(None if combat_cfg.get("ref_batch", None) in {None, ""} else str(combat_cfg.get("ref_batch"))),
        combat_covbat_cov_thresh=float(combat_cfg.get("covbat_cov_thresh", 0.9)),
        combat_min_samples_per_batch_train=int(combat_cfg.get("min_samples_per_batch_train", 2)),
    )


class IdentityHarmonizer:
    def fit(self, X: pd.DataFrame) -> "IdentityHarmonizer":
        return self

    def transform(
        self,
        X: pd.DataFrame,
        *,
        batch: pd.Series | None = None,
        discrete_covariates: pd.DataFrame | None = None,
        continuous_covariates: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # Keep signature compatible with other harmonizers (deployable transform).
        _ = batch, discrete_covariates, continuous_covariates
        return X


@dataclass
class MeanVarAlignHarmonizer:
    batch_all: pd.Series
    eps: float = 1e-6

    feature_names: list[str] | None = None
    mu_star_: np.ndarray | None = None
    sigma_star_: np.ndarray | None = None
    mu_by_batch_: dict[str, np.ndarray] | None = None
    sigma_by_batch_: dict[str, np.ndarray] | None = None
    seen_batches_: set[str] | None = None

    def fit(self, X: pd.DataFrame) -> "MeanVarAlignHarmonizer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MeanVarAlignHarmonizer.fit expects a pandas DataFrame.")

        self.feature_names = list(X.columns)
        X_np = X.to_numpy(dtype=float, copy=True)
        mu_star = X_np.mean(axis=0)
        sigma_star = X_np.std(axis=0)
        sigma_star[sigma_star == 0] = 1.0

        b = _safe_loc(self.batch_all, X.index, obj_name="batch_all").astype(str)
        mu_by: dict[str, np.ndarray] = {}
        sig_by: dict[str, np.ndarray] = {}
        for batch in sorted(set(b.tolist())):
            idx = (b == batch).to_numpy()
            xb = X_np[idx]
            mu_b = xb.mean(axis=0)
            sig_b = xb.std(axis=0)
            sig_b[sig_b == 0] = 1.0
            mu_by[str(batch)] = mu_b.astype(np.float32)
            sig_by[str(batch)] = sig_b.astype(np.float32)

        self.mu_star_ = mu_star.astype(np.float32)
        self.sigma_star_ = sigma_star.astype(np.float32)
        self.mu_by_batch_ = mu_by
        self.sigma_by_batch_ = sig_by
        self.seen_batches_ = set(mu_by.keys())
        return self

    def transform(
        self,
        X: pd.DataFrame,
        *,
        batch: pd.Series | None = None,
        discrete_covariates: pd.DataFrame | None = None,
        continuous_covariates: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        _ = discrete_covariates, continuous_covariates
        if (
            self.feature_names is None
            or self.mu_star_ is None
            or self.sigma_star_ is None
            or self.mu_by_batch_ is None
            or self.sigma_by_batch_ is None
            or self.seen_batches_ is None
        ):
            raise RuntimeError("MeanVarAlignHarmonizer is not fitted.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MeanVarAlignHarmonizer.transform expects a pandas DataFrame.")

        if list(X.columns) != list(self.feature_names):
            X = X.reindex(columns=self.feature_names)

        batch_src = self.batch_all if batch is None else batch
        b = _safe_loc(batch_src, X.index, obj_name="batch").astype(str)
        X_np = X.to_numpy(dtype=float, copy=True)
        out = X_np.copy()

        unseen: set[str] = set()
        for batch in sorted(set(b.tolist())):
            idx = (b == batch).to_numpy()
            if str(batch) not in self.seen_batches_:
                unseen.add(str(batch))
                continue
            mu_b = self.mu_by_batch_[str(batch)]
            sig_b = self.sigma_by_batch_[str(batch)]
            out[idx] = (X_np[idx] - mu_b) / (sig_b + float(self.eps)) * self.sigma_star_ + self.mu_star_

        if unseen:
            warnings.warn(
                f"MeanVarAlignHarmonizer: unseen batch in transform() -> identity for batches: {sorted(unseen)[:5]}",
                category=UserWarning,
            )

        return pd.DataFrame(out.astype(np.float32), index=X.index, columns=self.feature_names)


@dataclass
class ComBatHarmonizer:
    batch_all: pd.Series
    discrete_covariates: pd.DataFrame | None = None
    continuous_covariates: pd.DataFrame | None = None

    variant: str = "fortin"  # combatlearn uses `method=...`
    parametric: bool = True
    mean_only: bool = False
    eps: float = 1e-8
    ref_batch: str | None = None
    covbat_cov_thresh: float = 0.9
    min_samples_per_batch_train: int = 2

    feature_names: list[str] | None = None
    core_model_: Any | None = None  # combatlearn.core.ComBatModel
    batch_levels_: list[str] | None = None
    disc_levels_by_col_: dict[str, list[str]] | None = None
    seen_batches_: set[str] | None = None

    def fit(self, X: pd.DataFrame) -> "ComBatHarmonizer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ComBatHarmonizer.fit expects a pandas DataFrame.")

        self.feature_names = list(X.columns)
        b_train = _safe_loc(self.batch_all, X.index, obj_name="batch_all").astype(str)
        # ComBat(Fortin/Chen) relies on per-batch variance with ddof=1, so each batch
        # must have at least 2 samples within the *training fold*. Otherwise delta_hat
        # becomes NaN and will poison shrinkage estimates.
        counts = b_train.value_counts()
        min_n = int(self.min_samples_per_batch_train)
        if min_n < 2:
            raise ValueError("ComBatHarmonizer.min_samples_per_batch_train must be >=2 (ddof=1 variance requirement).")
        good_batches = sorted(counts[counts >= min_n].index.astype(str).tolist())
        self.seen_batches_ = set(good_batches)

        if not good_batches:
            warnings.warn(
                "ComBatHarmonizer: no eligible batches with >=2 samples in this fold; "
                "falling back to identity.",
                category=UserWarning,
            )
            self.core_model_ = None
            self.batch_levels_ = []
            self.disc_levels_by_col_ = {}
            return self

        mask_fit = b_train.isin(self.seen_batches_)
        X_fit = X.loc[mask_fit]

        # Lazy import (avoid forcing combatlearn dependency unless method=combat).
        # Use core model so transform can accept explicit (batch/covariates) for deployable use.
        from combatlearn.core import ComBatModel  # type: ignore

        # combatlearn's Fortin implementation builds dummy matrices and then selects
        # columns for all training levels. If transform() is called on a subset that
        # does not include all training levels, pd.get_dummies would omit missing
        # columns and combatlearn would raise KeyError. Use categorical dtypes with
        # fixed categories (=training levels) so get_dummies includes all columns.
        batch_levels = list(good_batches)
        self.batch_levels_ = list(batch_levels)
        batch_fit = b_train.loc[X_fit.index].astype(str)
        batch_fit = pd.Series(
            pd.Categorical(batch_fit, categories=self.batch_levels_),
            index=X_fit.index,
            name="batch",
        )

        disc_fit = None
        disc_levels_by_col: dict[str, list[str]] = {}
        if self.discrete_covariates is not None:
            disc_fit = _safe_loc(self.discrete_covariates, X_fit.index, obj_name="discrete_covariates").copy()
            for col in disc_fit.columns:
                # Use training-fold levels (X.index) rather than X_fit-only levels to avoid
                # accidental "unseen covariate" at transform-time (still no val leakage).
                tr_levels = sorted(
                    set(
                        _safe_loc(
                            self.discrete_covariates[col],
                            X.index,
                            obj_name=f"discrete_covariates[{col!r}]",
                        )
                        .astype(str)
                        .tolist()
                    )
                )
                disc_levels_by_col[str(col)] = list(tr_levels)
                disc_fit[col] = pd.Categorical(disc_fit[col].astype(str), categories=disc_levels_by_col[str(col)])
        self.disc_levels_by_col_ = disc_levels_by_col

        self.core_model_ = ComBatModel(
            method=str(self.variant),
            parametric=bool(self.parametric),
            mean_only=bool(self.mean_only),
            reference_batch=self.ref_batch,
            eps=float(self.eps),
            covbat_cov_thresh=float(self.covbat_cov_thresh),
        )
        self.core_model_.fit(
            X_fit,
            batch=batch_fit,
            discrete_covariates=disc_fit,
            continuous_covariates=(
                None
                if self.continuous_covariates is None
                else _safe_loc(self.continuous_covariates, X_fit.index, obj_name="continuous_covariates")
            ),
        )
        return self

    def transform(
        self,
        X: pd.DataFrame,
        *,
        batch: pd.Series | None = None,
        discrete_covariates: pd.DataFrame | None = None,
        continuous_covariates: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self.feature_names is None or self.seen_batches_ is None:
            raise RuntimeError("ComBatHarmonizer is not fitted.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ComBatHarmonizer.transform expects a pandas DataFrame.")

        if list(X.columns) != list(self.feature_names):
            X = X.reindex(columns=self.feature_names)

        if self.core_model_ is None:
            return X.astype(np.float32)
        if self.batch_levels_ is None or self.disc_levels_by_col_ is None:
            raise RuntimeError("ComBatHarmonizer internal state missing (batch/discrete levels).")

        batch_src = self.batch_all if batch is None else batch
        b = _safe_loc(batch_src, X.index, obj_name="batch").astype(str)
        mask_seen = b.isin(self.seen_batches_)

        if not bool(mask_seen.all()):
            unseen = sorted({str(x) for x in b[~mask_seen].tolist()})
            warnings.warn(
                f"ComBatHarmonizer: unseen/insufficient-sample batch in transform() -> identity for batches: {unseen[:5]}",
                category=UserWarning,
            )
            out = X.copy()
            if bool(mask_seen.any()):
                X_seen = X.loc[mask_seen]
                b_seen = pd.Series(
                    pd.Categorical(b.loc[X_seen.index].astype(str), categories=self.batch_levels_),
                    index=X_seen.index,
                    name="batch",
                )
                disc_seen = None
                if discrete_covariates is not None:
                    disc_seen = _safe_loc(discrete_covariates, X_seen.index, obj_name="discrete_covariates").copy()
                elif self.discrete_covariates is not None:
                    disc_seen = _safe_loc(self.discrete_covariates, X_seen.index, obj_name="discrete_covariates").copy()
                if disc_seen is not None:
                    for col in disc_seen.columns:
                        levels = self.disc_levels_by_col_.get(str(col), None)
                        if levels is None:
                            raise ValueError(f"Unknown discrete covariate column at transform: {col!r}")
                        vals = disc_seen[col].astype(str)
                        bad = sorted({str(x) for x in vals[~vals.isin(levels)].tolist()})[:5]
                        if bad:
                            raise ValueError(
                                f"Discrete covariate {col!r} has unseen levels at transform: {bad}. "
                                "This would change the ComBat design matrix."
                            )
                        disc_seen[col] = pd.Categorical(vals, categories=levels)

                cont_seen = None
                if continuous_covariates is not None:
                    cont_seen = _safe_loc(continuous_covariates, X_seen.index, obj_name="continuous_covariates")
                elif self.continuous_covariates is not None:
                    cont_seen = _safe_loc(self.continuous_covariates, X_seen.index, obj_name="continuous_covariates")

                X_seen_t = self.core_model_.transform(
                    X_seen,
                    batch=b_seen,
                    discrete_covariates=disc_seen,
                    continuous_covariates=cont_seen,
                )
                out.loc[mask_seen] = X_seen_t.to_numpy(dtype=float)
            return out.astype(np.float32)

        b_all = pd.Series(pd.Categorical(b.astype(str), categories=self.batch_levels_), index=X.index, name="batch")
        disc_all = None
        if discrete_covariates is not None:
            disc_all = _safe_loc(discrete_covariates, X.index, obj_name="discrete_covariates").copy()
        elif self.discrete_covariates is not None:
            disc_all = _safe_loc(self.discrete_covariates, X.index, obj_name="discrete_covariates").copy()
        if disc_all is not None:
            for col in disc_all.columns:
                levels = self.disc_levels_by_col_.get(str(col), None)
                if levels is None:
                    raise ValueError(f"Unknown discrete covariate column at transform: {col!r}")
                vals = disc_all[col].astype(str)
                bad = sorted({str(x) for x in vals[~vals.isin(levels)].tolist()})[:5]
                if bad:
                    raise ValueError(
                        f"Discrete covariate {col!r} has unseen levels at transform: {bad}. "
                        "This would change the ComBat design matrix."
                    )
                disc_all[col] = pd.Categorical(vals, categories=levels)

        cont_all = None
        if continuous_covariates is not None:
            cont_all = _safe_loc(continuous_covariates, X.index, obj_name="continuous_covariates")
        elif self.continuous_covariates is not None:
            cont_all = _safe_loc(self.continuous_covariates, X.index, obj_name="continuous_covariates")

        X_t = self.core_model_.transform(
            X,
            batch=b_all,
            discrete_covariates=disc_all,
            continuous_covariates=cont_all,
        )
        return X_t.astype(np.float32)


def make_harmonizer(ctx: HarmonizationContext) -> Any:
    if not ctx.enabled or ctx.method == "none":
        return IdentityHarmonizer()
    if ctx.batch_all is None:
        raise ValueError("harmonization is enabled but batch_all is None (unexpected).")

    if ctx.method == "meanvar":
        return MeanVarAlignHarmonizer(batch_all=ctx.batch_all, eps=float(ctx.meanvar_eps))
    if ctx.method == "combat":
        return ComBatHarmonizer(
            batch_all=ctx.batch_all,
            discrete_covariates=ctx.discrete_covariates_all,
            continuous_covariates=ctx.continuous_covariates_all,
            variant=str(ctx.combat_variant),
            parametric=bool(ctx.combat_parametric),
            mean_only=bool(ctx.combat_mean_only),
            eps=float(ctx.combat_eps),
            ref_batch=ctx.combat_ref_batch,
            covbat_cov_thresh=float(ctx.combat_covbat_cov_thresh),
            min_samples_per_batch_train=int(ctx.combat_min_samples_per_batch_train),
        )
    raise ValueError(f"Unsupported harmonization.method: {ctx.method!r}")


@dataclass
class HarmonizedPreprocessor:
    """
    A combined transformer used by train/explain/analyze:
        X_raw -> harmonizer.transform() -> preprocessor.transform() -> np.ndarray
    """

    harmonizer: Any
    preprocessor: Any

    def transform(self, X_raw: pd.DataFrame) -> np.ndarray:
        X_h = self.harmonizer.transform(X_raw)
        return self.preprocessor.transform(X_h)

    def transform_with_meta(self, X_raw: pd.DataFrame, *, cfg: dict[str, Any], meta: pd.DataFrame) -> np.ndarray:
        """
        Deployable transform API.

        Notes:
        - Batch/covariates are constructed from (cfg, meta) and do NOT rely on the training-run index.
        - This is required to transform a new dataset (new index) using a pickled preprocessor.
        """
        if not X_raw.index.equals(meta.index):
            raise ValueError(
                "transform_with_meta requires X_raw.index == meta.index for safe batch/covariate alignment. "
                "Do NOT reset_index(drop=True) or rebuild DataFrames before calling this method."
            )
        ctx = build_harmonization_context(cfg=cfg, meta=meta)
        X_h = self.harmonizer.transform(
            X_raw,
            batch=ctx.batch_all,
            discrete_covariates=ctx.discrete_covariates_all,
            continuous_covariates=ctx.continuous_covariates_all,
        )
        return self.preprocessor.transform(X_h)
