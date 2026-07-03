from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


def _safe_name(s: str) -> str:
    # Windows-friendly filename/column name: avoid '/', '-', etc.
    return (
        str(s)
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


@dataclass(frozen=True)
class BlockSlices:
    base: slice
    seg_intercept: slice | None
    interactions: slice | None


@dataclass
class RankGaussZScore:
    """
    Fold-inner preprocessing (strict plan-notation):

      1) Rank-based Gaussianization:
           x_tilde = Phi^{-1}((rank(x_raw) - 1/2) / n_train)
         For inference points, rank(x_raw) is approximated by the empirical CDF
         of the training fold (via searchsorted on sorted training values).

      2) Z-score standardization on the transformed values:
           x = (x_tilde - mu) / sigma

    Notes:
    - All parameters are fit on the training fold only; val/test reuse them.
    - NaNs must be handled upstream (this class assumes numeric ndarray input).
    """

    feature_names: list[str] | None = None
    sorted_train_: list[np.ndarray] | None = None  # per-feature sorted training values
    n_train_: int | None = None
    mu_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "RankGaussZScore":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RankGaussZScore.fit expects a pandas DataFrame.")

        self.feature_names = list(X.columns)
        X_np = X.to_numpy(dtype=float, copy=True)
        n, p = X_np.shape
        if n <= 1:
            raise ValueError("Need at least 2 training samples for RankGaussZScore.")

        self.n_train_ = int(n)
        self.sorted_train_ = [np.sort(X_np[:, j].astype(float, copy=False)) for j in range(p)]

        X_tilde = self._rank_gauss_np(X_np)
        mu = X_tilde.mean(axis=0)
        std = X_tilde.std(axis=0)
        std[std == 0] = 1.0

        self.mu_ = mu.astype(np.float32)
        self.std_ = std.astype(np.float32)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_names is None or self.sorted_train_ is None or self.n_train_ is None:
            raise RuntimeError("RankGaussZScore is not fitted.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RankGaussZScore.transform expects a pandas DataFrame.")

        # Align by column names (safer than positional assumptions).
        if list(X.columns) != list(self.feature_names):
            X = X.reindex(columns=self.feature_names)

        X_np = X.to_numpy(dtype=float, copy=True)
        X_tilde = self._rank_gauss_np(X_np)
        X_z = (X_tilde - self.mu_) / self.std_
        return X_z.astype(np.float32)

    def _rank_gauss_np(self, X_np: np.ndarray) -> np.ndarray:
        assert self.sorted_train_ is not None and self.n_train_ is not None
        n_train = self.n_train_

        out = np.empty_like(X_np, dtype=np.float64)
        u_min = 0.5 / n_train
        u_max = 1.0 - u_min
        for j, sorted_col in enumerate(self.sorted_train_):
            left = np.searchsorted(sorted_col, X_np[:, j], side="left")
            right = np.searchsorted(sorted_col, X_np[:, j], side="right")
            u = (left + right).astype(np.float64) / (2.0 * n_train)
            u = np.clip(u, u_min, u_max)
            out[:, j] = norm.ppf(u)
        return out


class ElasticNetDesignMatrix:
    """
    Build a design matrix for the Elastic Net baseline with:
    - strict Rank->Phi^{-1} Gaussianization + Z-score (fit on training fold only)
    - optional segment intercepts (gamma) and segment-feature interactions (delta)

    IMPORTANT:
    This class does NOT "scale columns to approximate penalty factors".
    Instead, it produces an explicit per-column penalty factor vector v_j so that
    the solver can implement the exact objective described in Elastic Net.md:

        L = (1/2n)||y - (b0 + X beta)||^2
            + lambda * [ alpha * sum_j v_j |beta_j| + (1-alpha)/2 * sum_j v_j beta_j^2 ]
    """

    def __init__(
        self,
        *,
        segment_col: str = "disc_level",
        segment_levels: list[str] | None = None,
        segment_reference: str = "L3-L4",
        include_segment_intercept: bool = True,
        include_segment_interactions: bool = True,
        segment_penalty_factor: float = 0.0,
        interaction_penalty_factor: float = 2.0,
    ) -> None:
        self.segment_col = segment_col
        self.segment_levels = segment_levels or ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
        self.segment_reference = segment_reference
        self.include_segment_intercept = bool(include_segment_intercept)
        self.include_segment_interactions = bool(include_segment_interactions)
        self.segment_penalty_factor = float(segment_penalty_factor)
        self.interaction_penalty_factor = float(interaction_penalty_factor)

    def fit(self, X: pd.DataFrame) -> "ElasticNetDesignMatrix":
        df = self._as_df(X)
        self.base_feature_names_ = [c for c in df.columns if c != self.segment_col]

        if self.segment_reference not in self.segment_levels:
            raise ValueError(
                f"segment_reference '{self.segment_reference}' must be in segment_levels: {self.segment_levels}"
            )
        if self.segment_penalty_factor < 0:
            raise ValueError("segment_penalty_factor must be >= 0 (0 means unpenalized gamma).")
        if self.interaction_penalty_factor <= 0:
            raise ValueError("interaction_penalty_factor must be > 0.")

        X_base = df[self.base_feature_names_].apply(pd.to_numeric, errors="coerce")
        self.impute_median_ = X_base.median(numeric_only=True)
        X_base = X_base.fillna(self.impute_median_)

        self.rgz_ = RankGaussZScore().fit(X_base)

        # Precompute design feature names + penalty factors.
        seg_levels = [lvl for lvl in self.segment_levels if lvl != self.segment_reference]
        names: list[str] = []
        v: list[float] = []

        # Base features: v=1
        names.extend(self.base_feature_names_)
        v.extend([1.0] * len(self.base_feature_names_))

        seg_names = [f"gamma_{_safe_name(lvl)}" for lvl in seg_levels] if self.include_segment_intercept else []
        if seg_names:
            names.extend(seg_names)
            v.extend([float(self.segment_penalty_factor)] * len(seg_names))

        int_names: list[str] = []
        if self.include_segment_interactions:
            for lvl in seg_levels:
                for feat in self.base_feature_names_:
                    int_names.append(f"delta_{_safe_name(lvl)}__{_safe_name(feat)}")
        if int_names:
            names.extend(int_names)
            v.extend([float(self.interaction_penalty_factor)] * len(int_names))

        self.feature_names_out_ = names
        self.penalty_factors_ = np.asarray(v, dtype=np.float64)

        base_slice = slice(0, len(self.base_feature_names_))
        seg_slice = None
        int_slice = None
        cur = len(self.base_feature_names_)
        if self.include_segment_intercept:
            seg_slice = slice(cur, cur + len(seg_names))
            cur += len(seg_names)
        if self.include_segment_interactions:
            int_slice = slice(cur, cur + len(int_names))
        self.block_slices_ = BlockSlices(base=base_slice, seg_intercept=seg_slice, interactions=int_slice)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = self._as_df(X)
        if not hasattr(self, "base_feature_names_"):
            raise RuntimeError("ElasticNetDesignMatrix must be fitted before transform().")

        X_base = df[self.base_feature_names_].apply(pd.to_numeric, errors="coerce")
        X_base = X_base.fillna(self.impute_median_)
        X_arr = self.rgz_.transform(X_base).astype(np.float64, copy=False)  # (n, p)

        seg = df[self.segment_col].astype(str).to_numpy()
        seg_levels = [lvl for lvl in self.segment_levels if lvl != self.segment_reference]
        seg_dummy = np.column_stack([(seg == lvl).astype(float) for lvl in seg_levels]) if seg_levels else None

        parts: list[np.ndarray] = [X_arr]

        if self.include_segment_intercept and seg_dummy is not None and seg_dummy.size:
            parts.append(seg_dummy.astype(np.float64, copy=False))

        if self.include_segment_interactions and seg_dummy is not None and seg_dummy.size:
            inter = (seg_dummy[:, :, None] * X_arr[:, None, :]).reshape(len(X_arr), -1)
            parts.append(inter.astype(np.float64, copy=False))

        out = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        return out

    def get_feature_names_out(self) -> list[str]:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("ElasticNetDesignMatrix must be fitted before calling get_feature_names_out().")
        return list(self.feature_names_out_)

    def get_penalty_factors(self) -> np.ndarray:
        if not hasattr(self, "penalty_factors_"):
            raise RuntimeError("ElasticNetDesignMatrix must be fitted before calling get_penalty_factors().")
        return self.penalty_factors_.copy()

    def get_block_slices(self) -> BlockSlices:
        if not hasattr(self, "block_slices_"):
            raise RuntimeError("ElasticNetDesignMatrix must be fitted before calling get_block_slices().")
        return self.block_slices_

    def _as_df(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self.segment_col not in X.columns:
            raise ValueError(f"Input DataFrame must contain segment_col '{self.segment_col}'.")
        return X
