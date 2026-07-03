from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class RankGaussZScore:
    """
    Fold-inner preprocessing required by the updated plan (方案1(新改).md 4.3.0):

      1) Rank-based Gaussianization (a.k.a. normal score transform):
           x_tilde = Phi^{-1}((rank(x_raw) - 1/2) / n_train)
         For inference points, rank(x_raw) is approximated by the empirical CDF
         of the training fold (via searchsorted on sorted training values).

      2) Z-score standardization on the transformed values:
           x = (x_tilde - mu) / sigma

    Notes:
    - All parameters are fit on the training fold only; val/test reuse them.
    - Input columns must be numeric; NaNs should be handled upstream.
    """

    feature_names: list[str] | None = None
    sorted_train_: list[np.ndarray] | None = None  # per-feature sorted training values
    n_train_: int | None = None
    mu_: np.ndarray | None = None
    std_: np.ndarray | None = None
    n_jobs: int = 1  # 0 = all cores (joblib convention); uses threads to avoid copies on Windows

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

        p = int(X_np.shape[1])
        nj = int(self.n_jobs) if self.n_jobs is not None else 1
        if nj == 0:
            nj = -1  # joblib convention: all cores

        if nj == 1 or p <= 1:
            out = np.empty_like(X_np, dtype=np.float64)
            for j, sorted_col in enumerate(self.sorted_train_):
                # Empirical CDF via insertion index.
                # count = number of training samples <= x.
                count = np.searchsorted(sorted_col, X_np[:, j], side="right")
                count = np.clip(count, 1, n_train)  # avoid 0 -> negative u, keep within (0,1)
                u = (count - 0.5) / n_train
                out[:, j] = norm.ppf(u)
            return out

        from joblib import Parallel, delayed

        def _one(j: int) -> np.ndarray:
            sorted_col = self.sorted_train_[j]
            count = np.searchsorted(sorted_col, X_np[:, j], side="right")
            count = np.clip(count, 1, n_train)
            u = (count - 0.5) / n_train
            return norm.ppf(u)

        cols = Parallel(n_jobs=nj, backend="threading", prefer="threads")(delayed(_one)(j) for j in range(p))
        return np.stack(cols, axis=1).astype(np.float64, copy=False)


@dataclass
class ZScore:
    """Fold-inner z-score standardization (fit on train fold only)."""

    feature_names: list[str] | None = None
    mu_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "ZScore":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ZScore.fit expects a pandas DataFrame.")
        self.feature_names = list(X.columns)
        X_np = X.to_numpy(dtype=float, copy=True)
        mu = X_np.mean(axis=0)
        std = X_np.std(axis=0)
        std[std == 0] = 1.0
        self.mu_ = mu.astype(np.float32)
        self.std_ = std.astype(np.float32)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_names is None or self.mu_ is None or self.std_ is None:
            raise RuntimeError("ZScore is not fitted.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ZScore.transform expects a pandas DataFrame.")

        if list(X.columns) != list(self.feature_names):
            X = X.reindex(columns=self.feature_names)
        X_np = X.to_numpy(dtype=float, copy=True)
        X_z = (X_np - self.mu_) / self.std_
        return X_z.astype(np.float32)
