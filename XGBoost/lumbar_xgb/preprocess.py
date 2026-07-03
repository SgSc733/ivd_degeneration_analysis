from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class RankGaussZScore:
    """
    Fold-inner preprocessing used in the main ProtoNAM pipeline (方案1.md 4.3.0):

    1) Rank-based Gaussianization:
         x_tilde = Phi^{-1}((rank(x_raw) - 1/2) / n_train)
       For inference points, rank(x_raw) is approximated by the empirical CDF
       of the training fold (via searchsorted on sorted training values).

    2) Z-score:
         x = (x_tilde - mu) / sigma
    """

    feature_names: list[str] | None = None
    sorted_train_: list[np.ndarray] | None = None
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
        for j, sorted_col in enumerate(self.sorted_train_):
            count = np.searchsorted(sorted_col, X_np[:, j], side="right")
            count = np.clip(count, 1, n_train)
            u = (count - 0.5) / n_train
            out[:, j] = norm.ppf(u)
        return out

