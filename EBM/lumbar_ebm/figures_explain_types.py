from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldContrib:
    fold: int
    feature_names: list[str]
    task_names: list[str]
    X_raw: pd.DataFrame | None
    # Per-feature additive contributions to s_raw (excluding intercept), shape (n_samples, n_feat).
    f_raw: np.ndarray
    f_mean: np.ndarray  # (n_feat,)
    f_centered: np.ndarray  # (n_samples, n_feat)
    intercept: float
