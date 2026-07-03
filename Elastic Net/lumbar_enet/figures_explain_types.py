from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldContrib:
    fold: int
    feature_names: list[str]
    task_names: list[str]
    X_raw: pd.DataFrame
    # f_raw is the cumulative per-feature contribution (before centering), shape (n_samples, n_feat).
    f_raw: np.ndarray
    f_mean: np.ndarray  # (n_feat,)
    f_centered: np.ndarray  # (n_samples, n_feat)
    # last-layer task weights/bias (main effects), aligned to feature_names.
    w_last: np.ndarray  # (n_tasks, n_feat)
    b_last: np.ndarray  # (n_tasks,)
