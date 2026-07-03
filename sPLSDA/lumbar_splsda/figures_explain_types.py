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
    # f_raw is the cumulative per-feature "shape output" (before centering), shape (n_samples, n_feat).
    # For sPLS-DA we define it on the severity axis as: f_raw[:,j] = beta_j * x_j (x after fold preprocessor).
    f_raw: np.ndarray
    f_mean: np.ndarray  # (n_feat,)
    f_centered: np.ndarray  # (n_samples, n_feat)
    # task weights/bias (kept to match the ProtoNAM figure logic). For sPLS-DA, we use identity weights.
    w_last: np.ndarray  # (n_tasks, n_feat)
    b_last: np.ndarray  # (n_tasks,)
