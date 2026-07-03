from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FoldShapContrib:
    fold: int
    feature_names: list[str]
    # phi_raw is the per-feature contribution (before centering), shape (n_samples, n_feat).
    phi_raw: np.ndarray
    phi_mean: np.ndarray  # (n_feat,)
    phi_centered: np.ndarray  # (n_samples, n_feat)
    feature_values: np.ndarray  # unprocessed feature values in feature_names order, shape (n_samples, n_feat)

