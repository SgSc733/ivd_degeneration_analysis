from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lumbar_enet.preprocess import ElasticNetDesignMatrix


@dataclass
class FittedElasticNet:
    """
    Minimal fitted model container for this workspace.

    Stores:
    - fold-inner fitted preprocessor/design builder
    - coefficients on the design matrix columns
    - unpenalized intercept
    """

    pre: ElasticNetDesignMatrix
    coef_: np.ndarray  # shape (p_design,)
    intercept_: float
    l1_ratio: float
    lambda_: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_design = self.pre.transform(X).astype(np.float64, copy=False)
        return (float(self.intercept_) + X_design @ self.coef_.astype(np.float64)).astype(np.float64)

    def predict_design(self, X_design: np.ndarray) -> np.ndarray:
        X_design = np.asarray(X_design, dtype=np.float64)
        return (float(self.intercept_) + X_design @ self.coef_.astype(np.float64)).astype(np.float64)

    def get_feature_names_out(self) -> list[str]:
        return self.pre.get_feature_names_out()

    def get_penalty_factors(self) -> np.ndarray:
        return self.pre.get_penalty_factors()

