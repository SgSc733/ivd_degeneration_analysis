import logging
from typing import Dict

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor

from calculator.base_calculator import GuiLoggerProxy


class CPTensorFeatures:

    def __init__(
        self,
        rank: int = 8,
        max_iter: int = 1000,
        tol: float = 1e-4,
        epsilon: float = 1e-6,
        top_components: int = 3,
        random_state: int = 0,
        logger_callback=None,
        debug_mode: bool = False,
    ):
        self.rank = int(rank)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.epsilon = float(epsilon)
        self.top_components = int(top_components)
        self.random_state = int(random_state)

        if logger_callback:
            self.logger = GuiLoggerProxy(logger_callback, debug_mode=debug_mode)
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.debug_mode = debug_mode

    def _fit_cp(self, tensor_data: np.ndarray) -> tuple[np.ndarray, list]:

        tensor = tl.tensor(tensor_data.astype(np.float64))
        weights, factors = parafac(
            tensor,
            rank=self.rank,
            n_iter_max=self.max_iter,
            tol=self.tol,
            init="svd",
            normalize_factors=True,
            random_state=self.random_state,
        )
        return np.array(weights, dtype=np.float64), [np.array(f, dtype=np.float64) for f in factors]

    def extract_features(self, roi_tensor: np.ndarray) -> Dict[str, float]:

        if roi_tensor.ndim != 3:
            raise ValueError(
                f"CPTensorFeatures 需要 3D 张量，当前形状为 {roi_tensor.shape}"
            )

        X = roi_tensor.astype(np.float64)
        if not np.any(np.isfinite(X)) or np.allclose(X, 0):
            self.logger.warning("CPTensorFeatures: ROI 张量为空或无有效数值，返回空特征。")
            return {}

        try:
            lambdas, factors = self._fit_cp(X)
        except Exception as e:
            self.logger.error(f"CP 分解失败: {e}")
            raise

        features: Dict[str, float] = {}

        eps = self.epsilon
        lam = np.array(lambdas, dtype=np.float64).ravel()
        if lam.size == 0:
            return {}

        order = np.argsort(-np.abs(lam))
        lam_sorted = lam[order]
        A = np.array(factors[0], dtype=np.float64)[:, order]
        B = np.array(factors[1], dtype=np.float64)[:, order]
        C = np.array(factors[2], dtype=np.float64)[:, order]

        R = lam_sorted.size

        for r in range(R):
            features[f"tensor_cp_lambda_k{r+1}"] = float(lam_sorted[r])

        E_lambda = float(np.sum(lam_sorted**2))
        if E_lambda <= 0:
            energy_ratios = np.zeros_like(lam_sorted)
        else:
            energy_ratios = (lam_sorted**2) / (E_lambda + eps)

        for r in range(R):
            features[f"tensor_cp_energy_ratio_k{r+1}"] = float(energy_ratios[r])

        denom = float(np.sum(lam_sorted**4) + eps)
        R_eff = float((E_lambda**2) / denom) if denom > 0 else 0.0
        features["tensor_cp_R_eff"] = R_eff

        S_lambda = float(np.sum(np.abs(lam_sorted)))
        features["tensor_cp_S_lambda"] = S_lambda
        features["tensor_cp_lambda_l2_norm"] = float(np.sqrt(max(E_lambda, 0.0)))

        K_eff = min(self.top_components, R)
        mode_factors = [A, B, C]

        for r in range(K_eff):
            for mode_idx, F in enumerate(mode_factors):
                vec = F[:, r]
                p = vec**2
                s = float(np.sum(p))
                if s <= 0:
                    p = np.ones_like(p) / p.size
                else:
                    p = p / s

                H = -float(np.sum(p * np.log(p + eps)))
                G = float(np.sum(p**2))

                m = mode_idx + 1
                features[f"tensor_cp_entropy_mode{m}_k{r+1}"] = H
                features[f"tensor_cp_gini_mode{m}_k{r+1}"] = G

        try:
            X_hat = cp_to_tensor((lambdas, factors))
            X_hat_np = tl.to_numpy(X_hat)
            diff = X - X_hat_np
            num = float(np.sum(diff**2))
            denom_r = float(np.sum(X**2)) + 1e-12
            r_cp = num / denom_r
        except Exception:
            r_cp = float("nan")

        msg_cn = f"CP分解误差：r_CP={r_cp:.4e}"
        if isinstance(self.logger, GuiLoggerProxy):
            self.logger.callback(msg_cn)
        else:
            self.logger.info(msg_cn)

        return features
