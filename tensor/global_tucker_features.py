import logging
from typing import Dict, Optional

import numpy as np
import tensorly as tl
from tensorly.tenalg import multi_mode_dot

from calculator.base_calculator import GuiLoggerProxy
from .roi_utils import mode_n_unfold


class GlobalTuckerTensorFeatures:

    def __init__(
        self,
        energy_threshold: float = 0.95,
        k_singular_values: int = 10,
        logger_callback=None,
        debug_mode: bool = False,
    ):
        self.eta = energy_threshold
        self.k_singular_values = k_singular_values

        if logger_callback:
            self.logger = GuiLoggerProxy(logger_callback, debug_mode=debug_mode)
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.debug_mode = debug_mode

    def _select_rank(self, singular_values: np.ndarray) -> int:

        if singular_values.size == 0:
            return 0
        energy = singular_values**2
        total = float(energy.sum())
        if total <= 0:
            return int(min(10, singular_values.size))

        cumulative = np.cumsum(energy) / total
        idx = int(np.searchsorted(cumulative, self.eta) + 1)
        return max(1, min(idx, singular_values.size))

    def extract_features(self, roi_tensor: np.ndarray) -> Dict[str, float]:

        if roi_tensor.ndim != 3:
            raise ValueError(f"GlobalTuckerTensorFeatures 需要 3D 张量，收到形状 {roi_tensor.shape}")

        X = roi_tensor.astype(np.float64)
        dims = X.shape

        U_list = []
        S_list = []
        ranks = []
        features: Dict[str, float] = {}

        for mode in range(3):
            Xn = mode_n_unfold(X, mode)
            U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
            U_list.append(U)
            S_list.append(S)

            Rn = self._select_rank(S)
            ranks.append(Rn)

            K = min(self.k_singular_values, S.size)
            if K == 0:
                continue

            mode_idx = mode + 1
            energy_total = float((S**2).sum()) if S.size > 0 else 0.0

            for k in range(K):
                sv = float(S[k])
                features[f"tensor_tucker_sigma_mode{mode_idx}_k{k+1}"] = sv
                if energy_total > 0:
                    e_k = float((S[k] ** 2) / energy_total)
                else:
                    e_k = 0.0
                features[f"tensor_tucker_energy_mode{mode_idx}_k{k+1}"] = e_k

        for i, Rn in enumerate(ranks, start=1):
            features[f"tensor_tucker_rank_R{i}"] = float(Rn)

        if any(Rn <= 0 for Rn in ranks):
            self.logger.warning("某一模的有效秩为 0，Tucker core 无法构造，返回奇异值相关特征。")
            return features

        U_trunc = [U_list[i][:, : ranks[i]] for i in range(3)]

        X_tl = tl.tensor(X)
        S_tilde = multi_mode_dot(
            X_tl, [U_trunc[0].T, U_trunc[1].T, U_trunc[2].T], modes=[0, 1, 2]
        )
        S_arr = tl.to_numpy(S_tilde)
        core_energy = float(np.sum(S_arr**2))
        features["tensor_tucker_core_energy"] = core_energy

        if core_energy > 0:
            p1 = np.sum(S_arr**2, axis=(1, 2)) / core_energy
            p2 = np.sum(S_arr**2, axis=(0, 2)) / core_energy
            p3 = np.sum(S_arr**2, axis=(0, 1)) / core_energy

            for i, val in enumerate(p1):
                features[f"tensor_tucker_p1_i{i+1}"] = float(val)
            for j, val in enumerate(p2):
                features[f"tensor_tucker_p2_j{j+1}"] = float(val)
            for k, val in enumerate(p3):
                features[f"tensor_tucker_p3_k{k+1}"] = float(val)

        X_hat = multi_mode_dot(
            S_tilde, [U_trunc[0], U_trunc[1], U_trunc[2]], modes=[0, 1, 2]
        )
        X_hat_np = tl.to_numpy(X_hat)
        diff_np = X - X_hat_np
        num = float(np.sum(diff_np ** 2))
        denom = float(np.sum(X ** 2))
        r = num / denom if denom > 0 else 0.0
        features["tensor_tucker_reconstruction_error_ratio"] = r

        if isinstance(self.logger, GuiLoggerProxy):
            self.logger.info(
                f"Tucker特征: 形状={dims}, R=({ranks[0]}, {ranks[1]}, {ranks[2]}), 重构误差比例 r={r:.4f}"
            )
        else:
            self.logger.info(
                f"Tucker features computed: shape={dims}, ranks={ranks}, r={r:.4f}"
            )

        return features
