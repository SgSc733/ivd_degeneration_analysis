import logging
from typing import Dict, List, Tuple

import numpy as np

from calculator.base_calculator import GuiLoggerProxy
from .roi_utils import mode_n_unfold, mode_n_fold


class PatchTensorFeatures:

    def __init__(
        self,
        patch_size: int = 4,
        similar_patches: int = 64,
        search_window: int = 15,
        internal_iterations: int = 50,
        external_iterations: int = 2,
        epsilon: float = 1e-16,
        alpha_feedback: float = 0.1,
        beta_noise: float = 0.3,
        max_patch_groups: int = 64,
        max_singular_values: int = 10,
        logger_callback=None,
        debug_mode: bool = False,
    ):
        self.patch_size = int(patch_size)
        self.similar_patches = int(similar_patches)
        self.search_window = int(search_window)
        self.internal_iterations = int(internal_iterations)
        self.external_iterations = int(external_iterations)
        self.epsilon = float(epsilon)
        self.alpha_feedback = float(alpha_feedback)
        self.beta_noise = float(beta_noise)
        self.max_patch_groups = int(max_patch_groups)
        self.max_singular_values = int(max_singular_values)

        if logger_callback:
            self.logger = GuiLoggerProxy(logger_callback, debug_mode=debug_mode)
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.debug_mode = debug_mode

        self.theta = np.ones(4, dtype=np.float64) / 4.0
        self.mu0 = 1.0
        self.rho = 1.05
        self.convergence_eps = 1e-4


    def _estimate_noise_sigma(self, roi_raw: np.ndarray, roi_mask: np.ndarray) -> float:

        background = roi_raw[roi_mask == 0]
        if background.size < 50:
            background = roi_raw.flatten()

        median = np.median(background)
        mad = np.median(np.abs(background - median))
        if mad < 1e-6:
            mad = np.std(background)
        sigma_hat = 1.4826 * mad
        sigma_n = float(self.beta_noise * sigma_hat)
        if isinstance(self.logger, GuiLoggerProxy):
            self.logger.info(f"估计噪声σ_n ≈ {sigma_n:.4f}")
        return sigma_n

    def _build_patch_coords(self, roi_mask: np.ndarray) -> List[Tuple[int, int, int]]:

        m = self.patch_size
        Z, Y, X = roi_mask.shape
        coords: List[Tuple[int, int, int]] = []

        step = max(1, m - 1)

        for z in range(0, max(Z - m + 1, 1), step):
            for y in range(0, max(Y - m + 1, 1), step):
                for x in range(0, max(X - m + 1, 1), step):
                    patch_mask = roi_mask[z : z + m, y : y + m, x : x + m]
                    if np.any(patch_mask):
                        coords.append((z, y, x))

        if not coords:
            return []

        rng = np.random.default_rng(0)
        if len(coords) > self.max_patch_groups:
            coords = list(rng.choice(coords, size=self.max_patch_groups, replace=False))

        return coords

    def _collect_similar_patches(
        self,
        volume: np.ndarray,
        roi_mask: np.ndarray,
        ref_coord: Tuple[int, int, int],
    ) -> np.ndarray:

        m = self.patch_size
        n_sim = self.similar_patches
        s = self.search_window
        Z, Y, X = volume.shape

        z0, y0, x0 = ref_coord
        cz = z0 + m // 2
        cy = y0 + m // 2
        cx = x0 + m // 2

        half_s = s // 2
        zc_min = max(m // 2, cz - half_s)
        zc_max = min(Z - m // 2 - 1, cz + half_s)
        yc_min = max(m // 2, cy - half_s)
        yc_max = min(Y - m // 2 - 1, cy + half_s)
        xc_min = max(m // 2, cx - half_s)
        xc_max = min(X - m // 2 - 1, cx + half_s)

        ref_patch = volume[z0 : z0 + m, y0 : y0 + m, x0 : x0 + m]
        ref_vec = ref_patch.reshape(-1)

        candidates: List[Tuple[float, int, int, int]] = []

        for cz_c in range(zc_min, zc_max + 1):
            z = cz_c - m // 2
            for cy_c in range(yc_min, yc_max + 1):
                y = cy_c - m // 2
                for cx_c in range(xc_min, xc_max + 1):
                    x = cx_c - m // 2

                    patch_mask = roi_mask[z : z + m, y : y + m, x : x + m]
                    if not np.any(patch_mask):
                        continue

                    patch = volume[z : z + m, y : y + m, x : x + m]
                    vec = patch.reshape(-1)
                    dist = float(np.linalg.norm(vec - ref_vec))
                    candidates.append((dist, z, y, x))

        if not candidates:
            return ref_patch[..., np.newaxis]

        candidates.sort(key=lambda t: t[0])
        top = candidates[: max(1, n_sim)]

        patches = []
        for _, z, y, x in top:
            patch = volume[z : z + m, y : y + m, x : x + m]
            patches.append(patch)

        GY = np.stack(patches, axis=-1)
        return GY

    def _weighted_log_shrink(self, mat: np.ndarray, theta_i: float, mu: float) -> np.ndarray:

        U, s, Vt = np.linalg.svd(mat, full_matrices=False)
        eps = self.epsilon

        delta = s
        c1 = delta - eps
        c2 = (delta + eps) ** 2 - 4.0 * theta_i / max(mu, 1e-8)

        pi = np.zeros_like(delta)
        mask = c2 >= 0
        pi[mask] = np.maximum(0.0, (c1[mask] + np.sqrt(c2[mask])) / 2.0)

        S_new = np.diag(pi)
        return U @ S_new @ Vt

    def _admm_logsum(self, GY: np.ndarray) -> np.ndarray:

        GX = GY.astype(np.float64).copy()
        Q = [np.zeros_like(GX) for _ in range(4)]
        mu = self.mu0

        for t in range(self.internal_iterations):
            O_list = []
            for mode in range(4):
                T_i = GX - Q[mode] / mu
                T_mat = mode_n_unfold(T_i, mode)
                O_mat = self._weighted_log_shrink(T_mat, self.theta[mode], mu)
                O_i = mode_n_fold(O_mat, mode, GX.shape)
                O_list.append(O_i)

            GX_prev = GX.copy()
            GX = 0.25 * sum(O_list[i] + Q[i] / mu for i in range(4))

            for i in range(4):
                Q[i] = Q[i] + mu * (O_list[i] - GX)

            mu *= self.rho

            num = float(np.sum((GX - GX_prev) ** 2))
            denom = float(np.sum(GX_prev**2)) + 1e-12
            if num / denom <= self.convergence_eps:
                break

        return GX


    def extract_features(self, roi_raw: np.ndarray, roi_mask: np.ndarray) -> Dict[str, float]:

        if roi_raw.shape != roi_mask.shape:
            raise ValueError("PatchTensorFeatures: ROI 图像与掩码形状不一致")

        if not np.any(roi_mask):
            return {}

        roi_raw = roi_raw.astype(np.float64)
        roi_mask = (roi_mask > 0).astype(np.uint8)

        sigma_n = self._estimate_noise_sigma(roi_raw, roi_mask)
        Y_raw = np.sqrt(roi_raw**2 + sigma_n**2)

        ref_coords = self._build_patch_coords(roi_mask)
        if not ref_coords:
            self.logger.warning("PatchTensorFeatures: 未找到有效的参考 patch，返回空特征。")
            return {}

        Z, Y_dim, X_dim = roi_raw.shape

        X_current = Y_raw.copy()
        Y_iter = Y_raw.copy()

        delta_all: List[List[np.ndarray]] = [[], [], [], []]
        omega_all: List[List[np.ndarray]] = [[], [], [], []]
        L_all: List[List[float]] = [[], [], [], []]
        Rlog_all: List[List[float]] = [[], [], [], []]
        eNL_list: List[float] = []

        for ext_it in range(max(1, self.external_iterations)):
            accum = np.zeros_like(Y_iter, dtype=np.float64)
            weight = np.zeros_like(Y_iter, dtype=np.float64)

            is_last_iter = (ext_it == self.external_iterations - 1)

            for idx, coord in enumerate(ref_coords):
                try:
                    GY = self._collect_similar_patches(Y_iter, roi_mask, coord)
                    GX = self._admm_logsum(GY)

                    m = self.patch_size
                    z0, y0, x0 = coord
                    patch_hat = GX[:, :, :, 0]
                    accum[z0 : z0 + m, y0 : y0 + m, x0 : x0 + m] += patch_hat
                    weight[z0 : z0 + m, y0 : y0 + m, x0 : x0 + m] += 1.0

                    if is_last_iter:
                        num = float(np.sum((GY - GX) ** 2))
                        denom = float(np.sum(GY**2)) + 1e-12
                        eNL = num / denom
                        eNL_list.append(eNL)

                        for mode in range(4):
                            GXn = mode_n_unfold(GX, mode)
                            _, s, _ = np.linalg.svd(GXn, full_matrices=False)
                            if s.size == 0:
                                continue

                            delta_all[mode].append(s)
                            omega = 1.0 / (s + self.epsilon)
                            omega_all[mode].append(omega)

                            L_i = float(np.sum(np.log(s + self.epsilon)))
                            L_all[mode].append(L_i)
                            Rlog_all[mode].append(L_i)

                except Exception as e:
                    self.logger.warning(f"PatchTensorFeatures: 第 {idx} 个 patch 组处理失败: {e}")
                    continue

            valid_mask = weight > 0
            X_new = X_current.copy()
            X_new[valid_mask] = accum[valid_mask] / np.maximum(weight[valid_mask], 1.0)

            if not is_last_iter:
                Y_iter = X_new + self.alpha_feedback * (Y_raw - X_new)
                X_current = X_new
            else:
                X_current = X_new

        features: Dict[str, float] = {}

        if eNL_list:
            features["tensor_patch_eNL_mean"] = float(np.mean(eNL_list))
            features["tensor_patch_eNL_std"] = float(np.std(eNL_list))

        for mode in range(4):
            mode_idx = mode + 1
            if delta_all[mode]:
                K = self.max_singular_values
                for k in range(K):
                    vals = []
                    for arr in delta_all[mode]:
                        if k < arr.size:
                            vals.append(arr[k])
                    if vals:
                        features[
                            f"tensor_patch_delta_mode{mode_idx}_k{k+1}_mean"
                        ] = float(np.mean(vals))
                        features[
                            f"tensor_patch_delta_mode{mode_idx}_k{k+1}_std"
                        ] = float(np.std(vals))

            if omega_all[mode]:
                all_omega = np.concatenate(omega_all[mode])
                features[f"tensor_patch_omega_mode{mode_idx}_mean"] = float(
                    np.mean(all_omega)
                )
                features[f"tensor_patch_omega_mode{mode_idx}_std"] = float(
                    np.std(all_omega)
                )

            if L_all[mode]:
                features[f"tensor_patch_L_mode{mode_idx}_mean"] = float(
                    np.mean(L_all[mode])
                )
                features[f"tensor_patch_L_mode{mode_idx}_std"] = float(
                    np.std(L_all[mode])
                )

            if Rlog_all[mode]:
                features[f"tensor_patch_Rlog_mode{mode_idx}_mean"] = float(
                    np.mean(Rlog_all[mode])
                )
                features[f"tensor_patch_Rlog_mode{mode_idx}_std"] = float(
                    np.std(Rlog_all[mode])
                )

        L_means = [np.mean(L_all[m]) for m in range(4) if L_all[m]]
        if L_means:
            features["tensor_patch_L_mean_all_modes"] = float(np.mean(L_means))

        if isinstance(self.logger, GuiLoggerProxy):
            self.logger.info(
                f"Patch张量特征: 使用 patch 组数={len(ref_coords)}, eNL_mean={features.get('tensor_patch_eNL_mean', 0):.4f}"
            )
        else:
            self.logger.info(
                f"Patch tensor features computed with {len(ref_coords)} groups."
            )

        return features
