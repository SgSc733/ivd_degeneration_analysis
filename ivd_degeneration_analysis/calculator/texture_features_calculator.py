import numpy as np
from skimage import feature
from scipy import stats
from typing import Dict, List, Optional, Tuple
import cv2
from .base_calculator import BaseCalculator
from concurrent.futures import ThreadPoolExecutor
from utils.memory_monitor import monitor_memory
import cv2.ximgproc
import psutil


class TextureFeaturesCalculator(BaseCalculator):

    def __init__(
        self,
        lbp_radius: int = 1,
        lbp_n_points: int = 8,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            "Extended Texture Features Calculator",
            enable_parallel=enable_parallel,
            **kwargs
        )

        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        if max_workers is not None:
            self.max_workers = max_workers

    @monitor_memory(threshold_percent=75)
    def calculate(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        self.validate_input(image, mask)
        if not np.any(mask):
            return {}

        features: Dict[str, float] = {}

        lbp_features = self._calculate_lbp_features(image, mask)
        features.update(lbp_features)

        morph_features = self._calculate_morphological_features(image, mask)
        features.update(morph_features)

        gradient_features = self._calculate_gradient_features(image, mask)
        features.update(gradient_features)
        try:
            prognostic_features = self._calculate_prognostic_features(features)
            features.update(prognostic_features)
        except Exception as e:
            self.logger.warning(f"预后特征计算失败: {str(e)}")
            features.update({
                'comprehensive_texture_degeneration_index': 0.0,
                'annular_structural_integrity_score': 0.0,
                'inflammatory_texture_pattern_recognition': 0.0,
                'mechanical_load_texture_adaptability': 0.0
            })

        return features

    def _calculate_prognostic_features(self, basic_features: Dict[str, float]) -> Dict[str, float]:
        epsilon = 1e-8

        lbp_entropy = basic_features.get('lbp_entropy', 0.0)
        lbp_energy = basic_features.get('lbp_energy', 0.0)

        branches = basic_features.get('morph_branch_points', 0.0)
        endpoints = basic_features.get('morph_end_points', 0.0)
        dist_mean = basic_features.get('morph_dist_mean', 0.0)
        dist_std = basic_features.get('morph_dist_std', 0.0)
        dist_max = basic_features.get('morph_dist_max', 0.0)

        grad_dir_entropy = basic_features.get('gradient_dir_entropy', 0.0)
        grad_resultant_len = basic_features.get('gradient_dir_mean_resultant_length', 0.0)
        grad_skew = basic_features.get('gradient_mag_skewness', 0.0)
        grad_kurt = basic_features.get('gradient_mag_kurtosis', 0.0)
        grad_mag_mean = basic_features.get('gradient_mag_mean', 0.0)
        norm_lbp_entropy = min(lbp_entropy / 4.0, 1.0)
        norm_branches = min(branches / 20.0, 1.0)
        norm_grad_entropy = min(grad_dir_entropy / 5.5, 1.0)

        dist_cv = dist_std / (dist_mean + epsilon)
        norm_dist_cv = min(dist_cv, 1.0)

        ctdi = (
            (0.3 * norm_lbp_entropy) +
            (0.3 * norm_branches) +
            (0.2 * norm_grad_entropy) +
            (0.2 * norm_dist_cv)
        )

        endpoints_score = 1.0 / (endpoints + 1.0)
        dir_consistency = grad_resultant_len
        optimal_thickness = 5.0
        thickness_score = float(np.exp(-0.5 * ((dist_max - optimal_thickness) ** 2)))

        asis = (
            (0.4 * endpoints_score) +
            (0.4 * dir_consistency) +
            (0.2 * thickness_score)
        )

        instability = 1.0 - lbp_energy
        skew_risk = (1.0 / (1.0 + np.exp(-grad_skew))) - 0.5
        skew_risk = max(0.0, min(float(skew_risk) * 2.0, 1.0))
        kurt_risk = 1.0 - min(abs(grad_kurt), 1.0)

        itpr = (
            (0.4 * instability) +
            (0.3 * skew_risk) +
            (0.3 * kurt_risk)
        )

        thickness_homogeneity = max(0.0, 1.0 - (dist_std / (dist_mean + epsilon)))
        grad_strength = float(np.tanh(grad_mag_mean / 50.0))
        branch_opt = max(0.0, 1.0 - (branches / 20.0))

        mlta = (
            (0.4 * thickness_homogeneity) +
            (0.3 * grad_strength) +
            (0.3 * branch_opt)
        )

        return {
            'comprehensive_texture_degeneration_index': float(ctdi),
            'annular_structural_integrity_score': float(asis),
            'inflammatory_texture_pattern_recognition': float(itpr),
            'mechanical_load_texture_adaptability': float(mlta)
        }
    def _calculate_lbp_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        lbp = feature.local_binary_pattern(
            image, self.lbp_n_points, self.lbp_radius, method='uniform'
        )

        lbp_masked = lbp[mask > 0]

        if lbp_masked.size == 0:
            return {}

        n_bins = self.lbp_n_points + 2
        hist, _ = np.histogram(
            lbp_masked, bins=n_bins, range=(0, n_bins), density=True
        )

        features: Dict[str, float] = {}

        for i in range(n_bins):
            features[f'lbp_hist_bin_{i}'] = float(hist[i])

        features['lbp_mean'] = float(np.mean(lbp_masked))
        features['lbp_std'] = float(np.std(lbp_masked))
        features['lbp_entropy'] = float(stats.entropy(hist, base=2))
        features['lbp_energy'] = float(np.sum(hist ** 2))

        return features

    def _calculate_morphological_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        features: Dict[str, float] = {}
        binary_mask = (mask > 0).astype(np.uint8)

        if np.sum(binary_mask) == 0:
            return {}

        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        dist_values = dist_transform[mask > 0]

        if dist_values.size > 0:
            features['morph_dist_mean'] = float(np.mean(dist_values))
            features['morph_dist_std'] = float(np.std(dist_values))
            features['morph_dist_max'] = float(np.max(dist_values))
            features['morph_thickness'] = float(np.max(dist_values) * 2)
        else:
            features['morph_dist_mean'] = 0.0
            features['morph_dist_std'] = 0.0
            features['morph_dist_max'] = 0.0
            features['morph_thickness'] = 0.0

        skeleton = cv2.ximgproc.thinning(binary_mask * 255)
        features['morph_skeleton_pixels'] = float(np.sum(skeleton > 0))

        branches, endpoints = self._analyze_skeleton(skeleton)
        features['morph_branch_points'] = float(branches)
        features['morph_end_points'] = float(endpoints)

        return features

    def _calculate_gradient_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        image_float = image.astype(np.float32)
        grad_x = cv2.Sobel(image_float, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_float, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        direction = np.arctan2(grad_y, grad_x)

        mag_masked = magnitude[mask > 0]
        dir_masked = direction[mask > 0]

        if mag_masked.size == 0:
            return {}

        features: Dict[str, float] = {}
        features['gradient_mag_mean'] = float(np.mean(mag_masked))
        features['gradient_mag_std'] = float(np.std(mag_masked))
        features['gradient_mag_max'] = float(np.max(mag_masked))
        features['gradient_mag_skewness'] = float(stats.skew(mag_masked))
        features['gradient_mag_kurtosis'] = float(stats.kurtosis(mag_masked))

        features['gradient_dir_entropy'] = float(self._circular_entropy(dir_masked))
        features['gradient_dir_mean_resultant_length'] = float(
            self._circular_mean_resultant_length(dir_masked)
        )

        return features

    def _analyze_skeleton(self, skeleton: np.ndarray) -> Tuple[int, int]:
        kernel = np.ones((3, 3), np.uint8)
        skeleton_binary = (skeleton > 0).astype(np.uint8)

        neighbor_count = cv2.filter2D(
            skeleton_binary, -1, kernel, borderType=cv2.BORDER_CONSTANT
        )

        neighbor_count = neighbor_count - skeleton_binary
        neighbor_count_on_skeleton = neighbor_count * skeleton_binary

        endpoints = np.sum(neighbor_count_on_skeleton == 1)
        branches = np.sum(neighbor_count_on_skeleton >= 3)

        return int(branches), int(endpoints)

    def _circular_entropy(self, angles: np.ndarray) -> float:
        if angles.size == 0:
            return 0.0
        n_bins = 36
        hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
        prob_dist = hist / np.sum(hist)
        return float(stats.entropy(prob_dist, base=2))

    def _circular_mean_resultant_length(self, angles: np.ndarray) -> float:
        if angles.size == 0:
            return 0.0
        mean_vector = np.mean(np.exp(1j * angles))
        return float(np.abs(mean_vector))

    def process_multi_slice(
        self,
        image_slices: List[np.ndarray],
        masks: List[np.ndarray],
        use_parallel: Optional[bool] = None
    ) -> Dict[str, float]:

        if use_parallel is None:
            use_parallel = self.enable_parallel

        if use_parallel and len(image_slices) >= 2:
            return self.process_multi_slice_parallel(image_slices, masks)

        all_features: Dict[str, List[float]] = {}
        valid_slices = 0

        for i, (img, mask) in enumerate(zip(image_slices, masks)):
            if np.any(mask):
                try:
                    slice_features = self.calculate(img, mask)
                except Exception as exc:
                    self.logger.warning(f"切片{i}处理失败: {exc}")
                    continue
                valid_slices += 1
                for k, v in slice_features.items():
                    if k in all_features:
                        all_features[k].append(v)
                    else:
                        all_features[k] = [v]

        if valid_slices == 0:
            return {}

        final_result = {k: float(np.mean(v)) for k, v in all_features.items()}
        return final_result

    @monitor_memory(threshold_percent=65)
    def process_multi_slice_parallel(
        self,
        image_slices: List[np.ndarray],
        masks: List[np.ndarray]
    ) -> Dict[str, float]:

        all_features: Dict[str, List[float]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.calculate, img, mask)
                for img, mask in zip(image_slices, masks)
                if np.any(mask)
            ]

            for future in futures:
                try:
                    result = future.result()
                    for k, v in result.items():
                        if k in all_features:
                            all_features[k].append(v)
                        else:
                            all_features[k] = [v]
                except Exception as e:
                    self.logger.error(f"并行计算单个切片时出错: {e}")

        if not all_features:
            return {}

        final_result = {k: float(np.mean(v)) for k, v in all_features.items()}
        return final_result
