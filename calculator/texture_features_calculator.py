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
    
    def __init__(self, 
                lbp_radius: int = 1,
                lbp_n_points: int = 8,
                enable_parallel: bool = True,  
                max_workers: Optional[int] = None):

        super().__init__("Extended Texture Features Calculator", enable_parallel=enable_parallel)
        
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        if max_workers is not None:
            self.max_workers = max_workers

    @monitor_memory(threshold_percent=75)
    def calculate(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        self.validate_input(image, mask)
        
        features = {}

        lbp_features = self._calculate_lbp_features(image, mask)
        features.update(lbp_features)

        morph_features = self._calculate_morphological_features(image, mask)
        features.update(morph_features)

        gradient_features = self._calculate_gradient_features(image, mask)
        features.update(gradient_features)
        
        return features
    
    def _calculate_lbp_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        lbp = feature.local_binary_pattern(
            image, self.lbp_n_points, self.lbp_radius, method='uniform'
        )

        lbp_masked = lbp[mask > 0]
        
        if lbp_masked.size == 0:
            return {}

        n_bins = self.lbp_n_points + 2 
        hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins), density=True)

        features = {}

        for i in range(n_bins):
            features[f'lbp_hist_bin_{i}'] = float(hist[i])

        features['lbp_mean'] = float(np.mean(lbp_masked))
        features['lbp_std'] = float(np.std(lbp_masked))
        features['lbp_entropy'] = float(stats.entropy(hist, base=2))
        features['lbp_energy'] = float(np.sum(hist ** 2))
        
        return features
    
    def _calculate_morphological_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        features = {}
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
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        mag_masked = magnitude[mask > 0]
        dir_masked = direction[mask > 0]
        
        if mag_masked.size == 0:
            return {}

        features = {}
        features['gradient_mag_mean'] = float(np.mean(mag_masked))
        features['gradient_mag_std'] = float(np.std(mag_masked))
        features['gradient_mag_max'] = float(np.max(mag_masked))
        features['gradient_mag_skewness'] = float(stats.skew(mag_masked))
        features['gradient_mag_kurtosis'] = float(stats.kurtosis(mag_masked))

        features['gradient_dir_entropy'] = float(self._circular_entropy(dir_masked))
        features['gradient_dir_mean_resultant_length'] = float(self._circular_mean_resultant_length(dir_masked))
        
        return features

    
    def _analyze_skeleton(self, skeleton: np.ndarray) -> Tuple[int, int]:

        kernel = np.ones((3, 3), np.uint8)
        skeleton_binary = (skeleton > 0).astype(np.uint8)

        neighbor_count = cv2.filter2D(skeleton_binary, -1, kernel, borderType=cv2.BORDER_CONSTANT)

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


    def process_multi_slice(self, image_slices: List[np.ndarray],
                        masks: List[np.ndarray],
                        use_parallel: Optional[bool] = None) -> Dict[str, float]:

        if use_parallel is None:
            use_parallel = self.enable_parallel
        
        if use_parallel and len(image_slices) >= 2:
            return self.process_multi_slice_parallel(image_slices, masks)

        all_features = {}
        valid_slices = 0
        for img, mask in zip(image_slices, masks):
            if np.any(mask):
                slice_features = self.calculate(img, mask)
                valid_slices += 1
                for k, v in slice_features.items():
                    if k in all_features:
                        all_features[k].append(v)
                    else:
                        all_features[k] = [v]

        if valid_slices == 0:
            return {}

        final_result = {k: np.mean(v) for k, v in all_features.items()}
        return final_result

    @monitor_memory(threshold_percent=65)
    def process_multi_slice_parallel(self, image_slices: List[np.ndarray],
                                masks: List[np.ndarray]) -> Dict[str, float]:

        all_features = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

            futures = [executor.submit(self.calculate, img, mask) 
                       for img, mask in zip(image_slices, masks) if np.any(mask)]

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

        final_result = {k: np.mean(v) for k, v in all_features.items()}
        return final_result
