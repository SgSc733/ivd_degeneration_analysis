import numpy as np
from skimage import feature, measure
from scipy import ndimage, stats
from typing import Dict, List, Optional, Tuple
import cv2
from .base_calculator import BaseCalculator
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from utils.memory_monitor import monitor_memory
import cv2.ximgproc


class TextureFeaturesCalculator(BaseCalculator):
    
    def __init__(self, 
                lbp_radius: int = 1,
                lbp_n_points: int = 8,
                glcm_distances: List[int] = None,
                glcm_angles: List[float] = None,
                enable_parallel: bool = True,  
                max_workers: Optional[int] = None):

        super().__init__("Texture Features Calculator", enable_parallel=enable_parallel)
        
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.glcm_distances = glcm_distances or [1, 3, 5]
        self.glcm_angles = glcm_angles or [0, np.pi/4, np.pi/2, 3*np.pi/4]
        if max_workers is not None:
            self.max_workers = max_workers

    @monitor_memory(threshold_percent=75)
    def calculate(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        self.validate_input(image, mask)
        
        features = {}

        lbp_features = self._calculate_lbp_features(image, mask)
        features.update(lbp_features)

        glcm_features = self._calculate_glcm_features(image, mask)
        features.update(glcm_features)

        glrlm_features = self._calculate_glrlm_features(image, mask)
        features.update(glrlm_features)

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

        n_bins = self.lbp_n_points + 2 
        hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins), density=True)

        features = {}

        for i in range(n_bins):
            features[f'lbp_hist_bin_{i}'] = float(hist[i])

        features['lbp_mean'] = float(np.mean(lbp_masked))
        features['lbp_std'] = float(np.std(lbp_masked))
        features['lbp_entropy'] = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))

        features['lbp_energy'] = float(np.sum(hist ** 2))
        features['lbp_uniformity'] = float(np.sum(hist ** 2))
        
        return features
    
    @monitor_memory(threshold_percent=70)
    def _calculate_glcm_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        image_norm = self._normalize_to_uint8(image)

        roi = image_norm.copy()
        roi[mask == 0] = 0
        
        features = {}
        
        for distance in self.glcm_distances:
            for angle in self.glcm_angles:
                glcm = feature.graycomatrix(
                    roi, [distance], [angle], levels=256, symmetric=True, normed=True
                )
                
                angle_deg = int(np.degrees(angle))
                prefix = f'glcm_d{distance}_a{angle_deg}'

                features[f'{prefix}_contrast'] = float(
                    feature.graycoprops(glcm, 'contrast')[0, 0]
                )

                features[f'{prefix}_correlation'] = float(
                    feature.graycoprops(glcm, 'correlation')[0, 0]
                )

                features[f'{prefix}_energy'] = float(
                    feature.graycoprops(glcm, 'energy')[0, 0]
                )

                features[f'{prefix}_homogeneity'] = float(
                    feature.graycoprops(glcm, 'homogeneity')[0, 0]
                )

                glcm_2d = glcm[:, :, 0, 0]

                entropy = -np.sum(glcm_2d[glcm_2d > 0] * np.log2(glcm_2d[glcm_2d > 0]))
                features[f'{prefix}_entropy'] = float(entropy)

                features[f'{prefix}_max_probability'] = float(np.max(glcm_2d))
                
        return features
    
    @monitor_memory(threshold_percent=70)
    def _calculate_glcm_features_parallel(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        image_norm = self._normalize_to_uint8(image)
        roi = image_norm.copy()
        roi[mask == 0] = 0
        
        features = {}
        
        def compute_single_glcm(params):
            distance, angle = params

            glcm = feature.graycomatrix(
                roi, [distance], [angle], levels=256, symmetric=True, normed=True
            )

            angle_deg = int(np.degrees(angle))
            prefix = f'glcm_d{distance}_a{angle_deg}'
            
            result = {
                f'{prefix}_contrast': float(feature.graycoprops(glcm, 'contrast')[0, 0]),
                f'{prefix}_correlation': float(feature.graycoprops(glcm, 'correlation')[0, 0]),
                f'{prefix}_energy': float(feature.graycoprops(glcm, 'energy')[0, 0]),
                f'{prefix}_homogeneity': float(feature.graycoprops(glcm, 'homogeneity')[0, 0])
            }

            glcm_2d = glcm[:, :, 0, 0]
            entropy = -np.sum(glcm_2d[glcm_2d > 0] * np.log2(glcm_2d[glcm_2d > 0]))
            result[f'{prefix}_entropy'] = float(entropy)
            result[f'{prefix}_max_probability'] = float(np.max(glcm_2d))
            
            return result

        params_list = [(d, a) for d in self.glcm_distances for a in self.glcm_angles]

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(params_list))) as executor:
            results = list(executor.map(compute_single_glcm, params_list))

        for result in results:
            features.update(result)
        
        return features
    
    @monitor_memory(threshold_percent=80)
    def _calculate_glrlm_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        image_norm = self._normalize_to_uint8(image)
        n_levels = 16 
        image_quantized = (image_norm // (256 // n_levels)).astype(np.uint8)
        
        features = {}

        angles = [0, 45, 90, 135]
        
        for angle in angles:
            glrlm = self._compute_glrlm(image_quantized, mask, angle, n_levels)
            
            if glrlm is None:
                continue

            prefix = f'glrlm_a{angle}'

            sre = self._glrlm_sre(glrlm)
            features[f'{prefix}_sre'] = float(sre)

            lre = self._glrlm_lre(glrlm)
            features[f'{prefix}_lre'] = float(lre)

            gln = self._glrlm_gln(glrlm)
            features[f'{prefix}_gln'] = float(gln)

            rln = self._glrlm_rln(glrlm)
            features[f'{prefix}_rln'] = float(rln)

            rp = self._glrlm_rp(glrlm, mask)
            features[f'{prefix}_rp'] = float(rp)
            
        return features
    
    def _calculate_morphological_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        features = {}

        binary_mask = (mask > 0).astype(np.uint8)

        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

        dist_values = dist_transform[mask > 0]
        
        if len(dist_values) > 0:
            features['morph_dist_mean'] = float(np.mean(dist_values))
            features['morph_dist_std'] = float(np.std(dist_values))
            features['morph_dist_max'] = float(np.max(dist_values))
            features['morph_thickness'] = float(np.max(dist_values) * 2) 
        else:
            features['morph_dist_mean'] = 0.0
            features['morph_dist_std'] = 0.0
            features['morph_dist_max'] = 0.0
            features['morph_thickness'] = 0.0

        skeleton = self._compute_skeleton(binary_mask)
        skeleton_pixels = np.sum(skeleton > 0)
        features['morph_skeleton_pixels'] = float(skeleton_pixels)
        
        branches, endpoints = self._analyze_skeleton(skeleton)
        features['morph_branch_points'] = float(branches)
        features['morph_end_points'] = float(endpoints)
        
        return features
    
    def _compute_skeleton(self, binary_mask: np.ndarray) -> np.ndarray:

        skeleton = cv2.ximgproc.thinning(binary_mask * 255)
        return skeleton
    
    def _calculate_gradient_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        mag_masked = magnitude[mask > 0]
        dir_masked = direction[mask > 0]
        
        features = {}

        features['gradient_mag_mean'] = float(np.mean(mag_masked))
        features['gradient_mag_std'] = float(np.std(mag_masked))
        features['gradient_mag_max'] = float(np.max(mag_masked))
        features['gradient_mag_skewness'] = float(stats.skew(mag_masked))
        features['gradient_mag_kurtosis'] = float(stats.kurtosis(mag_masked))

        features['gradient_dir_entropy'] = float(self._circular_entropy(dir_masked))
        features['gradient_dir_uniformity'] = float(self._circular_uniformity(dir_masked))
        
        return features
    
    def _normalize_to_uint8(self, image: np.ndarray) -> np.ndarray:

        img_min = np.min(image)
        img_max = np.max(image)
        
        if img_max == img_min:
            return np.zeros_like(image, dtype=np.uint8)
        
        normalized = (image - img_min) / (img_max - img_min) * 255
        return normalized.astype(np.uint8)
    
    def _compute_glrlm(self, image: np.ndarray, mask: np.ndarray, 
                    angle: int, n_levels: int) -> Optional[np.ndarray]:

        h, w = image.shape
        max_run_length = max(h, w)
        glrlm = np.zeros((n_levels, max_run_length))
        
        if angle == 0:  
            for i in range(h):
                if np.any(mask[i, :] > 0):
                    runs = self._get_runs(image[i, :], mask[i, :])
                    for gray_level, run_length in runs:
                        if gray_level < n_levels and run_length > 0:
                            glrlm[gray_level, run_length-1] += 1
                            
        elif angle == 90:  
            for j in range(w):
                if np.any(mask[:, j] > 0):
                    runs = self._get_runs(image[:, j], mask[:, j])
                    for gray_level, run_length in runs:
                        if gray_level < n_levels and run_length > 0:
                            glrlm[gray_level, run_length-1] += 1
                            
        elif angle == 45:  
            for start_row in range(h):
                diagonal_pixels = []
                diagonal_mask = []
                i, j = start_row, 0
                while i < h and j < w:
                    diagonal_pixels.append(image[i, j])
                    diagonal_mask.append(mask[i, j])
                    i += 1
                    j += 1
                
                if len(diagonal_pixels) > 0 and np.any(np.array(diagonal_mask) > 0):
                    runs = self._get_runs(np.array(diagonal_pixels), np.array(diagonal_mask))
                    for gray_level, run_length in runs:
                        if gray_level < n_levels and run_length > 0:
                            glrlm[gray_level, run_length-1] += 1

            for start_col in range(1, w):
                diagonal_pixels = []
                diagonal_mask = []
                i, j = 0, start_col
                while i < h and j < w:
                    diagonal_pixels.append(image[i, j])
                    diagonal_mask.append(mask[i, j])
                    i += 1
                    j += 1
                
                if len(diagonal_pixels) > 0 and np.any(np.array(diagonal_mask) > 0):
                    runs = self._get_runs(np.array(diagonal_pixels), np.array(diagonal_mask))
                    for gray_level, run_length in runs:
                        if gray_level < n_levels and run_length > 0:
                            glrlm[gray_level, run_length-1] += 1
                            
        elif angle == 135:  
            for start_row in range(h):
                diagonal_pixels = []
                diagonal_mask = []
                i, j = start_row, w - 1
                while i < h and j >= 0:
                    diagonal_pixels.append(image[i, j])
                    diagonal_mask.append(mask[i, j])
                    i += 1
                    j -= 1
                
                if len(diagonal_pixels) > 0 and np.any(np.array(diagonal_mask) > 0):
                    runs = self._get_runs(np.array(diagonal_pixels), np.array(diagonal_mask))
                    for gray_level, run_length in runs:
                        if gray_level < n_levels and run_length > 0:
                            glrlm[gray_level, run_length-1] += 1

            for start_col in range(w - 2, -1, -1):
                diagonal_pixels = []
                diagonal_mask = []
                i, j = 0, start_col
                while i < h and j >= 0:
                    diagonal_pixels.append(image[i, j])
                    diagonal_mask.append(mask[i, j])
                    i += 1
                    j -= 1
                
                if len(diagonal_pixels) > 0 and np.any(np.array(diagonal_mask) > 0):
                    runs = self._get_runs(np.array(diagonal_pixels), np.array(diagonal_mask))
                    for gray_level, run_length in runs:
                        if gray_level < n_levels and run_length > 0:
                            glrlm[gray_level, run_length-1] += 1
        
        return glrlm
    
    def _get_runs(self, line: np.ndarray, mask_line: np.ndarray) -> List[Tuple[int, int]]:

        runs = []
        
        if len(line) == 0 or not np.any(mask_line > 0):
            return runs
        
        current_value = line[0]
        current_length = 1
        
        for i in range(1, len(line)):
            if mask_line[i] > 0 and line[i] == current_value:
                current_length += 1
            else:
                if mask_line[i-1] > 0:
                    runs.append((current_value, current_length))
                current_value = line[i]
                current_length = 1

        if mask_line[-1] > 0:
            runs.append((current_value, current_length))
        
        return runs
    
    def _glrlm_sre(self, glrlm: np.ndarray) -> float:

        i, j = np.meshgrid(range(glrlm.shape[0]), range(1, glrlm.shape[1]+1), indexing='ij')
        sre = np.sum(glrlm / (j ** 2))
        return sre / np.sum(glrlm) if np.sum(glrlm) > 0 else 0
    
    def _glrlm_lre(self, glrlm: np.ndarray) -> float:

        i, j = np.meshgrid(range(glrlm.shape[0]), range(1, glrlm.shape[1]+1), indexing='ij')
        lre = np.sum(glrlm * (j ** 2))
        return lre / np.sum(glrlm) if np.sum(glrlm) > 0 else 0
    
    def _glrlm_gln(self, glrlm: np.ndarray) -> float:

        gray_level_sum = np.sum(glrlm, axis=1)
        gln = np.sum(gray_level_sum ** 2)
        return gln / np.sum(glrlm) if np.sum(glrlm) > 0 else 0
    
    def _glrlm_rln(self, glrlm: np.ndarray) -> float:

        run_length_sum = np.sum(glrlm, axis=0)
        rln = np.sum(run_length_sum ** 2)
        return rln / np.sum(glrlm) if np.sum(glrlm) > 0 else 0
    
    def _glrlm_rp(self, glrlm: np.ndarray, mask: np.ndarray) -> float:

        total_runs = np.sum(glrlm)
        total_pixels = np.sum(mask > 0)
        return total_runs / total_pixels if total_pixels > 0 else 0
    
    def _analyze_skeleton(self, skeleton: np.ndarray) -> Tuple[int, int]:

        kernel = np.ones((3, 3), np.uint8)

        skeleton_binary = (skeleton > 0).astype(np.uint8)
        neighbor_count = cv2.filter2D(skeleton_binary, -1, kernel) - skeleton_binary

        endpoints = np.sum((neighbor_count == 1) & (skeleton_binary == 1))

        branches = np.sum((neighbor_count >= 3) & (skeleton_binary == 1))
        
        return int(branches), int(endpoints)
    
    def _circular_entropy(self, angles: np.ndarray) -> float:

        n_bins = 36  
        hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi), density=True)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.0
        
        return -np.sum(hist * np.log2(hist))
    
    def _circular_uniformity(self, angles: np.ndarray) -> float:

        mean_vector = np.mean(np.exp(1j * angles))
        return np.abs(mean_vector)

    def process_multi_slice(self, image_slices: List[np.ndarray],
                        masks: List[np.ndarray],
                        use_parallel: Optional[bool] = None) -> Dict[str, float]:

        if use_parallel is None:
            use_parallel = self.enable_parallel
        
        if use_parallel and len(image_slices) >= 3:
            return self.process_multi_slice_parallel(image_slices, masks)

        texture_features = {}
        for i, (img, mask) in enumerate(zip(image_slices, masks)):
            slice_features = self.calculate(img, mask)
            for k, v in slice_features.items():
                if k in texture_features:
                    texture_features[k].append(v)
                else:
                    texture_features[k] = [v]

        texture_result = {k: np.mean(v) for k, v in texture_features.items()}
        return texture_result

    @monitor_memory(threshold_percent=65)
    def process_multi_slice_parallel(self, image_slices: List[np.ndarray],
                                masks: List[np.ndarray]) -> Dict[str, float]:

        def process_single_slice(args):
            i, img, mask = args
            try:
                calculator = TextureFeaturesCalculator(
                    lbp_radius=self.lbp_radius,
                    lbp_n_points=self.lbp_n_points,
                    glcm_distances=self.glcm_distances,
                    glcm_angles=self.glcm_angles
                )
                calculator.enable_parallel = True
                calculator.max_workers = 2 

                result = calculator.calculate_parallel(img, mask)
                return (i, result, None)
            except Exception as e:
                return (i, None, str(e))
        
        args_list = [(i, img, mask) 
                    for i, (img, mask) 
                    in enumerate(zip(image_slices, masks))]

        with ThreadPoolExecutor(max_workers=min(2, len(args_list))) as executor:
            results = list(executor.map(process_single_slice, args_list))

        all_features = {}
        valid_slices = 0
        
        for i, result, error in sorted(results, key=lambda x: x[0]):
            if error:
                self.logger.warning(f"切片{i}处理失败: {error}")
            elif result:
                valid_slices += 1
                for k, v in result.items():
                    if k in all_features:
                        all_features[k].append(v)
                    else:
                        all_features[k] = [v]
        
        if valid_slices == 0:
            raise ValueError("没有成功处理的切片")

        texture_result = {k: np.mean(v) for k, v in all_features.items()}
        texture_result['valid_slices'] = valid_slices
        
        return texture_result

    def calculate_parallel(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        features = {}

        lbp_features = self._calculate_lbp_features(image, mask)
        features.update(lbp_features)

        glcm_features = self._calculate_glcm_features_parallel(image, mask)
        features.update(glcm_features)

        glrlm_features = self._calculate_glrlm_features(image, mask)
        features.update(glrlm_features)

        morph_features = self._calculate_morphological_features(image, mask)
        features.update(morph_features)

        gradient_features = self._calculate_gradient_features(image, mask)
        features.update(gradient_features)
        
        return features
    
    def _calculate_glcm_features_parallel_safe(self, image: np.ndarray, 
                                            mask: np.ndarray) -> Dict[str, float]:

        import gc
        
        image_norm = self._normalize_to_uint8(image)
        roi = image_norm.copy()
        roi[mask == 0] = 0
        
        features = {}

        params_list = [(d, a) for d in self.glcm_distances for a in self.glcm_angles]

        memory_info = psutil.virtual_memory()
        if memory_info.percent > 70:
            batch_size = 4
            for i in range(0, len(params_list), batch_size):
                batch = params_list[i:i+batch_size]
                
                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = list(executor.map(
                        lambda p: self._compute_single_glcm_safe(roi, p[0], p[1]), 
                        batch
                    ))
                
                for result in results:
                    features.update(result)

                gc.collect()
        else:
            return self._calculate_glcm_features_parallel(image, mask)
        
        return features

    def _compute_single_glcm_safe(self, roi: np.ndarray, distance: int, 
                                angle: float) -> Dict[str, float]:

        try:
            glcm = feature.graycomatrix(
                roi, [distance], [angle], levels=256, symmetric=True, normed=True
            )
            
            angle_deg = int(np.degrees(angle))
            prefix = f'glcm_d{distance}_a{angle_deg}'
            
            result = {
                f'{prefix}_contrast': float(feature.graycoprops(glcm, 'contrast')[0, 0]),
                f'{prefix}_correlation': float(feature.graycoprops(glcm, 'correlation')[0, 0]),
                f'{prefix}_energy': float(feature.graycoprops(glcm, 'energy')[0, 0]),
                f'{prefix}_homogeneity': float(feature.graycoprops(glcm, 'homogeneity')[0, 0])
            }
            
            glcm_2d = glcm[:, :, 0, 0]
            entropy = -np.sum(glcm_2d[glcm_2d > 0] * np.log2(glcm_2d[glcm_2d > 0]))
            result[f'{prefix}_entropy'] = float(entropy)
            result[f'{prefix}_max_probability'] = float(np.max(glcm_2d))
            
            return result
            
        except MemoryError:
            self.logger.error(f"内存错误：distance={distance}, angle={angle}")
            return {}