import numpy as np
import cv2
from skimage import filters
from typing import Dict, List, Tuple, Optional
from .base_calculator import BaseCalculator
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from utils.memory_monitor import monitor_memory


class GaborCalculator(BaseCalculator):
    
    def __init__(self, wavelengths: List[float] = None, 
                orientations: List[float] = None,
                frequency: float = 0.1,
                sigma: Optional[float] = None,
                gamma: float = 0.5,
                psi: float = 0,
                enable_parallel: bool = True,  
                max_workers: Optional[int] = None):

        super().__init__("Gabor Calculator", enable_parallel=enable_parallel)
        
        self.wavelengths = wavelengths or [2, 4, 6, 8, 10]
        self.orientations = orientations or np.linspace(0, np.pi, 8, endpoint=False)
        self.frequency = frequency
        self.sigma = sigma
        self.gamma = gamma
        self.psi = psi
        if max_workers is not None:
            self.max_workers = max_workers

    @monitor_memory(threshold_percent=75)   
    def calculate(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        self.validate_input(image, mask)
        
        roi = self._extract_roi(image, mask)
        roi_normalized = self._normalize_image(roi)
        
        features = {}
        feature_index = 0
        
        for wavelength in self.wavelengths:
            for orientation in self.orientations:
                real, imag = self._apply_gabor_filter(
                    roi_normalized, wavelength, orientation
                )
                
                magnitude = np.sqrt(real**2 + imag**2)
                
                stats = self._extract_statistics(magnitude, mask)
                
                orientation_deg = np.degrees(orientation)
                prefix = f'gabor_w{wavelength}_o{int(orientation_deg)}'
                
                features[f'{prefix}_mean'] = stats['mean']
                features[f'{prefix}_std'] = stats['std']
                features[f'{prefix}_energy'] = stats['energy']
                features[f'{prefix}_entropy'] = stats['entropy']
                features[f'{prefix}_skewness'] = stats['skewness']
                
                feature_index += 5
        
        self.logger.info(f"提取了{feature_index}个Gabor特征")
        
        return features
    
    @monitor_memory(threshold_percent=70)
    def calculate_parallel(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        if not self.enable_parallel:
            return self.calculate(image, mask)
        
        self.validate_input(image, mask)
        
        roi = self._extract_roi(image, mask)
        roi_normalized = self._normalize_image(roi)
        
        features = {}
        
        def compute_single_gabor(params):
            wavelength, orientation = params
            
            real, imag = self._apply_gabor_filter(roi_normalized, wavelength, orientation)
            magnitude = np.sqrt(real**2 + imag**2)
            
            stats = self._extract_statistics(magnitude, mask)
            
            orientation_deg = np.degrees(orientation)
            prefix = f'gabor_w{wavelength}_o{int(orientation_deg)}'
            
            return {
                f'{prefix}_mean': stats['mean'],
                f'{prefix}_std': stats['std'],
                f'{prefix}_energy': stats['energy'],
                f'{prefix}_entropy': stats['entropy'],
                f'{prefix}_skewness': stats['skewness']
            }
        
        params_list = [(w, o) for w in self.wavelengths for o in self.orientations]
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(params_list))) as executor:
            results = list(executor.map(compute_single_gabor, params_list))
        
        for result in results:
            features.update(result)
        
        self.logger.info(f"并行提取了{len(features)}个Gabor特征")
        
        return features
    
    @monitor_memory(threshold_percent=80)
    def _apply_gabor_filter(self, image: np.ndarray, 
                           wavelength: float, 
                           orientation: float) -> Tuple[np.ndarray, np.ndarray]:

        frequency = 1.0 / wavelength
        sigma = self.sigma or 0.56 * wavelength
        
        real, imag = filters.gabor(
            image,
            frequency=frequency,
            theta=orientation,
            sigma_x=sigma,
            sigma_y=sigma/self.gamma,
            mode='reflect'
        )
        
        return real, imag
    
    def _extract_roi(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:

        coords = np.column_stack(np.where(mask > 0))
        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0)

        roi = image[min_row:max_row+1, min_col:max_col+1].copy()
        roi_mask = mask[min_row:max_row+1, min_col:max_col+1]
        
        roi[roi_mask == 0] = 0
        
        return roi
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:

        non_zero = image[image > 0]
        if len(non_zero) == 0:
            return image
        
        min_val = non_zero.min()
        max_val = non_zero.max()
        
        if max_val == min_val:
            return image
        
        normalized = (image - min_val) / (max_val - min_val)
        normalized[image == 0] = 0
        
        return normalized
    
    def _extract_statistics(self, response: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        valid_pixels = response[mask > 0]
        
        if len(valid_pixels) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'energy': 0.0,
                'entropy': 0.0,
                'skewness': 0.0
            }
        
        mean = np.mean(valid_pixels)
        std = np.std(valid_pixels)
        energy = np.sum(valid_pixels ** 2)
        
        hist, _ = np.histogram(valid_pixels, bins=256, density=True)
        hist = hist[hist > 0]  
        entropy = -np.sum(hist * np.log2(hist))
        
        if std > 0:
            skewness = np.mean(((valid_pixels - mean) / std) ** 3)
        else:
            skewness = 0.0
        
        return {
            'mean': float(mean),
            'std': float(std),
            'energy': float(energy),
            'entropy': float(entropy),
            'skewness': float(skewness)
        }
    
    def process_multi_slice(self, image_slices: List[np.ndarray],
                        masks: List[np.ndarray],
                        use_parallel: Optional[bool] = None) -> Dict[str, float]:
        
        if use_parallel is None:
            use_parallel = self.enable_parallel
        
        if use_parallel and len(image_slices) >= 3:
            return self.process_multi_slice_parallel(image_slices, masks)
        
        gabor_features = {}
        for i, (img, mask) in enumerate(zip(image_slices, masks)):
            slice_features = self.calculate(img, mask)
            for k, v in slice_features.items():
                if k in gabor_features:
                    gabor_features[k].append(v)
                else:
                    gabor_features[k] = [v]
        
        gabor_result = {k: np.mean(v) for k, v in gabor_features.items()}
        return gabor_result

    def process_multi_slice_parallel(self, image_slices: List[np.ndarray],
                                masks: List[np.ndarray]) -> Dict[str, float]:

        def process_single_slice(args):
            i, img, mask = args
            try:
                result = self.calculate_parallel(img, mask)
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
        
        gabor_result = {k: np.mean(v) for k, v in all_features.items()}
        gabor_result['valid_slices'] = valid_slices
        
        return gabor_result

    def calculate_parallel_with_memory_management(self, image: np.ndarray, 
                                                mask: np.ndarray) -> Dict[str, float]:

        memory_info = psutil.virtual_memory()
        available_gb = memory_info.available / (1024**3)
        
        total_filters = len(self.wavelengths) * len(self.orientations)
        
        if available_gb < 2:
            self.logger.warning("内存不足，使用串行计算")
            return self.calculate(image, mask)
        elif available_gb < 4:
            max_workers = 2
        else:
            max_workers = min(self.max_workers, total_filters)
        
        original_max_workers = self.max_workers
        self.max_workers = max_workers
        
        try:
            result = self.calculate_parallel(image, mask)
        finally:
            self.max_workers = original_max_workers
        
        return result