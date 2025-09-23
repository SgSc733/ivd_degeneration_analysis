import numpy as np
import cv2
from typing import Dict, Optional, List
from .base_calculator import BaseCalculator
from utils.memory_monitor import monitor_memory


class HuMomentsCalculator(BaseCalculator):
    
    def __init__(self):
        super().__init__("Hu Moments Calculator")

    @monitor_memory(threshold_percent=90)    
    def calculate(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        self.validate_input(image, mask)
        
        binary_mask = (mask > 0).astype(np.uint8) * 255
        
        moments = cv2.moments(binary_mask)
        
        hu_moments = cv2.HuMoments(moments)
        
        hu_moments_log = np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        features = {}
        for i in range(7):
            features[f'hu_moment_{i+1}'] = float(hu_moments[i][0])
            features[f'hu_moment_log_{i+1}'] = float(hu_moments_log[i][0])
        
        shape_features = self._calculate_shape_features(binary_mask)
        features.update(shape_features)
        
        return features
    
    def _calculate_shape_features(self, binary_mask: np.ndarray) -> Dict[str, float]:

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return {
                'eccentricity': 0.0,
                'solidity': 0.0,
                'extent': 0.0,
                'compactness': 0.0
            }
        
        contour = max(contours, key=cv2.contourArea)
        
        M = cv2.moments(contour)
        
        if M['m00'] > 0:
            mu20 = M['mu20'] / M['m00']
            mu02 = M['mu02'] / M['m00']
            mu11 = M['mu11'] / M['m00']
            
            lambda1 = 0.5 * ((mu20 + mu02) + np.sqrt((mu20 - mu02)**2 + 4 * mu11**2))
            lambda2 = 0.5 * ((mu20 + mu02) - np.sqrt((mu20 - mu02)**2 + 4 * mu11**2))
            
            if lambda1 > 0:
                eccentricity = np.sqrt(1 - lambda2 / lambda1)
            else:
                eccentricity = 0.0
        else:
            eccentricity = 0.0
        
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        solidity = area / hull_area if hull_area > 0 else 0.0
        
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        
        extent = area / rect_area if rect_area > 0 else 0.0
        
        perimeter = cv2.arcLength(contour, True)
        compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
        
        return {
            'eccentricity': float(eccentricity),
            'solidity': float(solidity),
            'extent': float(extent),
            'compactness': float(compactness)
        }
    
    def process_multi_slice(self, image_slices: List[np.ndarray],
                        masks: List[np.ndarray],
                        use_parallel: Optional[bool] = None) -> Dict[str, float]:
        if use_parallel is None:
            use_parallel = self.enable_parallel
        
        if use_parallel and len(image_slices) >= 5:
            return self.process_multi_slice_parallel(image_slices, masks)
        
        hu_features = {}
        for i, (img, mask) in enumerate(zip(image_slices, masks)):
            slice_features = self.calculate(img, mask)
            for k, v in slice_features.items():
                if k in hu_features:
                    hu_features[k].append(v)
                else:
                    hu_features[k] = [v]
        
        hu_result = {k: np.mean(v) for k, v in hu_features.items()}
        return hu_result

    def process_multi_slice_parallel(self, image_slices: List[np.ndarray],
                                masks: List[np.ndarray]) -> Dict[str, float]:
        
        def process_single_slice(args):
            i, img, mask = args
            try:
                result = self.calculate(img, mask)
                return (i, result, None)
            except Exception as e:
                return (i, None, str(e))
        
        args_list = [(i, img, mask) 
                    for i, (img, mask) 
                    in enumerate(zip(image_slices, masks))]
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(args_list))) as executor:
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
        
        hu_result = {k: np.mean(v) for k, v in all_features.items()}
        hu_result['valid_slices'] = valid_slices
        
        return hu_result
