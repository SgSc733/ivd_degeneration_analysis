import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.mixture import GaussianMixture
import cv2
from .base_calculator import BaseCalculator
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from utils.memory_monitor import monitor_memory


class T2SignalIntensityCalculator(BaseCalculator):
    
    def __init__(self, roi_method: str = 'TARGET', 
                brightness_percentile: int = 75,
                min_roi_size: int = 20,
                enable_parallel: bool = True,
                max_workers: Optional[int] = None):

        super().__init__(name='T2SignalIntensity', enable_parallel=enable_parallel)
        self.roi_method = roi_method.upper()
        self.brightness_percentile = brightness_percentile
        self.min_roi_size = min_roi_size
        if max_workers is not None:
            self.max_workers = max_workers
        
        if self.roi_method not in ['TARGET', 'ELLIPS', 'WD']:
            raise ValueError(f"不支持的ROI方法: {roi_method}")
    
    @monitor_memory(threshold_percent=80)
    def calculate(self, image: np.ndarray, disc_mask: np.ndarray, 
                  csf_mask: np.ndarray, vertebra_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:

        self.validate_input(image, disc_mask)
        
        if self.roi_method == 'WD':
            roi_mask = self._create_whole_disc_roi(disc_mask)
        elif self.roi_method == 'ELLIPS':
            roi_mask = self._create_ellipsoid_roi(disc_mask)
        else: 
            roi_mask = self._create_target_roi(image, disc_mask)
        
        roi_si = self._calculate_mean_si(image, roi_mask)
        
        csf_si = self._calculate_mean_si(image, csf_mask)
        
        si_ratio = roi_si / csf_si if csf_si > 0 else 0
        
        result = {
            'roi_method': self.roi_method,
            'roi_si': roi_si,
            'csf_si': csf_si,
            'si_ratio': si_ratio,
            'roi_size': np.sum(roi_mask > 0),
            'roi_mask': roi_mask
        }
        
        if vertebra_mask is not None:
            vertebra_si = self._calculate_mean_si(image, vertebra_mask)
            result['vertebra_si'] = vertebra_si
            result['si_ratio_vertebra'] = roi_si / vertebra_si if vertebra_si > 0 else 0
        
        result.update(self._calculate_roi_statistics(image, roi_mask))
        
        return result
    
    def _create_whole_disc_roi(self, disc_mask: np.ndarray) -> np.ndarray:

        return disc_mask.copy()
    
    def _create_ellipsoid_roi(self, disc_mask: np.ndarray) -> np.ndarray:

        contours, _ = cv2.findContours(disc_mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(disc_mask)
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            center, (width, height), angle = ellipse

            width *= 0.8
            height *= 0.8
            
            roi_mask = np.zeros_like(disc_mask)
            cv2.ellipse(roi_mask, (int(center[0]), int(center[1])), 
                       (int(width/2), int(height/2)), angle, 0, 360, 1, -1)
            
            roi_mask = roi_mask * disc_mask
        else:
            roi_mask = self._create_center_roi(disc_mask, ratio=0.6)
        
        return roi_mask
    
    @monitor_memory(threshold_percent=85)
    def _create_target_roi(self, image: np.ndarray, disc_mask: np.ndarray) -> np.ndarray:

        disc_pixels = image[disc_mask > 0]
        
        if len(disc_pixels) == 0:
            return np.zeros_like(disc_mask)
        
        brightness_threshold = np.percentile(disc_pixels, self.brightness_percentile)

        bright_mask = (image > brightness_threshold) & (disc_mask > 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask.astype(np.uint8), 
                                      cv2.MORPH_OPEN, kernel)

        num_labels, labels = cv2.connectedComponents(bright_mask)
        
        if num_labels <= 1:
            brightness_threshold = np.percentile(disc_pixels, self.brightness_percentile - 10)
            bright_mask = (image > brightness_threshold) & (disc_mask > 0)
            bright_mask = cv2.morphologyEx(bright_mask.astype(np.uint8), 
                                          cv2.MORPH_OPEN, kernel)
            num_labels, labels = cv2.connectedComponents(bright_mask)
        
        best_roi = np.zeros_like(disc_mask)
        min_cv = float('inf') 
        
        for label_id in range(1, num_labels):
            region_mask = (labels == label_id)
            region_size = np.sum(region_mask)
            
            if region_size < self.min_roi_size:
                continue
            
            region_pixels = image[region_mask]
            mean_si = np.mean(region_pixels)
            std_si = np.std(region_pixels)
            cv = std_si / mean_si if mean_si > 0 else float('inf')
            
            if cv < min_cv:
                min_cv = cv
                best_roi = region_mask
        
        if np.sum(best_roi) < self.min_roi_size:
            best_roi = self._create_center_roi(disc_mask, ratio=0.4)
        
        return best_roi.astype(np.uint8)
    
    def _create_center_roi(self, disc_mask: np.ndarray, ratio: float = 0.5) -> np.ndarray:

        moments = cv2.moments(disc_mask.astype(np.uint8))
        if moments['m00'] == 0:
            return np.zeros_like(disc_mask)
        
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        area = np.sum(disc_mask > 0)
        radius = int(np.sqrt(area / np.pi) * ratio)
        
        roi_mask = np.zeros_like(disc_mask)
        cv2.circle(roi_mask, (cx, cy), radius, 1, -1)
        
        roi_mask = roi_mask * disc_mask
        
        return roi_mask
    
    def _calculate_mean_si(self, image: np.ndarray, mask: np.ndarray) -> float:

        masked_pixels = image[mask > 0]
        if len(masked_pixels) == 0:
            return 0.0
        return float(np.mean(masked_pixels))
    
    def _calculate_roi_statistics(self, image: np.ndarray, roi_mask: np.ndarray) -> Dict[str, float]:

        roi_pixels = image[roi_mask > 0]
        
        if len(roi_pixels) == 0:
            return {
                'roi_std': 0.0,
                'roi_cv': 0.0,
                'roi_min': 0.0,
                'roi_max': 0.0,
                'roi_median': 0.0
            }
        
        mean_val = np.mean(roi_pixels)
        std_val = np.std(roi_pixels)
        
        return {
            'roi_std': float(std_val),
            'roi_cv': float(std_val / mean_val) if mean_val > 0 else 0.0,
            'roi_min': float(np.min(roi_pixels)),
            'roi_max': float(np.max(roi_pixels)),
            'roi_median': float(np.median(roi_pixels))
        }
    
    def process_multi_slice(self, images: List[np.ndarray], 
                           disc_masks: List[np.ndarray],
                           csf_masks: List[np.ndarray],
                           vertebra_masks: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:

        if len(images) != len(disc_masks) or len(images) != len(csf_masks):
            raise ValueError("图像和掩模数量不匹配")
        
        slice_results = []
        
        for i, (image, disc_mask, csf_mask) in enumerate(zip(images, disc_masks, csf_masks)):
            vertebra_mask = vertebra_masks[i] if vertebra_masks else None
            result = self.calculate(image, disc_mask, csf_mask, vertebra_mask)
            slice_results.append(result)
        
        aggregated = {
            'roi_method': self.roi_method,
            'num_slices': len(slice_results),
            'si_ratio': np.mean([r['si_ratio'] for r in slice_results]),
            'si_ratio_std': np.std([r['si_ratio'] for r in slice_results]),
            'roi_si': np.mean([r['roi_si'] for r in slice_results]),
            'csf_si': np.mean([r['csf_si'] for r in slice_results]),
            'mean_roi_size': np.mean([r['roi_size'] for r in slice_results]),
            'slice_results': slice_results
        }
        
        if vertebra_masks:
            aggregated['si_ratio_vertebra'] = np.mean([r['si_ratio_vertebra'] 
                                                       for r in slice_results 
                                                       if 'si_ratio_vertebra' in r])
        
        return aggregated
    
    @monitor_memory(threshold_percent=75)
    def process_multi_slice_parallel(self, images: List[np.ndarray], 
                                disc_masks: List[np.ndarray],
                                csf_masks: List[np.ndarray],
                                vertebra_masks: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:

        if not self.enable_parallel or len(images) < 3:
            return self.process_multi_slice(images, disc_masks, csf_masks, vertebra_masks)
        
        def process_single_slice(args):
            if len(args) == 4:
                i, image, disc_mask, csf_mask = args
                vertebra_mask = None
            else:
                i, image, disc_mask, csf_mask, vertebra_mask = args
            
            try:
                result = self.calculate(image, disc_mask, csf_mask, vertebra_mask)
                return (i, result, None)
            except Exception as e:
                return (i, None, str(e))

        if vertebra_masks:
            args_list = [(i, img, disc, csf, vert) 
                        for i, (img, disc, csf, vert) 
                        in enumerate(zip(images, disc_masks, csf_masks, vertebra_masks))]
        else:
            args_list = [(i, img, disc, csf) 
                        for i, (img, disc, csf) 
                        in enumerate(zip(images, disc_masks, csf_masks))]

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(args_list))) as executor:
            results = list(executor.map(process_single_slice, args_list))

        slice_results = []
        
        for i, result, error in sorted(results, key=lambda x: x[0]):
            if error:
                self.logger.warning(f"切片{i}处理失败: {error}")
            elif result:
                slice_results.append(result)
        
        if not slice_results:
            raise ValueError("没有成功处理的切片")

        aggregated = {
            'roi_method': self.roi_method,
            'num_slices': len(slice_results),
            'si_ratio': np.mean([r['si_ratio'] for r in slice_results]),
            'si_ratio_std': np.std([r['si_ratio'] for r in slice_results]),
            'roi_si': np.mean([r['roi_si'] for r in slice_results]),
            'csf_si': np.mean([r['csf_si'] for r in slice_results]),
            'mean_roi_size': np.mean([r['roi_size'] for r in slice_results]),
            'slice_results': slice_results
        }
        
        if vertebra_masks:
            aggregated['si_ratio_vertebra'] = np.mean([r['si_ratio_vertebra'] 
                                                    for r in slice_results 
                                                    if 'si_ratio_vertebra' in r])
        
        return aggregated
