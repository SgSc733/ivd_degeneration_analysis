import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from .base_calculator import BaseCalculator
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from utils.memory_monitor import monitor_memory


class DHICalculator(BaseCalculator):
    
    def __init__(self, central_ratio: float = 0.8, 
                calculate_dwr: bool = True,
                consider_bulging: bool = True,
                enable_parallel: bool = False, 
                max_workers: Optional[int] = None, **kwargs):

        super().__init__("DHI Calculator", enable_parallel=enable_parallel, **kwargs)
        self.central_ratio = central_ratio
        self.calculate_dwr = calculate_dwr
        self.consider_bulging = consider_bulging
        if max_workers is not None:
            self.max_workers = max_workers
        
    @monitor_memory(threshold_percent=85)
    def calculate(self, upper_vertebra_mask: np.ndarray,
                 disc_mask: np.ndarray,
                 lower_vertebra_mask: np.ndarray,
                 is_l5_s1: bool = False) -> Dict[str, float]:

        try:
            self.validate_input(upper_vertebra_mask, upper_vertebra_mask)
            self.validate_input(disc_mask, disc_mask)
            self.validate_input(lower_vertebra_mask, lower_vertebra_mask)
            
            upper_corners = self._calculate_vertebral_corners(upper_vertebra_mask)
            
            if is_l5_s1:
                lower_corners = self._calculate_s1_corners(lower_vertebra_mask)
            else:
                lower_corners = self._calculate_vertebral_corners(lower_vertebra_mask)
            
            upper_diameter = self._calculate_vertebral_diameter(upper_corners)
            lower_diameter = self._calculate_vertebral_diameter(lower_corners)
            
            upper_vh = self._calculate_vertebral_height(upper_vertebra_mask, upper_diameter)
            lower_vh = self._calculate_vertebral_height(lower_vertebra_mask, lower_diameter)
            
            disc_params = self._calculate_disc_parameters(
                disc_mask, upper_corners, lower_corners
            )
            
            if is_l5_s1:
                dhi = disc_params['disc_height'] / upper_vh if upper_vh > 0 else 0
            else:
                dhi = self._calculate_dhi(disc_params['disc_height'], upper_vh, lower_vh)
            
            result = {
                'dhi': dhi,
                'disc_height': disc_params['disc_height'],
                'disc_width_small': disc_params['small_width'],
                'disc_width_big': disc_params['big_width'],
                'upper_vh': upper_vh,
                'lower_vh': lower_vh,
                'upper_diameter': upper_diameter,
                'lower_diameter': lower_diameter,
                'upper_corners': upper_corners.tolist(),
                'lower_corners': lower_corners.tolist(),
                'central_points': disc_params['central_points']
            }
            
            if self.calculate_dwr:
                dwr = disc_params['disc_height'] / disc_params['big_width'] \
                      if disc_params['big_width'] > 0 else 0
                result['dwr'] = dwr
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"DHI计算失败: {str(e)}")
        
    def _extract_central_region(self, disc_mask: np.ndarray,
                                anterior_mid: np.ndarray,
                                posterior_mid: np.ndarray,
                                percentage: float) -> np.ndarray:

        h, w = disc_mask.shape
        center_line_vector = posterior_mid - anterior_mid
        line_length = np.linalg.norm(center_line_vector)
        
        if line_length < 1e-6:
            return np.zeros_like(disc_mask, dtype=np.uint8)
            
        direction_vector = center_line_vector / line_length

        center_point = (anterior_mid + posterior_mid) / 2
        half_width = line_length * percentage / 2
        
        ys, _ = np.where(disc_mask > 0)
        disc_approx_height = (np.max(ys) - np.min(ys)) if len(ys) > 0 else 20
        half_height = disc_approx_height * 0.75 / 2

        perp_vector = np.array([-direction_vector[1], direction_vector[0]])

        p1 = center_point - half_width * direction_vector + half_height * perp_vector
        p2 = center_point + half_width * direction_vector + half_height * perp_vector
        p3 = center_point + half_width * direction_vector - half_height * perp_vector
        p4 = center_point - half_width * direction_vector - half_height * perp_vector
        
        contour = np.array([p1, p2, p3, p4], dtype=np.int32)
        
        central_mask = np.zeros_like(disc_mask, dtype=np.uint8)
        cv2.fillPoly(central_mask, [contour], 255)
        
        return cv2.bitwise_and(central_mask, disc_mask.astype(np.uint8))
    
    def _calculate_vertebral_corners(self, vertebra_mask: np.ndarray) -> np.ndarray:

        if len(vertebra_mask.shape) > 2:
            vertebra_mask = vertebra_mask.squeeze()
            
        mask_uint8 = (vertebra_mask * 255).astype(np.uint8)
        
        kernel = np.ones((3, 3), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        corners = cv2.goodFeaturesToTrack(
            mask_uint8,
            maxCorners=4,
            qualityLevel=0.01,
            minDistance=21,  
            blockSize=9,    
            useHarrisDetector=False,
            k=0.04
        )
        
        if corners is None or len(corners) != 4:
            return self._fallback_corner_detection(vertebra_mask)
            
        corners = np.squeeze(corners).astype(int)
        corners = self._sort_corners_robust(corners)
        
        return corners
    
    def _calculate_s1_corners(self, s1_mask: np.ndarray) -> np.ndarray:
        if len(s1_mask.shape) > 2:
            s1_mask = s1_mask.squeeze()
            
        mask_uint8 = (s1_mask * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        corners = cv2.goodFeaturesToTrack(
            mask_uint8,
            maxCorners=4,
            qualityLevel=0.01,
            minDistance=21,
            blockSize=9,
            useHarrisDetector=False,
            k=0.04
        )
        
        if corners is None or len(corners) != 4:
            return self._fallback_corner_detection(s1_mask)
            
        corners = np.squeeze(corners).astype(int)
        return self._sort_corners_robust(corners)
    
    def _fallback_corner_detection(self, mask: np.ndarray) -> np.ndarray:

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        corners = np.array([
            [cmin, rmin],
            [cmax, rmin],
            [cmin, rmax],
            [cmax, rmax]
        ])
        
        return corners
    
    def _sort_corners_robust(self, corners: np.ndarray) -> np.ndarray:
        corners_copy = corners.copy()
        
        sum_wh = np.sum(corners, axis=1)
        idx_min = np.argmin(sum_wh)  
        idx_max = np.argmax(sum_wh)  
        
        sorted_corners = np.zeros_like(corners)
        sorted_corners[0] = corners[idx_min]  
        sorted_corners[3] = corners[idx_max]  
        
        remaining_indices = [i for i in range(4) if i not in [idx_min, idx_max]]
        remaining_corners = corners[remaining_indices]
        
        if remaining_corners[0, 1] < remaining_corners[1, 1]:
            sorted_corners[1] = remaining_corners[0] 
            sorted_corners[2] = remaining_corners[1]  
        else:
            sorted_corners[1] = remaining_corners[1] 
            sorted_corners[2] = remaining_corners[0] 
        
        if sorted_corners[1, 0] < sorted_corners[0, 0]:  
            sorted_corners[[0, 1]] = sorted_corners[[1, 0]]
        if sorted_corners[3, 0] < sorted_corners[2, 0]:  
            sorted_corners[[2, 3]] = sorted_corners[[3, 2]]
        
        return sorted_corners
    
    def _calculate_disc_parameters(self, disc_mask: np.ndarray,
                                  upper_corners: np.ndarray,
                                  lower_corners: np.ndarray) -> Dict:

        anterior_mid = (upper_corners[2] + lower_corners[0]) / 2
        posterior_mid = (upper_corners[3] + lower_corners[1]) / 2
        
        small_width = np.linalg.norm(posterior_mid - anterior_mid)
        
        if self.consider_bulging:
            big_width = self._calculate_big_width(disc_mask, anterior_mid, posterior_mid)
        else:
            big_width = small_width
        
        central_mask = self._extract_central_region(
            disc_mask, anterior_mid, posterior_mid, self.central_ratio
        )
        
        central_area = np.sum(central_mask > 0)
        
        disc_height = central_area / (small_width * self.central_ratio) if small_width > 0 else 0
        
        central_points = self._calculate_central_division_points(
            anterior_mid, posterior_mid, self.central_ratio
        )
        
        return {
            'disc_height': disc_height,
            'small_width': small_width,
            'big_width': big_width,
            'central_mask': central_mask,
            'central_points': central_points
        }
    
    def _calculate_big_width(self, disc_mask: np.ndarray,
                            anterior_mid: np.ndarray,
                            posterior_mid: np.ndarray) -> float:

        h0, w0 = int(anterior_mid[1]), int(anterior_mid[0])
        h1, w1 = int(posterior_mid[1]), int(posterior_mid[0])
        
        disc_mask = disc_mask.astype(np.uint8)
        height, width = disc_mask.shape
        
        if w1 == w0:
            left_bound = w0
            for w in range(w0, -1, -1):
                if disc_mask[h0, w] == 0:
                    left_bound = w + 1
                    break
            
            right_bound = w1
            for w in range(w1, width):
                if disc_mask[h1, w] == 0:
                    right_bound = w - 1
                    break
            
            big_width = right_bound - left_bound
        else:
            slope = (h1 - h0) / (w1 - w0)
            intercept = h0 - slope * w0
            
            left_point = None
            for w in range(w0, -1, -1):
                h = int(slope * w + intercept)
                if 0 <= h < height:
                    if disc_mask[h, w] == 0:
                        left_point = [h, w + 1]
                        break
            
            right_point = None
            for w in range(w1, width):
                h = int(slope * w + intercept)
                if 0 <= h < height:
                    if disc_mask[h, w] == 0:
                        right_point = [h, w - 1]
                        break
            
            if left_point and right_point:
                big_width = np.linalg.norm(np.array(right_point) - np.array(left_point))
            else:
                big_width = np.linalg.norm(posterior_mid - anterior_mid)
        
        return big_width
    
    def _calculate_central_division_points(self, anterior_mid: np.ndarray,
                                          posterior_mid: np.ndarray,
                                          ratio: float) -> np.ndarray:

        center = (anterior_mid + posterior_mid) / 2
        
        direction = posterior_mid - anterior_mid
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
        
        perpendicular = np.array([-direction_norm[1], direction_norm[0]])
        
        half_length = np.linalg.norm(direction) * ratio / 2
        half_height = 0.75 
        
        points = np.array([
            center - half_length * direction_norm - half_height * perpendicular,  
            center - half_length * direction_norm + half_height * perpendicular,  
            center + half_length * direction_norm - half_height * perpendicular,  
            center + half_length * direction_norm + half_height * perpendicular  
        ])
        
        return points.astype(int)
    
    def _calculate_vertebral_diameter(self, corners: np.ndarray) -> float:

        corners_sorted_by_x = corners[np.argsort(corners[:, 0])]
        anterior_points = corners_sorted_by_x[:2]
        posterior_points = corners_sorted_by_x[2:]
        
        anterior_midpoint = np.mean(anterior_points, axis=0)
        posterior_midpoint = np.mean(posterior_points, axis=0)
        
        return np.linalg.norm(anterior_midpoint - posterior_midpoint)

    def _calculate_vertebral_height(self, vertebra_mask: np.ndarray, diameter: float) -> float:

        area = np.sum(vertebra_mask > 0)
        if diameter < 1e-6:
            return 0
        return area / diameter

    def _calculate_vertebral_diameter(self, corners: np.ndarray) -> float:

        corners_sorted_by_x = corners[np.argsort(corners[:, 0])]
        anterior_points = corners_sorted_by_x[:2]
        posterior_points = corners_sorted_by_x[2:]
        
        anterior_midpoint = np.mean(anterior_points, axis=0)
        posterior_midpoint = np.mean(posterior_points, axis=0)
        
        return np.linalg.norm(anterior_midpoint - posterior_midpoint)

    def _calculate_vertebral_height(self, vertebra_mask: np.ndarray, diameter: float) -> float:

        area = np.sum(vertebra_mask > 0)
        if diameter < 1e-6:
            return 0
        return area / diameter

    def _calculate_dhi(self, disc_height: float, 
                    upper_vertebra_height: float,
                    lower_vertebra_height: float) -> float:

        denominator = upper_vertebra_height + lower_vertebra_height
        if denominator < 1e-6:
            return 0
            
        dhi = 2 * disc_height / denominator
        return dhi
    
    def process_multi_slice(self, upper_masks: List[np.ndarray], 
                           disc_masks: List[np.ndarray],
                           lower_masks: List[np.ndarray],
                           is_l5_s1: bool = False) -> Dict[str, float]:

        if not (len(upper_masks) == len(disc_masks) == len(lower_masks)):
            raise ValueError("切片列表长度不一致")
        
        dhi_results = []
        valid_slices = 0
        
        for i, (upper, disc, lower) in enumerate(zip(upper_masks, disc_masks, lower_masks)):
            try:
                result = self.calculate(upper, disc, lower, is_l5_s1)
                dhi_results.append(result)
                valid_slices += 1
            except Exception as e:
                print(f"切片{i}处理失败: {str(e)}")
                continue
        
        if valid_slices == 0:
            raise ValueError("没有成功处理的切片")
        
        avg_result = {
            'dhi': np.mean([r['dhi'] for r in dhi_results]),
            'dhi_std': np.std([r['dhi'] for r in dhi_results]),
            'disc_height': np.mean([r['disc_height'] for r in dhi_results]),
            'disc_width_small': np.mean([r['disc_width_small'] for r in dhi_results]),
            'disc_width_big': np.mean([r['disc_width_big'] for r in dhi_results]),
            'upper_vh': np.mean([r['upper_vh'] for r in dhi_results]),
            'lower_vh': np.mean([r['lower_vh'] for r in dhi_results]),
            'valid_slices': valid_slices
        }
        
        if self.calculate_dwr:
            avg_result['dwr'] = np.mean([r['dwr'] for r in dhi_results if 'dwr' in r])
            avg_result['dwr_std'] = np.std([r['dwr'] for r in dhi_results if 'dwr' in r])
        
        return avg_result
    
    def process_multi_slice_parallel(self, upper_masks: List[np.ndarray], 
                                disc_masks: List[np.ndarray],
                                lower_masks: List[np.ndarray],
                                is_l5_s1: bool = False) -> Dict[str, float]:
        
        if not self.enable_parallel or len(disc_masks) < 3:
            return self.process_multi_slice(upper_masks, disc_masks, lower_masks, is_l5_s1)
        
        def process_single_slice(args):
            i, upper, disc, lower = args
            try:
                result = self.calculate(upper, disc, lower, is_l5_s1)
                return (i, result, None)
            except Exception as e:
                return (i, None, str(e))
        
        args_list = [(i, upper, disc, lower) 
                    for i, (upper, disc, lower) 
                    in enumerate(zip(upper_masks, disc_masks, lower_masks))]
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(args_list))) as executor:
            results = list(executor.map(process_single_slice, args_list))
        
        dhi_results = []
        valid_slices = 0
        
        for i, result, error in sorted(results, key=lambda x: x[0]):
            if result:
                dhi_results.append(result)
                valid_slices += 1
        
        if valid_slices == 0:
            raise ValueError("没有成功处理的切片")
        
        avg_result = {
            'dhi': np.mean([r['dhi'] for r in dhi_results]),
            'dhi_std': np.std([r['dhi'] for r in dhi_results]),
            'disc_height': np.mean([r['disc_height'] for r in dhi_results]),
            'disc_width_small': np.mean([r['disc_width_small'] for r in dhi_results]),
            'disc_width_big': np.mean([r['disc_width_big'] for r in dhi_results]),
            'upper_vh': np.mean([r['upper_vh'] for r in dhi_results]),
            'lower_vh': np.mean([r['lower_vh'] for r in dhi_results]),
            'valid_slices': valid_slices
        }
        
        if self.calculate_dwr:
            avg_result['dwr'] = np.mean([r['dwr'] for r in dhi_results if 'dwr' in r])
            avg_result['dwr_std'] = np.std([r['dwr'] for r in dhi_results if 'dwr' in r])
        
        return avg_result
