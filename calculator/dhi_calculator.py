import math
import numpy as np
import cv2
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from .base_calculator import BaseCalculator
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

        disc = disc_mask.astype(np.uint8).copy()
        height, width = disc.shape

        points_xy = self._calculate_central_division_points(anterior_mid, posterior_mid, percentage)
        points_hw = points_xy[:, [1, 0]].astype(int)

        lu, ld, ru, rd = points_hw
        if lu[0] > ld[0]:
            lu, ld = ld, lu
        if ru[0] > rd[0]:
            ru, rd = rd, ru

        l_h, l_w = self._get_pixel_hw(int(lu[0]), int(lu[1]), int(ld[0]), int(ld[1]))
        for hh, ww in zip(l_h, l_w):
            if 0 <= hh < height:
                ww = int(np.clip(ww, 0, width))
                disc[hh, :ww] = 0

        r_h, r_w = self._get_pixel_hw(int(ru[0]), int(ru[1]), int(rd[0]), int(rd[1]))
        for hh, ww in zip(r_h, r_w):
            if 0 <= hh < height:
                ww = int(np.clip(ww, 0, width))
                disc[hh, ww:] = 0

        return disc
    
    def _calculate_vertebral_corners(self, vertebra_mask: np.ndarray) -> np.ndarray:

        if len(vertebra_mask.shape) > 2:
            vertebra_mask = vertebra_mask.squeeze()
            
        mask_f32 = np.float32(vertebra_mask)

        corners = cv2.goodFeaturesToTrack(
            mask_f32,
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
            
        mask_f32 = np.float32(s1_mask)

        corners = cv2.goodFeaturesToTrack(
            mask_f32,
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
        return self._sort_corners_robust(corners, is_s1=True)
    
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
    
    def _sort_corners_robust(self, corners: np.ndarray, is_s1: bool = False) -> np.ndarray:

        sum_wh = np.sum(corners, axis=1)
        idx_min = int(np.argmin(sum_wh))
        idx_max = int(np.argmax(sum_wh))

        sorted_corners = np.zeros_like(corners)
        sorted_corners[0] = corners[idx_min]
        sorted_corners[3] = corners[idx_max]

        remaining_indices = [i for i in range(4) if i not in [idx_min, idx_max]]
        remaining = corners[remaining_indices]

        if remaining.shape != (2, 2):
            return sorted_corners

        if is_s1:
            if np.sum(remaining[0]) > np.sum(remaining[1]):
                sorted_corners[2] = remaining[0]
                sorted_corners[1] = remaining[1]
            else:
                sorted_corners[2] = remaining[1]
                sorted_corners[1] = remaining[0]
        else:
            if remaining[0, 1] > remaining[1, 1]:
                sorted_corners[2] = remaining[0]
                sorted_corners[1] = remaining[1]
            else:
                sorted_corners[2] = remaining[1]
                sorted_corners[1] = remaining[0]

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

        disc_height = central_area / small_width if small_width > 1e-6 else 0
        
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

        disc = np.asarray(disc_mask, dtype=np.uint8)
        height, width = disc.shape

        p1_hw = np.array([float(anterior_mid[1]), float(anterior_mid[0])])
        p2_hw = np.array([float(posterior_mid[1]), float(posterior_mid[0])])

        h0 = int(np.clip(int(p1_hw[0]), 0, height - 1))
        w0 = int(np.clip(int(p1_hw[1]), 0, width - 1))
        h1 = int(np.clip(int(p2_hw[0]), 0, height - 1))
        w1 = int(np.clip(int(p2_hw[1]), 0, width - 1))

        point_you = [h1, w1]
        point_zuo = [h0, w0]

        if h0 == h1:
            for s in range(w1, width):
                if disc[h1, s] == 0:
                    point_you = [h1, s - 1]
                    break

            for t in range(w0, 0, -1):
                if disc[h0, t] == 0:
                    point_zuo = [h0, t + 1]
                    break
        else:
            if abs(p2_hw[1] - p1_hw[1]) < 1e-6:
                return float(np.linalg.norm(p2_hw - p1_hw))

            m = (p2_hw[0] - p1_hw[0]) / (p2_hw[1] - p1_hw[1])
            b = p1_hw[0] - m * p1_hw[1]

            for s in range(w1, width):
                hh = int(m * s + b)
                if 0 <= hh < height and disc[hh, s] == 0:
                    point_you = [int(m * (s - 1) + b), s - 1]
                    break

            for t in range(w0, 0, -1):
                hh = int(m * t + b)
                if 0 <= hh < height and disc[hh, t] == 0:
                    point_zuo = [int(m * (t + 1) + b), t + 1]
                    break

        diff = np.array(point_you) - np.array(point_zuo)
        return math.hypot(float(diff[0]), float(diff[1]))
    
    def _get_pixel_hw(self, h0: int, w0: int, h1: int, w1: int) -> Tuple[List[int], List[int]]:

        piont_h: List[int] = []
        piont_w: List[int] = []

        if h0 == h1 and w0 == w1:
            return [h0], [w0]

        if h0 > h1:
            h0, h1 = h1, h0
            w0, w1 = w1, w0

        if w0 == w1:
            for j in range(h0, h1):
                piont_h.append(j)
                piont_w.append(w0)
            return piont_h, piont_w

        m = (h1 - h0) / (w1 - w0)
        b = h0 - m * w0

        for j in range(h0, h1):
            w_temp = (j - b) / m
            piont_w.append(int(round(w_temp)))
            piont_h.append(j)

        return piont_h, piont_w

    def _calculate_central_division_points(self, anterior_mid: np.ndarray,
                                          posterior_mid: np.ndarray,
                                          ratio: float) -> np.ndarray:

        p1_hw = np.array([float(anterior_mid[1]), float(anterior_mid[0])])
        p2_hw = np.array([float(posterior_mid[1]), float(posterior_mid[0])])

        delta_h = abs(float(p1_hw[0] - p2_hw[0]))
        delta_w = abs(float(p1_hw[1] - p2_hw[1]))

        c0 = (p1_hw + p2_hw) / 2.0

        miu_half = 0.5 * float(ratio)
        qiang_half = 0.5 * 0.75

        if p1_hw[0] < p2_hw[0]:
            c0lu = np.array([c0[0] - miu_half * delta_h - qiang_half * delta_w,
                             c0[1] - miu_half * delta_w + qiang_half * delta_h])
            c0ld = np.array([c0[0] - miu_half * delta_h + qiang_half * delta_w,
                             c0[1] - miu_half * delta_w - qiang_half * delta_h])
            c0ru = np.array([c0[0] + miu_half * delta_h - qiang_half * delta_w,
                             c0[1] + miu_half * delta_w + qiang_half * delta_h])
            c0rd = np.array([c0[0] + miu_half * delta_h + qiang_half * delta_w,
                             c0[1] + miu_half * delta_w - qiang_half * delta_h])
        else:
            c0lu = np.array([c0[0] + miu_half * delta_h - qiang_half * delta_w,
                             c0[1] - miu_half * delta_w - qiang_half * delta_h])
            c0ld = np.array([c0[0] + miu_half * delta_h + qiang_half * delta_w,
                             c0[1] - miu_half * delta_w + qiang_half * delta_h])
            c0ru = np.array([c0[0] - miu_half * delta_h - qiang_half * delta_w,
                             c0[1] + miu_half * delta_w - qiang_half * delta_h])
            c0rd = np.array([c0[0] - miu_half * delta_h + qiang_half * delta_w,
                             c0[1] + miu_half * delta_w + qiang_half * delta_h])

        points_hw = np.array([np.int0(c0lu), np.int0(c0ld), np.int0(c0ru), np.int0(c0rd)], dtype=int)
        return points_hw[:, [1, 0]]
    
    def _calculate_vertebral_diameter(self, corners: np.ndarray) -> float:

        point_h = corners[:, 1].astype(float)
        point_w = corners[:, 0].astype(float)

        wid_up = math.hypot(point_h[0] - point_h[1], point_w[0] - point_w[1])
        wid_down = math.hypot(point_h[2] - point_h[3], point_w[2] - point_w[3])
        return (wid_up + wid_down) / 2.0

    def _calculate_vertebral_height(self, vertebra_mask: np.ndarray, diameter: float) -> float:
        area = float(np.sum(vertebra_mask))
        return area / diameter if diameter > 1e-6 else 0.0

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
