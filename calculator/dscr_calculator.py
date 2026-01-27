import numpy as np
from scipy.interpolate import splprep, splev
from scipy import ndimage
import cv2
from typing import Dict, List, Optional, Tuple
import logging
from .base_calculator import BaseCalculator
from config import Config
from scipy.ndimage import distance_transform_edt


class DSCRCalculator(BaseCalculator):
    
    def __init__(self, spline_smoothing: float = 0, 
                 spline_degree: int = 2,
                 min_landmarks: int = 3,
                 **kwargs):
        super().__init__("DSCR Calculator",**kwargs)
        self.spline_smoothing = spline_smoothing
        self.spline_degree = spline_degree
        self.min_landmarks = min_landmarks
        

    def calculate(self, disc_mask: np.ndarray, 
                 dural_sac_mask: np.ndarray, 
                 full_mask: np.ndarray,
                 slice_idx: int,
                 disc_level: str = None) -> Dict[str, float]:

        result = {}

        if not np.any(disc_mask):
            self.logger.warning("No disc region found")
            return {'dscr': -1, 'error': 'No disc region'}
            
        if not np.any(dural_sac_mask):
            self.logger.warning("No dural sac region found")
            return {'dscr': -1, 'error': 'No dural sac region'}

        landmark_points = self._detect_landmarks_auto(full_mask, dural_sac_mask, slice_idx)
        
        if len(landmark_points) < self.min_landmarks:
            self.logger.warning(f"Slice-{slice_idx}: 自动检测到的地标点不足: {len(landmark_points)} < {self.min_landmarks}")
            return {'dscr': -1, 'error': f'Insufficient auto-detected landmarks on slice {slice_idx}'}

        landmarks_yx = landmark_points[:, [1, 0]]
        ideal_curve = self._fit_ideal_curve(landmarks_yx)
        if ideal_curve is None:
            return {'dscr': -1, 'error': f'Failed to fit ideal curve on slice {slice_idx}'}

        disc_center_y = self._find_disc_center(disc_mask)

        ideal_diameter = self._calculate_ideal_diameter(
            ideal_curve, dural_sac_mask, disc_center_y
        )

        actual_diameter = self._calculate_actual_diameter(
            dural_sac_mask, disc_center_y
        )

        if ideal_diameter > 0:
            dscr = (1 - actual_diameter / ideal_diameter) * 100
            dscr = np.clip(dscr, 0, 100)
        else:
            dscr = 0
            
        result = {
            'dscr': dscr,
            'actual_diameter_pixels': actual_diameter,
            'ideal_diameter_pixels': ideal_diameter,
            'disc_center_y': disc_center_y,
            'num_landmarks': len(landmark_points)
        }
        
        if disc_level:
            result['disc_level'] = disc_level
            
        self.logger.info(f"Slice-{slice_idx}, {disc_level}: DSCR calculated: {dscr:.1f}% (d={actual_diameter:.1f}, m={ideal_diameter:.1f})")
        
        return result
    
    def _fallback_corner_detection(self, mask: np.ndarray) -> np.ndarray:

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return np.array([])
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        corners = np.array([
            [cmin, rmin], [cmax, rmin],
            [cmin, rmax], [cmax, rmax]
        ])
        return corners

    def _find_vertebra_corners(self, vertebra_mask: np.ndarray) -> Optional[np.ndarray]:

        if not np.any(vertebra_mask):
            return None
            
        mask_uint8 = (vertebra_mask * 255).astype(np.uint8)
        
        kernel = np.ones((3, 3), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        corners = cv2.goodFeaturesToTrack(
            mask_uint8,
            maxCorners=4,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=9,
            useHarrisDetector=False,
            k=0.04
        )
        
        if corners is None or len(corners) < 4:
            self.logger.warning("标准角点检测失败，启用备用方法。")
            corners = self._fallback_corner_detection(vertebra_mask)
            if corners.shape[0] < 4:
                return None
            return corners
        
        return np.squeeze(corners).astype(int)
    

    def _fallback_corner_detection(self, mask: np.ndarray) -> np.ndarray:

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return np.array([])
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        corners = np.array([
            [cmin, rmin], [cmax, rmin],
            [cmin, rmax], [cmax, rmax]
        ])
        return corners

    def _find_vertebra_corners(self, vertebra_mask: np.ndarray) -> Optional[np.ndarray]:

        if not np.any(vertebra_mask):
            return None
            
        mask_uint8 = (vertebra_mask * 255).astype(np.uint8)
        
        kernel = np.ones((3, 3), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        corners = cv2.goodFeaturesToTrack(
            mask_uint8,
            maxCorners=4,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=9,
            useHarrisDetector=False,
            k=0.04
        )
        
        if corners is None or len(corners) < 4:
            self.logger.warning("标准角点检测失败，启用备用方法。")
            corners = self._fallback_corner_detection(vertebra_mask)
            if corners.shape[0] < 4:
                return None
        
        return np.squeeze(corners).astype(int)

    
    def _detect_single_vertebra_landmark(self, vertebra_mask: np.ndarray, dural_sac_mask: np.ndarray, vert_name: str, slice_idx: int) -> Optional[Tuple[float, float]]:

        self.logger.info(f"Slice-{slice_idx}, {vert_name}: 开始自动地标检测 (距离法)...")

        corners = self._find_vertebra_corners(vertebra_mask)
        if corners is None or len(corners) < 4:
            self.logger.warning(f"Slice-{slice_idx}, {vert_name} [Step 1]: 未能检测到4个角点，跳过此椎体。")
            return None
        self.logger.info(f"Slice-{slice_idx}, {vert_name} [Step 1]: 检测到4个原始角点: {corners.tolist()}")

        if not np.any(dural_sac_mask):
            self.logger.warning(f"Slice-{slice_idx}, {vert_name} [Step 2]: 硬脊膜囊掩码为空，无法计算距离。")
            return None

        distance_map = distance_transform_edt(dural_sac_mask == 0)
        
        distances = []
        for corner in corners:
            cx, cy = corner[0], corner[1]
            if 0 <= cy < distance_map.shape[0] and 0 <= cx < distance_map.shape[1]:
                dist = distance_map[cy, cx]
                distances.append(dist)
            else:
                distances.append(np.inf)

        self.logger.info(f"Slice-{slice_idx}, {vert_name} [Step 2]: 各角点到椎管的距离: {['%.2f' % d for d in distances]}")

        if len(distances) < 2:
            self.logger.warning(f"Slice-{slice_idx}, {vert_name} [Step 3]: 有效距离计算少于2个，无法确定后缘。")
            return None
            
        sorted_indices = np.argsort(distances)
        posterior_indices = sorted_indices[:2]
        posterior_corners = corners[posterior_indices]

        self.logger.info(f"Slice-{slice_idx}, {vert_name} [Step 3]: 识别出的后缘角点为: {posterior_corners.tolist()}")

        midpoint_y = np.mean(posterior_corners[:, 1])
        y_final_int = int(round(midpoint_y))
        
        self.logger.info(f"Slice-{slice_idx}, {vert_name} [Step 4]: 后缘角点中点的Y坐标为 {midpoint_y:.2f} (取整为 {y_final_int})")

        if not (0 <= y_final_int < vertebra_mask.shape[0]):
             self.logger.warning(f"Slice-{slice_idx}, {vert_name} [Step 4]: 计算出的Y值 {y_final_int} 超出图像范围。")
             return None

        line_vert_x_coords = np.where(vertebra_mask[y_final_int, :])[0]
        if len(line_vert_x_coords) == 0:
            self.logger.warning(f"Slice-{slice_idx}, {vert_name} [Step 5]: 在Y={y_final_int}水平线上未找到椎体像素。")
            return None
        self.logger.info(f"Slice-{slice_idx}, {vert_name} [Step 5]: 在Y={y_final_int}线上找到 {len(line_vert_x_coords)} 个椎体像素。")

        sac_pixels_on_line = np.where(dural_sac_mask[y_final_int, :])[0]
        
        if len(sac_pixels_on_line) == 0:
            self.logger.warning(f"Slice-{slice_idx}, {vert_name} [Step 6]: 在Y={y_final_int}水平线上未找到椎管像素，无法进行过滤。")
            return None
        
        sac_right_boundary = sac_pixels_on_line.max()
        
        filtered_line_x_coords = line_vert_x_coords[line_vert_x_coords < sac_right_boundary]
        
        self.logger.info(f"Slice-{slice_idx}, {vert_name} [Step 6]: 椎管右边界为 x={sac_right_boundary}。过滤后剩下 {len(filtered_line_x_coords)} 个有效椎体像素。")

        if len(filtered_line_x_coords) == 0:
            self.logger.warning(f"Slice-{slice_idx}, {vert_name} [Step 6]: 过滤后无有效椎体像素（可能所有椎体像素都在椎管之后）。")
            return None
        
        x_final = np.percentile(filtered_line_x_coords, 99)
        
        final_y = midpoint_y
        
        self.logger.info(f"Slice-{slice_idx}, {vert_name} [Step 7]: 最终地标点确定为 (x={x_final:.2f}, y={final_y:.2f})")
        
        return (final_y, x_final)

    def _detect_landmarks_auto(self, full_mask: np.ndarray, dural_sac_mask: np.ndarray, slice_idx: int) -> np.ndarray:

        landmarks = []
        
        vertebra_map = {
            'L1': Config.DISC_LABELS['L1-L2']['upper'],
            'L2': Config.DISC_LABELS['L2-L3']['upper'],
            'L3': Config.DISC_LABELS['L3-L4']['upper'],
            'L4': Config.DISC_LABELS['L4-L5']['upper'],
            'L5': Config.DISC_LABELS['L5-S1']['upper']
        }

        for vert_name, vert_label in vertebra_map.items():
            vertebra_mask = (full_mask == vert_label)
            landmark_yx = self._detect_single_vertebra_landmark(vertebra_mask, dural_sac_mask, vert_name, slice_idx)
            
            if landmark_yx:
                landmarks.append((landmark_yx[1], landmark_yx[0]))

        if not landmarks:
            return np.array([])
            
        return np.array(landmarks)
    

    def _fit_ideal_curve(self, landmarks_yx: np.ndarray) -> Optional[Tuple]:

        if len(landmarks_yx) < self.min_landmarks:
            return None

        sorted_indices = np.argsort(landmarks_yx[:, 0])
        sorted_landmarks_yx = landmarks_yx[sorted_indices]
        
        landmarks_for_spline = [sorted_landmarks_yx[:, 1], sorted_landmarks_yx[:, 0]]
        
        try:
            tck, u = splprep(
                landmarks_for_spline, 
                s=self.spline_smoothing,
                k=min(self.spline_degree, len(sorted_landmarks_yx) - 1)
            )
            return (tck, u)
        except Exception as e:
            self.logger.error(f"Spline fitting failed: {e}")
            return None
    
    def _find_disc_center(self, disc_mask: np.ndarray) -> int:

        disc_rows = np.where(np.any(disc_mask, axis=1))[0]
        if len(disc_rows) == 0:
            return disc_mask.shape[0] // 2
        return int(np.mean([disc_rows.min(), disc_rows.max()]))
    
    def _calculate_ideal_diameter(self, ideal_curve: Tuple,
                                 dural_sac_mask: np.ndarray,
                                 center_y: int) -> float:

        tck, u = ideal_curve

        num_points = 1000
        x_curve, y_curve = splev(np.linspace(0, 1, num_points), tck)

        distances = np.abs(y_curve - center_y)
        closest_idx = np.argmin(distances)
        ideal_posterior_x = x_curve[closest_idx]

        sac_line = dural_sac_mask[center_y, :]
        sac_pixels = np.where(sac_line > 0)[0]
        
        if len(sac_pixels) == 0:
            return 0
            
        actual_posterior_x = sac_pixels.max()

        ideal_diameter = abs(actual_posterior_x - ideal_posterior_x)
        
        return ideal_diameter
    
    def _calculate_actual_diameter(self, dural_sac_mask: np.ndarray,
                                  center_y: int) -> float:

        sac_line = dural_sac_mask[center_y, :]
        sac_pixels = np.where(sac_line > 0)[0]
        
        if len(sac_pixels) == 0:
            return 0

        actual_diameter = sac_pixels.max() - sac_pixels.min()
        
        return actual_diameter
    

    def process_multi_slice(self, disc_masks: List[np.ndarray],
                          dural_sac_masks: List[np.ndarray],
                          full_masks: List[np.ndarray],
                          disc_level: str = None) -> Dict[str, float]:

        all_results = []
        for i, (disc, sac, full) in enumerate(zip(disc_masks, dural_sac_masks, full_masks)):
            result = self.calculate(disc, sac, full, slice_idx=i, disc_level=disc_level)
            if result and 'dscr' in result and result['dscr'] > 0:
                all_results.append(result)

        if not all_results:
            return {
                'dscr': np.nan,
                'dscr_std': np.nan,
                'dscr_max': np.nan,
                'actual_diameter_pixels': np.nan,
                'ideal_diameter_pixels': np.nan,
                'disc_center_y': np.nan,
                'num_landmarks': np.nan,
                'num_valid_slices': 0,
                'error': None,
            }

        dscr_values = [r['dscr'] for r in all_results]

        final_result = {
            'dscr': float(np.mean(dscr_values)),
            'dscr_std': float(np.std(dscr_values)),
            'dscr_max': float(np.max(dscr_values)),
        }

        final_result.update({k: v for k, v in all_results[0].items() if k not in final_result})

        return final_result
