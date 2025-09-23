import numpy as np
from scipy.interpolate import splprep, splev
from scipy import ndimage
import cv2
from typing import Dict, List, Optional, Tuple
import logging
from .base_calculator import BaseCalculator


class DSCRCalculator(BaseCalculator):
    
    def __init__(self, spline_smoothing: float = 0, 
                 spline_degree: int = 2,
                 min_landmarks: int = 3,
                 **kwargs):
        super().__init__("DSCR Calculator")
        self.spline_smoothing = spline_smoothing
        self.spline_degree = spline_degree
        self.min_landmarks = min_landmarks
        
    def calculate(self, disc_mask: np.ndarray, 
                 dural_sac_mask: np.ndarray, 
                 landmark_mask: np.ndarray,
                 disc_level: str = None) -> Dict[str, float]:

        result = {}

        if not np.any(disc_mask):
            self.logger.warning("No disc region found")
            return {'dscr': -1, 'error': 'No disc region'}
            
        if not np.any(dural_sac_mask):
            self.logger.warning("No dural sac region found")
            return {'dscr': -1, 'error': 'No dural sac region'}

        landmark_points = self._extract_landmarks(landmark_mask)
        if len(landmark_points) < self.min_landmarks:
            self.logger.warning(f"Insufficient landmarks: {len(landmark_points)} < {self.min_landmarks}")
            return {'dscr': -1, 'error': 'Insufficient landmarks'}

        ideal_curve = self._fit_ideal_curve(landmark_points)
        if ideal_curve is None:
            return {'dscr': -1, 'error': 'Failed to fit ideal curve'}

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
            
        self.logger.info(f"DSCR calculated: {dscr:.1f}% (d={actual_diameter:.1f}, m={ideal_diameter:.1f})")
        
        return result
    
    def _extract_landmarks(self, landmark_mask: np.ndarray) -> np.ndarray:

        labeled, num_features = ndimage.label(landmark_mask > 0)
        
        landmarks = []
        for i in range(1, num_features + 1):
            coords = np.argwhere(labeled == i)
            if len(coords) > 0:
                centroid = coords.mean(axis=0)
                landmarks.append(centroid)
                
        return np.array(landmarks)
    
    def _fit_ideal_curve(self, landmarks: np.ndarray) -> Optional[Tuple]:

        if len(landmarks) < self.min_landmarks:
            return None

        sorted_indices = np.argsort(landmarks[:, 0])
        sorted_landmarks = landmarks[sorted_indices]
        
        try:

            tck, u = splprep(
                [sorted_landmarks[:, 1], sorted_landmarks[:, 0]], 
                s=self.spline_smoothing,
                k=min(self.spline_degree, len(sorted_landmarks) - 1)
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
                          landmark_masks: List[np.ndarray],
                          disc_level: str = None) -> Dict[str, float]:

        
        all_results = []
        for disc, sac, landmark in zip(disc_masks, dural_sac_masks, landmark_masks):
            result = self.calculate(disc, sac, landmark, disc_level)
            if result and 'dscr' in result and result['dscr'] >= 0:
                all_results.append(result)
                
        if not all_results:
            return {'dscr': -1, 'error': 'No valid DSCR calculated'}

        max_result = max(all_results, key=lambda x: x['dscr'])

        max_result['dscr_mean'] = np.mean([r['dscr'] for r in all_results])
        max_result['dscr_std'] = np.std([r['dscr'] for r in all_results])
        max_result['num_valid_slices'] = len(all_results)
        
        return max_result