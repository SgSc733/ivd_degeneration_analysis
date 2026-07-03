import numpy as np
import cv2
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor  # ✅ 修复并行路径缺失的import

from .base_calculator import BaseCalculator
from utils.memory_monitor import monitor_memory


class HuMomentsCalculator(BaseCalculator):

    def __init__(self, **kwargs):
        super().__init__("Hu Moments Calculator", **kwargs)

    @monitor_memory(threshold_percent=90)
    def calculate(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        self.validate_input(image, mask)

        binary_mask = (mask > 0).astype(np.uint8) * 255

        moments = cv2.moments(binary_mask)

        hu_moments = cv2.HuMoments(moments)
        hu_moments_log = np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        features: Dict[str, float] = {}
        for i in range(7):
            features[f'hu_moment_{i+1}'] = float(hu_moments[i][0])
            features[f'hu_moment_log_{i+1}'] = float(hu_moments_log[i][0])

        shape_features = self._calculate_shape_features(binary_mask)
        features.update(shape_features)

        try:
            prognostic_features = self._calculate_prognostic_features(
                binary_mask=binary_mask,
                hu_moments=hu_moments.flatten(),
                hu_moments_log=hu_moments_log.flatten(),
                shape_features=shape_features
            )
            features.update(prognostic_features)
        except Exception as e:
            self.logger.warning(f"预后特征计算失败: {str(e)}")
            features.update({
                'shape_stability_index': 0.0,
                'load_distribution_index': 0.0,
                'shape_degeneration_risk': 0.0,
                'annular_integrity_iou': 0.0
            })

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

            lambda1 = 0.5 * ((mu20 + mu02) + np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2))
            lambda2 = 0.5 * ((mu20 + mu02) - np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2))

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

    def _calculate_prognostic_features(
        self,
        binary_mask: np.ndarray,
        hu_moments: np.ndarray,
        hu_moments_log: np.ndarray,
        shape_features: Dict[str, float]
    ) -> Dict[str, float]:

        epsilon = 1e-8

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return {
                'shape_stability_index': 0.0,
                'load_distribution_index': 0.0,
                'shape_degeneration_risk': 0.0,
                'annular_integrity_iou': 0.0
            }

        contour = max(contours, key=cv2.contourArea)

        sigma_log = float(np.std(hu_moments_log))
        ssi = 1.0 / (sigma_log + 1e-6)
        ssi = float(np.clip(ssi, 0.0, 1.0))

        ldi = 1.0
        try:
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if hull_indices is not None and len(hull_indices) > 3 and len(contour) > 3:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None and len(defects) > 0:
                    depths = defects[:, 0, 3] / 256.0
                    irregularity = float(np.mean(depths))
                    irregularity = float(np.clip(irregularity, 0.0, 1.0))
                    ldi = 1.0 - irregularity
                    ldi = float(np.clip(ldi, 0.0, 1.0))
        except Exception:
            ldi = 1.0

        ecc_risk = float(np.clip(shape_features.get('eccentricity', 0.0), 0.0, 1.0))

        density = float(np.clip(shape_features.get('solidity', 1.0), 0.0, 1.0))
        density_risk = float(np.clip(1.0 - density, 0.0, 1.0))

        hu_asymmetry = float((hu_moments[1] + hu_moments[2]) / (hu_moments[0] + epsilon))
        hu_asymmetry = float(np.clip(hu_asymmetry, 0.0, 1.0))

        sdrs = 0.3 * ecc_risk + 0.3 * density_risk + 0.4 * hu_asymmetry
        sdrs = float(np.clip(sdrs, 0.0, 1.0))

        aia_iou = 0.0
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)

                ellipse_mask = np.zeros_like(binary_mask)
                cv2.ellipse(ellipse_mask, ellipse, 255, -1)

                intersection = cv2.bitwise_and(binary_mask, ellipse_mask)
                union = cv2.bitwise_or(binary_mask, ellipse_mask)

                area_intersection = int(np.sum(intersection > 0))
                area_union = int(np.sum(union > 0))

                aia_iou = float(area_intersection / (area_union + epsilon))
            except Exception:
                aia_iou = 0.0

        return {
            'shape_stability_index': float(ssi),
            'load_distribution_index': float(ldi),
            'shape_degeneration_risk': float(sdrs),
            'annular_integrity_iou': float(aia_iou)
        }

    def process_multi_slice(
        self,
        image_slices: List[np.ndarray],
        masks: List[np.ndarray],
        use_parallel: Optional[bool] = None
    ) -> Dict[str, float]:
        if use_parallel is None:
            use_parallel = self.enable_parallel

        if use_parallel and len(image_slices) >= 5:
            return self.process_multi_slice_parallel(image_slices, masks)

        hu_features: Dict[str, List[float]] = {}
        for i, (img, mask) in enumerate(zip(image_slices, masks)):
            try:
                slice_features = self.calculate(img, mask)
            except Exception as exc:
                self.logger.warning(f"切片{i}处理失败: {exc}")
                continue
            for k, v in slice_features.items():
                if k in hu_features:
                    hu_features[k].append(v)
                else:
                    hu_features[k] = [v]

        if not hu_features:
            return {}

        return {k: float(np.mean(v)) for k, v in hu_features.items()}

    def process_multi_slice_parallel(
        self,
        image_slices: List[np.ndarray],
        masks: List[np.ndarray]
    ) -> Dict[str, float]:
        def process_single_slice(args):
            i, img, mask = args
            try:
                result = self.calculate(img, mask)
                return (i, result, None)
            except Exception as e:
                return (i, None, str(e))

        args_list = [
            (i, img, mask)
            for i, (img, mask) in enumerate(zip(image_slices, masks))
        ]

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(args_list))) as executor:
            results = list(executor.map(process_single_slice, args_list))

        all_features: Dict[str, List[float]] = {}
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

        hu_result = {k: float(np.mean(v)) for k, v in all_features.items()}
        hu_result['valid_slices'] = valid_slices

        return hu_result
