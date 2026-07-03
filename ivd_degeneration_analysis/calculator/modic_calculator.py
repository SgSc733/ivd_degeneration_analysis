import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from skimage import feature
from concurrent.futures import ThreadPoolExecutor

from .base_calculator import BaseCalculator
from utils.memory_monitor import monitor_memory


class MODICCalculator(BaseCalculator):

    def __init__(
        self,
        endplate_ratio: float = 0.15,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        **kwargs
    ):
        super().__init__("MODIC Calculator", enable_parallel=enable_parallel, **kwargs)
        self.endplate_ratio = float(endplate_ratio)
        if max_workers is not None:
            self.max_workers = int(max_workers)

    @monitor_memory(threshold_percent=85)
    def calculate(
        self,
        t2_image: np.ndarray,
        vertebra_mask: np.ndarray,
        t1_image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:

        self.validate_input(t2_image, vertebra_mask)

        upper_roi, lower_roi = self._extract_endplate_rois(vertebra_mask)

        results: List[Dict[str, Any]] = []

        vertebra_t2_pixels = t2_image[vertebra_mask > 0]
        ref_t2_mean = float(np.mean(vertebra_t2_pixels)) if vertebra_t2_pixels.size > 0 else 0.0

        ref_t1_mean = 0.0
        if t1_image is not None:
            self.validate_input(t1_image, vertebra_mask)
            vertebra_t1_pixels = t1_image[vertebra_mask > 0]
            ref_t1_mean = float(np.mean(vertebra_t1_pixels)) if vertebra_t1_pixels.size > 0 else 0.0

        for loc_name, roi_mask in (("upper", upper_roi), ("lower", lower_roi)):
            if np.sum(roi_mask) == 0:
                continue

            feats = self._calculate_roi_features(t2_image, roi_mask, ref_t2_mean)

            modic_type, severity = self._classify_modic_type(
                feats=feats,
                ref_t2=ref_t2_mean,
                t1_img=t1_image,
                roi_mask=roi_mask,
                ref_t1=ref_t1_mean
            )

            feats["modic_type"] = int(modic_type)
            feats["modic_severity"] = float(severity)
            feats["location"] = str(loc_name)

            results.append(feats)

        output: Dict[str, Any] = {"num_regions": int(len(results))}
        for i, res in enumerate(results):
            prefix = f"region_{i}_{res['location']}"
            for k, v in res.items():
                if isinstance(v, (int, float, str)):
                    output[f"{prefix}_{k}"] = v

        output["_raw_regions"] = results
        return output

    def _extract_endplate_rois(self, vert_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        coords = np.column_stack(np.where(vert_mask > 0))
        if coords.size == 0:
            return (
                np.zeros_like(vert_mask, dtype=np.uint8),
                np.zeros_like(vert_mask, dtype=np.uint8),
            )

        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)

        height = int(max_y - min_y)
        roi_height = int(round(height * self.endplate_ratio))
        roi_height = max(1, roi_height)

        upper_mask = np.zeros_like(vert_mask, dtype=np.uint8)
        lower_mask = np.zeros_like(vert_mask, dtype=np.uint8)

        upper_rect = vert_mask[min_y: min_y + roi_height, min_x: max_x + 1]
        upper_mask[min_y: min_y + roi_height, min_x: max_x + 1] = (upper_rect > 0).astype(np.uint8)

        lower_rect = vert_mask[max_y - roi_height: max_y + 1, min_x: max_x + 1]
        lower_mask[max_y - roi_height: max_y + 1, min_x: max_x + 1] = (lower_rect > 0).astype(np.uint8)

        return upper_mask, lower_mask

    def _detect_modic_area(self, image: np.ndarray, roi_mask: np.ndarray, ref_t2: float) -> np.ndarray:

        if not np.isfinite(ref_t2) or abs(ref_t2) < 1e-8:
            return np.zeros_like(roi_mask, dtype=np.uint8)

        high_thr = ref_t2 * 1.15
        low_thr = ref_t2 * 0.85

        abnormal = ((image > high_thr) | (image < low_thr)) & (roi_mask > 0)
        return abnormal.astype(np.uint8)

    def _calculate_roi_features(self, image: np.ndarray, mask: np.ndarray, ref_t2: float) -> Dict[str, float]:
        pixels = image[mask > 0]
        if pixels.size == 0:
            return {}

        mean_i = float(np.mean(pixels))
        std_i = float(np.std(pixels))
        min_i = float(np.min(pixels))
        max_i = float(np.max(pixels))

        modic_area = self._detect_modic_area(image, mask, ref_t2)
        modic_size = float(np.sum(modic_area > 0))

        heterogeneity = float(std_i / (mean_i + 1e-8))

        y, x = np.where(mask > 0)
        ymin, ymax = int(y.min()), int(y.max())
        xmin, xmax = int(x.min()), int(x.max())

        roi_img = image[ymin: ymax + 1, xmin: xmax + 1].astype(np.float32, copy=False)
        roi_mask = (mask[ymin: ymax + 1, xmin: xmax + 1] > 0)

        lbp = feature.local_binary_pattern(roi_img, P=8, R=1, method="uniform")
        lbp_vals = lbp[roi_mask]
        if lbp_vals.size == 0:
            lbp_entropy = 0.0
        else:
            hist, _ = np.histogram(lbp_vals.ravel(), bins=np.arange(0, 11), density=True)
            lbp_entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))

        grad_x = cv2.Sobel(roi_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi_img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        mean_gradient = float(np.mean(grad_mag[roi_mask])) if np.any(roi_mask) else 0.0

        return {
            "modic_mean_intensity": mean_i,
            "modic_std_intensity": std_i,
            "modic_intensity_range": float(max_i - min_i),
            "modic_size": modic_size,
            "modic_heterogeneity": heterogeneity,
            "modic_lbp_entropy": float(lbp_entropy),
            "modic_mean_gradient": mean_gradient,
        }

    def _classify_modic_type(
        self,
        feats: Dict[str, Any],
        ref_t2: float,
        t1_img: Optional[np.ndarray],
        roi_mask: np.ndarray,
        ref_t1: float
    ) -> Tuple[int, float]:

        mean_t2 = float(feats.get("modic_mean_intensity", 0.0))
        heterogeneity = float(feats.get("modic_heterogeneity", 0.0))

        is_high_t2 = mean_t2 > (ref_t2 * 1.15)
        is_low_t2 = mean_t2 < (ref_t2 * 0.85)

        is_high_het = heterogeneity > 0.25

        if t1_img is not None:
            t1_pixels = t1_img[roi_mask > 0]
            mean_t1 = float(np.mean(t1_pixels)) if t1_pixels.size > 0 else 0.0
            is_high_t1 = mean_t1 > (ref_t1 * 1.15)
            is_low_t1 = mean_t1 < (ref_t1 * 0.85)

            if is_high_t2 and is_low_t1:
                return 1, 1.0
            if is_high_t2 and is_high_t1:
                return 2, 0.6
            if is_low_t2 and is_low_t1:
                return 3, 0.7

        if is_high_t2 and is_high_het:
            return 1, 0.9
        if is_high_t2 and (not is_high_het):
            return 2, 0.5
        if is_low_t2:
            return 3, 0.6

        return 0, 0.0

    def process_multi_slice(
        self,
        t2_images: List[np.ndarray],
        vertebra_masks: List[np.ndarray],
        t1_images: Optional[List[np.ndarray]] = None,
        use_parallel: Optional[bool] = None
    ) -> Dict[str, float]:

        if use_parallel is None:
            use_parallel = self.enable_parallel

        if use_parallel and len(t2_images) >= 5:
            return self.process_multi_slice_parallel(t2_images, vertebra_masks, t1_images)

        if t1_images is None:
            t1_images = [None] * len(t2_images)

        all_regions: List[Dict[str, Any]] = []
        for t2, v_mask, t1 in zip(t2_images, vertebra_masks, t1_images):
            res = self.calculate(t2, v_mask, t1)
            all_regions.extend(res.get("_raw_regions", []))

        valid_regions = [r for r in all_regions if int(r.get("modic_type", 0)) in (1, 2, 3)]
        if not valid_regions:
            return self._get_empty_result()

        prognostic_feats = self._calculate_prognostic_features(valid_regions)

        avg_feats: Dict[str, float] = {}
        for k in ["modic_mean_intensity", "modic_size", "modic_heterogeneity"]:
            vals = [float(r.get(k, 0.0)) for r in valid_regions]
            avg_feats[f"avg_{k}"] = float(np.mean(vals)) if len(vals) > 0 else 0.0

        return {**avg_feats, **prognostic_feats}

    @monitor_memory(threshold_percent=80)
    def process_multi_slice_parallel(
        self,
        t2_images: List[np.ndarray],
        vertebra_masks: List[np.ndarray],
        t1_images: Optional[List[np.ndarray]] = None
    ) -> Dict[str, float]:

        if not self.enable_parallel or len(t2_images) < 2:
            return self.process_multi_slice(t2_images, vertebra_masks, t1_images, use_parallel=False)

        if t1_images is None:
            t1_images = [None] * len(t2_images)

        args_list = list(zip(t2_images, vertebra_masks, t1_images))

        def _task(args):
            return self.calculate(*args)

        max_workers = getattr(self, "max_workers", None)
        if not isinstance(max_workers, int) or max_workers <= 0:
            max_workers = min(4, len(args_list))

        with ThreadPoolExecutor(max_workers=min(max_workers, len(args_list))) as executor:
            results = list(executor.map(_task, args_list))

        all_regions: List[Dict[str, Any]] = []
        for res in results:
            all_regions.extend(res.get("_raw_regions", []))

        valid_regions = [r for r in all_regions if int(r.get("modic_type", 0)) in (1, 2, 3)]
        if not valid_regions:
            return self._get_empty_result()

        prognostic_feats = self._calculate_prognostic_features(valid_regions)

        avg_feats: Dict[str, float] = {}
        for k in ["modic_mean_intensity", "modic_size", "modic_heterogeneity"]:
            vals = [float(r.get(k, 0.0)) for r in valid_regions]
            avg_feats[f"avg_{k}"] = float(np.mean(vals)) if len(vals) > 0 else 0.0

        return {**avg_feats, **prognostic_feats}

    def _calculate_prognostic_features(self, regions: List[Dict[str, Any]]) -> Dict[str, float]:

        total_regions = len(regions)
        if total_regions == 0:
            return self._get_empty_result()

        types = [int(r.get("modic_type", 0)) for r in regions]
        severities = [float(r.get("modic_severity", 0.0)) for r in regions]

        count_type_1 = types.count(1)
        count_type_3 = types.count(3)

        ratio_type_1 = float(count_type_1 / total_regions)
        ratio_type_3 = float(count_type_3 / total_regions)

        mean_severity = float(np.mean(severities)) if severities else 0.0
        std_severity = float(np.std(severities)) if severities else 0.0

        risk_count = min(total_regions / 10.0, 1.0)
        risk_type1 = ratio_type_1
        risk_severity = mean_severity

        modic_prognostic_risk = float(0.3 * risk_count + 0.4 * risk_type1 + 0.3 * risk_severity)
        modic_inflammatory_activity = float(ratio_type_1 * mean_severity)

        dist_stability = float(np.clip(1.0 - std_severity, 0.0, 1.0))
        term_severity_inv = float(np.clip(1.0 - mean_severity, 0.0, 1.0))

        modic_structural_stability = float(
            0.4 * ratio_type_3 +
            0.3 * dist_stability +
            0.3 * term_severity_inv
        )

        return {
            "modic_prognostic_risk": modic_prognostic_risk,
            "modic_inflammatory_activity": modic_inflammatory_activity,
            "modic_structural_stability": modic_structural_stability,
            "modic_type1_ratio": ratio_type_1,
            "modic_total_regions": float(total_regions),
        }

    def _get_empty_result(self) -> Dict[str, float]:
        return {
            "modic_prognostic_risk": 0.0,
            "modic_inflammatory_activity": 0.0,
            "modic_structural_stability": 0.0,
            "modic_type1_ratio": 0.0,
            "modic_total_regions": 0.0,
            "avg_modic_mean_intensity": 0.0,
            "avg_modic_size": 0.0,
            "avg_modic_heterogeneity": 0.0,
        }


ModicClassifier = MODICCalculator