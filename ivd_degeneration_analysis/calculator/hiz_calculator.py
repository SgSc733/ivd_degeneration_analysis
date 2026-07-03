import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from .base_calculator import BaseCalculator
from utils.memory_monitor import monitor_memory


class HIZCalculator(BaseCalculator):

    def __init__(
        self,
        threshold_factor: float = 2.0,
        min_hiz_size: int = 5,
        max_hiz_size: int = 500,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        **kwargs
    ):
        super().__init__("HIZ Calculator", enable_parallel=enable_parallel, **kwargs)

        self.threshold_factor = float(threshold_factor)
        self.min_hiz_size = int(min_hiz_size)
        self.max_hiz_size = int(max_hiz_size)

        if max_workers is not None:
            self.max_workers = int(max_workers)

    @monitor_memory(threshold_percent=85)
    def calculate(
        self,
        image: np.ndarray,
        disc_mask: np.ndarray,
        dural_sac_mask: Optional[np.ndarray],
        posterior_edge_mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:

        self.validate_input(image, disc_mask)

        disc_bin = (disc_mask > 0).astype(np.uint8)

        candidates, csf_int_ref = self._detect_candidates(
            image=image,
            disc_bin=disc_bin,
            sac_mask=dural_sac_mask
        )

        if not candidates:
            return self._get_empty_result()

        best_candidate = self._select_best_candidate(candidates)
        hiz_mask = best_candidate["mask"]

        intensity_features = self._calculate_intensity_features(
            image=image,
            hiz_mask=hiz_mask,
            csf_ref=csf_int_ref
        )

        spatial_features = self._calculate_spatial_features(
            hiz_mask=hiz_mask,
            disc_bin=disc_bin,
            posterior_edge_mask=posterior_edge_mask
        )

        morph_features = self._calculate_morphological_features(hiz_mask)

        base_features: Dict[str, float] = {
            "num_candidates": float(len(candidates)),
            "hiz_size": float(best_candidate["size"]),
            **intensity_features,
            **spatial_features,
            **morph_features,
        }

        prognostic_features = self._calculate_prognostic_features(base_features)

        result = dict(base_features)
        result.update(prognostic_features)
        return result

    def _detect_candidates(
        self,
        image: np.ndarray,
        disc_bin: np.ndarray,
        sac_mask: Optional[np.ndarray]
    ) -> Tuple[List[Dict], float]:
        if sac_mask is not None and np.any(sac_mask):
            sac_bin = (sac_mask > 0)
            csf_pixels = image[sac_bin]
            i_csf = float(np.percentile(csf_pixels, 75))
        else:
            i_csf = float(np.max(image) * 0.5)

        t_hiz = i_csf * self.threshold_factor

        thresh_map = ((image > t_hiz) & (disc_bin > 0)).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            thresh_map, connectivity=8
        )

        candidates: List[Dict] = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if self.min_hiz_size <= area <= self.max_hiz_size:
                mask = (labels == i).astype(np.uint8)
                mean_val = float(np.mean(image[mask > 0])) if area > 0 else 0.0
                candidates.append({
                    "mask": mask,
                    "size": area,
                    "mean_intensity": mean_val
                })

        return candidates, i_csf

    def _select_best_candidate(self, candidates: List[Dict]) -> Dict:
        best_score = -1.0
        best_candidate: Optional[Dict] = None

        for cand in candidates:
            i_norm = float(cand["mean_intensity"]) / 255.0
            s_norm = min(float(cand["size"]) / 100.0, 1.0)
            score = 0.6 * i_norm + 0.4 * s_norm

            if score > best_score:
                best_score = score
                best_candidate = cand

        return best_candidate if best_candidate is not None else candidates[0]

    def _calculate_intensity_features(
        self,
        image: np.ndarray,
        hiz_mask: np.ndarray,
        csf_ref: float
    ) -> Dict[str, float]:
        pixels = image[hiz_mask > 0]
        if pixels.size == 0:
            return {
                "hiz_mean_intensity": 0.0,
                "hiz_max_intensity": 0.0,
                "hiz_intensity_ratio": 0.0,
                "hiz_intensity_std": 0.0
            }

        mean_val = float(np.mean(pixels))
        max_val = float(np.max(pixels))
        std_val = float(np.std(pixels))
        ratio = float(mean_val / csf_ref) if csf_ref > 0 else 0.0

        return {
            "hiz_mean_intensity": mean_val,
            "hiz_max_intensity": max_val,
            "hiz_intensity_ratio": ratio,
            "hiz_intensity_std": std_val
        }
    def _calculate_spatial_features(
        self,
        hiz_mask: np.ndarray,
        disc_bin: np.ndarray,
        posterior_edge_mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        epsilon = 1e-8

        hiz_bin = (hiz_mask > 0).astype(np.uint8)

        M_hiz = cv2.moments(hiz_bin)
        if M_hiz["m00"] == 0:
            return {
                "hiz_centroid_x": 0.0,
                "hiz_centroid_y": 0.0,
                "hiz_position_x": 0.0,
                "hiz_position_y": 0.0,
                "hiz_boundary_distance": 0.0,
                "hiz_posterior_distance": 100.0,
                "disc_radius": 1.0,
                "hiz_dist_to_center_abs": 0.0
            }

        cx_hiz = float(M_hiz["m10"] / (M_hiz["m00"] + epsilon))
        cy_hiz = float(M_hiz["m01"] / (M_hiz["m00"] + epsilon))

        M_disc = cv2.moments(disc_bin)
        if M_disc["m00"] == 0:
            cx_disc, cy_disc = 0.0, 0.0
        else:
            cx_disc = float(M_disc["m10"] / (M_disc["m00"] + epsilon))
            cy_disc = float(M_disc["m01"] / (M_disc["m00"] + epsilon))

        pos_x = float(cx_hiz - cx_disc)
        pos_y = float(cy_hiz - cy_disc)

        disc_area = float(M_disc["m00"])
        r_disc = float(np.sqrt(disc_area / np.pi)) if disc_area > 0 else 1.0

        boundary_distance = 0.0
        if np.any(disc_bin) and np.any(hiz_bin):
            dt_disc = cv2.distanceTransform(disc_bin.astype(np.uint8), cv2.DIST_L2, 5)
            vals = dt_disc[hiz_bin > 0]
            if vals.size > 0:
                boundary_distance = float(np.min(vals))

        if posterior_edge_mask is not None and np.any(posterior_edge_mask) and np.any(hiz_bin):
            post_bin = (posterior_edge_mask > 0).astype(np.uint8)
            inv = (post_bin == 0).astype(np.uint8)
            dt_post = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
            vals = dt_post[hiz_bin > 0]
            posterior_distance = float(np.min(vals)) if vals.size > 0 else 0.0
        else:
            posterior_distance = 100.0

        dist_center = float(np.sqrt(pos_x ** 2 + pos_y ** 2))

        return {
            "hiz_centroid_x": cx_hiz,
            "hiz_centroid_y": cy_hiz,
            "hiz_position_x": pos_x,
            "hiz_position_y": pos_y,
            "hiz_boundary_distance": boundary_distance,
            "hiz_posterior_distance": posterior_distance,
            "disc_radius": r_disc,
            "hiz_dist_to_center_abs": dist_center
        }

    def _calculate_morphological_features(self, hiz_mask: np.ndarray) -> Dict[str, float]:
        epsilon = 1e-8

        hiz_u8 = (hiz_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(hiz_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "hiz_circularity": 0.0,
                "hiz_elongation": 1.0,
                "hiz_compactness": 0.0
            }

        cnt = max(contours, key=cv2.contourArea)

        area = float(cv2.contourArea(cnt))
        perimeter = float(cv2.arcLength(cnt, True))

        if area > 0 and perimeter > 0:
            circularity = float((4.0 * np.pi * area) / ((perimeter ** 2) + epsilon))
            compactness = float(((perimeter ** 2) / ((4.0 * np.pi * area) + epsilon)))
        else:
            circularity = 0.0
            compactness = 0.0

        if len(cnt) >= 5:
            try:
                (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
                major = float(max(MA, ma))
                minor = float(min(MA, ma))
                elongation = float(major / (minor + epsilon))
            except Exception:
                elongation = 1.0
        else:
            elongation = 1.0

        return {
            "hiz_circularity": float(circularity),
            "hiz_elongation": float(elongation),
            "hiz_compactness": float(compactness)
        }

    def _calculate_prognostic_features(self, feats: Dict[str, float]) -> Dict[str, float]:
        epsilon = 1e-8

        f1 = min(float(feats.get("hiz_mean_intensity", 0.0)) / 255.0, 1.0)
        f2 = min(float(feats.get("hiz_size", 0.0)) / float(self.max_hiz_size), 1.0)

        d_post = float(feats.get("hiz_posterior_distance", 50.0))
        f3 = 1.0 - min(d_post / 50.0, 1.0)

        circularity = float(feats.get("hiz_circularity", 1.0))
        f4 = 1.0 - circularity
        f4 = max(0.0, f4)

        r_prognostic = 0.3 * f1 + 0.3 * f2 + 0.2 * f3 + 0.2 * f4

        dist_center = float(feats.get("hiz_dist_to_center_abs", 0.0))
        r_disc = float(feats.get("disc_radius", 1.0))
        i_mechanical = dist_center / (r_disc + epsilon)

        ratio = float(feats.get("hiz_intensity_ratio", 0.0))
        mean_i = float(feats.get("hiz_mean_intensity", 0.0))
        std_i = float(feats.get("hiz_intensity_std", 0.0))
        sigma_rel = float(std_i / (mean_i + epsilon))

        a_inflammatory = (
            0.6 * min(ratio / 3.0, 1.0) +
            0.4 * min(2.0 * sigma_rel, 1.0)
        )

        size_val = float(feats.get("hiz_size", 0.0))
        n_regions = float(feats.get("num_candidates", 1.0))
        elongation = float(feats.get("hiz_elongation", 1.0))

        s_tear = (
            0.4 * min(size_val / 100.0, 1.0) +
            0.3 * min(n_regions / 5.0, 1.0) +
            0.3 * min(elongation / 3.0, 1.0)
        )

        return {
            "prognostic_risk_score": float(r_prognostic),
            "mechanical_instability_index": float(i_mechanical),
            "inflammatory_activity_index": float(a_inflammatory),
            "annular_tear_severity": float(s_tear)
        }

    def _get_empty_result(self) -> Dict[str, float]:
        keys = [
            "hiz_mean_intensity", "hiz_max_intensity", "hiz_intensity_ratio", "hiz_intensity_std",
            "hiz_centroid_x", "hiz_centroid_y",
            "hiz_position_x", "hiz_position_y",
            "hiz_boundary_distance", "hiz_posterior_distance",
            "disc_radius", "hiz_dist_to_center_abs",
            "hiz_circularity", "hiz_elongation", "hiz_compactness",
            "prognostic_risk_score", "mechanical_instability_index",
            "inflammatory_activity_index", "annular_tear_severity",
            "num_candidates", "hiz_size",
        ]
        return {k: 0.0 for k in keys}

    def process_multi_slice(
        self,
        images: List[np.ndarray],
        disc_masks: List[np.ndarray],
        dural_sac_masks: List[np.ndarray],
        posterior_edge_masks: Optional[List[Optional[np.ndarray]]] = None,
        use_parallel: Optional[bool] = None
    ) -> Dict[str, float]:
        if use_parallel is None:
            use_parallel = self.enable_parallel

        if posterior_edge_masks is None:
            posterior_edge_masks = [None] * len(images)

        if use_parallel and len(images) >= 5:
            return self.process_multi_slice_parallel(images, disc_masks, dural_sac_masks, posterior_edge_masks)

        all_results: List[Dict[str, float]] = []
        for img, disc, sac, post in zip(images, disc_masks, dural_sac_masks, posterior_edge_masks):
            res = self.calculate(img, disc, sac, post)
            if res.get("hiz_size", 0.0) > 0:
                all_results.append(res)

        if not all_results:
            return self._get_empty_result()

        aggregated: Dict[str, float] = {}
        for k in all_results[0].keys():
            aggregated[k] = float(np.mean([r[k] for r in all_results]))

        aggregated["valid_hiz_slices"] = float(len(all_results))
        return aggregated

    @monitor_memory(threshold_percent=80)
    def process_multi_slice_parallel(
        self,
        images: List[np.ndarray],
        disc_masks: List[np.ndarray],
        dural_sac_masks: List[np.ndarray],
        posterior_edge_masks: List[Optional[np.ndarray]]
    ) -> Dict[str, float]:
        if not self.enable_parallel or len(images) < 2:
            return self.process_multi_slice(images, disc_masks, dural_sac_masks, posterior_edge_masks, use_parallel=False)

        def _task(args):
            img, disc, sac, post = args
            return self.calculate(img, disc, sac, post)

        args_list = list(zip(images, disc_masks, dural_sac_masks, posterior_edge_masks))

        max_workers = min(int(self.max_workers), len(args_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_task, args_list))

        valid_results = [r for r in results if r.get("hiz_size", 0.0) > 0]
        if not valid_results:
            return self._get_empty_result()

        aggregated: Dict[str, float] = {}
        for k in valid_results[0].keys():
            aggregated[k] = float(np.mean([r[k] for r in valid_results]))

        aggregated["valid_hiz_slices"] = float(len(valid_results))
        return aggregated