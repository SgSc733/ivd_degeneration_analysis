import numpy as np
import cv2
from skimage import filters
from typing import Dict, List, Tuple, Optional
from .base_calculator import BaseCalculator
from concurrent.futures import ThreadPoolExecutor
import psutil
from utils.memory_monitor import monitor_memory


class GaborCalculator(BaseCalculator):

    def __init__(self, wavelengths: List[float] = None,
                 orientations: List[float] = None,
                 frequency: float = 0.1,
                 sigma: Optional[float] = None,
                 gamma: float = 0.5,
                 psi: float = 0,
                 enable_parallel: bool = True,
                 max_workers: Optional[int] = None, **kwargs):

        super().__init__("Gabor Calculator", enable_parallel=enable_parallel, **kwargs)

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

        roi, roi_mask = self._extract_roi(image, mask)
        roi_normalized = self._normalize_image(roi)

        features: Dict[str, float] = {}
        feature_index = 0

        prognostic_data: List[Dict] = []

        for wavelength in self.wavelengths:
            for orientation in self.orientations:
                real, imag = self._apply_gabor_filter(roi_normalized, wavelength, orientation)
                magnitude = np.sqrt(real ** 2 + imag ** 2)

                stats = self._extract_statistics(magnitude, roi_mask)

                orientation_deg = np.degrees(orientation)
                prefix = f"gabor_w{wavelength}_o{int(orientation_deg)}"

                features[f"{prefix}_mean"] = stats["mean"]
                features[f"{prefix}_std"] = stats["std"]
                features[f"{prefix}_energy"] = stats["energy"]
                features[f"{prefix}_entropy"] = stats["entropy"]
                features[f"{prefix}_skewness"] = stats["skewness"]

                valid_pixels = magnitude[roi_mask > 0]

                canny_density = 0.0
                if valid_pixels.size > 0:
                    mag_uint8 = cv2.normalize(
                        magnitude, None, 0, 255, cv2.NORM_MINMAX
                    ).astype(np.uint8)
                    edges = cv2.Canny(mag_uint8, 30, 100)
                    edge_pixels = np.sum((edges > 0) & (roi_mask > 0))
                    total_pixels = np.sum(roi_mask > 0)
                    canny_density = edge_pixels / (total_pixels + 1e-8)

                prognostic_data.append({
                    "wavelength": float(wavelength),
                    "orientation": float(orientation),
                    "vector": valid_pixels,
                    "mean": float(stats["mean"]),
                    "std": float(stats["std"]),
                    "canny_density": float(canny_density),
                })

                feature_index += 5

        prognostic_features = self._compute_prognostic_features(prognostic_data)
        features.update(prognostic_features)

        self.logger.info(f"提取了{feature_index}个Gabor特征")

        return features

    @monitor_memory(threshold_percent=70)
    def calculate_parallel(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        if not self.enable_parallel:
            return self.calculate(image, mask)

        self.validate_input(image, mask)

        roi, roi_mask = self._extract_roi(image, mask)
        roi_normalized = self._normalize_image(roi)

        features: Dict[str, float] = {}

        def compute_single_gabor(params):
            wavelength, orientation = params

            real, imag = self._apply_gabor_filter(roi_normalized, wavelength, orientation)
            magnitude = np.sqrt(real ** 2 + imag ** 2)

            stats = self._extract_statistics(magnitude, roi_mask)

            orientation_deg = np.degrees(orientation)
            prefix = f"gabor_w{wavelength}_o{int(orientation_deg)}"

            valid_pixels = magnitude[roi_mask > 0]

            canny_density = 0.0
            if valid_pixels.size > 0:
                mag_uint8 = cv2.normalize(
                    magnitude, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                edges = cv2.Canny(mag_uint8, 30, 100)
                edge_pixels = np.sum((edges > 0) & (roi_mask > 0))
                total_pixels = np.sum(roi_mask > 0)
                canny_density = edge_pixels / (total_pixels + 1e-8)

            result_dict = {
                f"{prefix}_mean": stats["mean"],
                f"{prefix}_std": stats["std"],
                f"{prefix}_energy": stats["energy"],
                f"{prefix}_entropy": stats["entropy"],
                f"{prefix}_skewness": stats["skewness"],
            }

            prog_data = {
                "wavelength": float(wavelength),
                "orientation": float(orientation),
                "vector": valid_pixels,
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "canny_density": float(canny_density),
            }

            return result_dict, prog_data

        params_list = [(w, o) for w in self.wavelengths for o in self.orientations]

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(params_list))) as executor:
            results = list(executor.map(compute_single_gabor, params_list))

        prognostic_data_list: List[Dict] = []
        for feat_dict, prog_data in results:
            features.update(feat_dict)
            prognostic_data_list.append(prog_data)

        prognostic_features = self._compute_prognostic_features(prognostic_data_list)
        features.update(prognostic_features)

        self.logger.info(f"并行提取了{len(features)}个Gabor特征")

        return features

    def _compute_prognostic_features(self, data_list: List[Dict]) -> Dict[str, float]:

        if not data_list:
            return {
                "consistency": 0.0,
                "directional_degeneration": 0.0,
                "texture_integrity": 0.0,
                "inflammatory_pattern": 0.0,
            }

        epsilon = 1e-8

        try:
            vectors = [
                d["vector"] for d in data_list
                if getattr(d.get("vector", None), "size", 0) > 0
            ]
            if len(vectors) > 1:
                min_len = min(v.size for v in vectors)
                matrix = np.stack([v[:min_len] for v in vectors], axis=0)

                corr_matrix = np.abs(np.corrcoef(matrix))
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

                upper = np.triu_indices_from(corr_matrix, k=1)
                consistency = float(np.mean(corr_matrix[upper])) if upper[0].size > 0 else 0.0
            else:
                consistency = 1.0
        except Exception:
            consistency = 0.0

        try:
            dir_means: Dict[float, List[float]] = {}
            for d in data_list:
                ori = float(d["orientation"])
                dir_means.setdefault(ori, []).append(float(d["mean"]))

            theta_responses = [float(np.mean(vals)) for vals in dir_means.values() if len(vals) > 0]
            if theta_responses:
                mu_theta = float(np.mean(theta_responses))
                sigma_theta = float(np.std(theta_responses))
                directional_degeneration = mu_theta / (sigma_theta + epsilon)
            else:
                directional_degeneration = 0.0
        except Exception:
            directional_degeneration = 0.0

        try:
            short_wave_densities = [
                float(d["canny_density"]) for d in data_list
                if float(d["wavelength"]) <= 4
            ]
            texture_integrity = float(np.mean(short_wave_densities)) if short_wave_densities else 0.0
        except Exception:
            texture_integrity = 0.0

        try:
            long_wave = [d for d in data_list if float(d["wavelength"]) >= 6]
            cv_values: List[float] = []
            for d in long_wave:
                m = float(d["mean"])
                s = float(d["std"])
                cv_values.append(s / m if m > epsilon else 0.0)

            inflammatory_pattern = float(np.mean(cv_values)) if cv_values else 0.0
        except Exception:
            inflammatory_pattern = 0.0

        return {
            "consistency": float(consistency),
            "directional_degeneration": float(directional_degeneration),
            "texture_integrity": float(texture_integrity),
            "inflammatory_pattern": float(inflammatory_pattern),
        }

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
            sigma_y=sigma / self.gamma,
            mode="reflect",
        )
        return real, imag

    def _extract_roi(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        coords = np.column_stack(np.where(mask > 0))
        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0)

        roi = image[min_row:max_row + 1, min_col:max_col + 1].copy()
        roi_mask = mask[min_row:max_row + 1, min_col:max_col + 1]

        roi[roi_mask == 0] = 0
        return roi, roi_mask

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        non_zero = image[image > 0]
        if len(non_zero) == 0:
            return image

        p1, p99 = np.percentile(non_zero, [1, 99])

        if p99 <= p1:
            min_val = non_zero.min()
            max_val = non_zero.max()
            if max_val > min_val:
                normalized = (image - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(image, dtype=np.float64)
        else:
            normalized = np.clip((image - p1) / (p99 - p1), 0, 1)

        normalized[image == 0] = 0
        return normalized

    def _extract_statistics(self, response: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        valid_pixels = response[mask > 0]

        if len(valid_pixels) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "energy": 0.0,
                "entropy": 0.0,
                "skewness": 0.0,
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
            "mean": float(mean),
            "std": float(std),
            "energy": float(energy),
            "entropy": float(entropy),
            "skewness": float(skewness),
        }

    def process_multi_slice(self, image_slices: List[np.ndarray],
                            masks: List[np.ndarray],
                            use_parallel: Optional[bool] = None) -> Dict[str, float]:

        if use_parallel is None:
            use_parallel = self.enable_parallel

        if use_parallel and len(image_slices) >= 3:
            return self.process_multi_slice_parallel(image_slices, masks)

        gabor_features: Dict[str, List[float]] = {}
        for i, (img, mask) in enumerate(zip(image_slices, masks)):
            try:
                slice_features = self.calculate(img, mask)
            except Exception as exc:
                self.logger.warning(f"切片{i}处理失败: {exc}")
                continue
            for k, v in slice_features.items():
                gabor_features.setdefault(k, []).append(v)

        if not gabor_features:
            return {}

        return {k: float(np.mean(v)) for k, v in gabor_features.items()}

    def process_multi_slice_parallel(self, image_slices: List[np.ndarray],
                                     masks: List[np.ndarray]) -> Dict[str, float]:

        def process_single_slice(args):
            i, img, mask = args
            try:
                result = self.calculate_parallel(img, mask)
                return (i, result, None)
            except Exception as e:
                return (i, None, str(e))

        args_list = [(i, img, mask) for i, (img, mask) in enumerate(zip(image_slices, masks))]

        with ThreadPoolExecutor(max_workers=min(2, len(args_list))) as executor:
            results = list(executor.map(process_single_slice, args_list))

        all_features: Dict[str, List[float]] = {}
        valid_slices = 0

        for i, result, error in sorted(results, key=lambda x: x[0]):
            if error:
                self.logger.warning(f"切片{i}处理失败: {error}")
            elif result:
                valid_slices += 1
                for k, v in result.items():
                    all_features.setdefault(k, []).append(v)

        if valid_slices == 0:
            raise ValueError("没有成功处理的切片")

        gabor_result = {k: float(np.mean(v)) for k, v in all_features.items()}
        gabor_result["valid_slices"] = valid_slices
        return gabor_result

    def calculate_parallel_with_memory_management(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        memory_info = psutil.virtual_memory()
        available_gb = memory_info.available / (1024 ** 3)

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
