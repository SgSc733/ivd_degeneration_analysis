import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from scipy import stats
from .base_calculator import BaseCalculator
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import psutil
from utils.memory_monitor import monitor_memory

try:
    from skimage.feature import local_binary_pattern
except Exception:
    local_binary_pattern = None


class FractalDimensionCalculator(BaseCalculator):

    def __init__(self, threshold_percent: float = 0.65,
                 min_box_size: int = 1,
                 max_box_size: Optional[int] = None,
                 enable_parallel: bool = False,
                 max_workers: Optional[int] = None, **kwargs):

        super().__init__("FD Calculator", enable_parallel=enable_parallel, **kwargs)
        self.threshold_percent = threshold_percent
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size
        if max_workers is not None:
            self.max_workers = max_workers

    def preprocess_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:

        roi = image.copy()
        roi[mask == 0] = 0

        roi_normalized = self._normalize_grayscale(roi, mask)

        try:
            roi_uint8 = roi_normalized.astype(np.uint8)
            roi_masked = cv2.bitwise_and(roi_uint8, roi_uint8, mask=mask.astype(np.uint8))
            threshold_value = cv2.threshold(
                roi_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[0]

            if threshold_value < 50 or threshold_value > 200:
                valid_pixels = roi_normalized[mask > 0]
                threshold_value = np.percentile(valid_pixels, self.threshold_percent * 100)

        except Exception:
            threshold_value = 255 * self.threshold_percent

        _, binary = cv2.threshold(roi_normalized, threshold_value, 255, cv2.THRESH_BINARY)

        edges = cv2.Canny(binary.astype(np.uint8), 30, 100)

        if np.sum(edges > 0) < 50:
            kernel = np.ones((3, 3), np.uint8)
            gradient = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
            edges = np.maximum(edges, gradient)

        edges[mask == 0] = 0
        return edges

    def _normalize_grayscale(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:

        valid_pixels = image[mask > 0]

        if len(valid_pixels) == 0:
            return np.zeros_like(image, dtype=np.uint8)

        min_val = np.percentile(valid_pixels, 1)
        max_val = np.percentile(valid_pixels, 99)

        normalized = (image - min_val) / (max_val - min_val) * 255
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        normalized[mask == 0] = 0
        return normalized

    @monitor_memory(threshold_percent=65)
    def box_counting(self, binary_image: np.ndarray) -> Tuple[List[int], List[int]]:

        h, w = binary_image.shape
        max_dim = max(h, w)

        if self.max_box_size is None:
            max_box = min(max_dim // 4, 128)
        else:
            max_box = self.max_box_size

        box_sizes = []
        size = self.min_box_size
        while size <= max_box:
            box_sizes.append(size)
            size *= 2

        counts = []
        for box_size in box_sizes:
            count = self._count_boxes(binary_image, box_size)
            counts.append(count)
            self.logger.debug(f"Box size: {box_size}, Count: {count}")

        return box_sizes, counts

    def _count_boxes(self, image: np.ndarray, box_size: int) -> int:

        h, w = image.shape
        count = 0

        for i in range(0, h, box_size):
            for j in range(0, w, box_size):
                box = image[i:min(i + box_size, h), j:min(j + box_size, w)]
                if np.any(box > 0):
                    count += 1

        return count

    def calculate_fd(self, box_sizes: List[int], counts: List[int]) -> Dict[str, float]:

        log_sizes = np.log(box_sizes)
        log_counts = np.log(counts)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)

        fd = -slope

        return {
            'fd': fd,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err,
            'slope': slope,
            'intercept': intercept
        }

    @monitor_memory(threshold_percent=70)
    def calculate(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        try:
            edges = self.preprocess_image(image, mask)

            if np.sum(edges > 0) < 10:
                raise ValueError("边缘像素太少，无法计算分形维度")

            box_sizes, counts = self.box_counting(edges)

            if len(box_sizes) < 3:
                raise ValueError("数据点太少，无法进行可靠的线性回归")

            result = self.calculate_fd(box_sizes, counts)

            result['num_edge_pixels'] = np.sum(edges > 0)
            result['num_data_points'] = len(box_sizes)

            epsilon = 1e-8

            sdi = result['fd']

            total_roi_pixels = np.sum(mask > 0)
            eci = result['num_edge_pixels'] / (total_roi_pixels + epsilon)

            ths = 0.0
            try:
                if local_binary_pattern is not None:
                    norm_img = self._normalize_grayscale(image, mask)
                    lbp = local_binary_pattern(norm_img, 8, 1, method='uniform')
                    valid_lbp = lbp[mask > 0]
                    if len(valid_lbp) > 0:
                        lbp_mean = np.mean(valid_lbp)
                        lbp_std = np.std(valid_lbp)
                        ths = lbp_std / (lbp_mean + epsilon)
            except Exception:
                ths = 0.0

            msc_ratios = []
            scales = [1, 2, 4]
            h, w = image.shape

            for s in scales:
                if s == 1:
                    curr_edges_count = result['num_edge_pixels']
                    curr_mask_count = total_roi_pixels
                else:
                    new_w, new_h = w // s, h // s
                    if new_w < 5 or new_h < 5:
                        msc_ratios.append(0.0)
                        continue

                    small_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    small_mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                    try:
                        small_edges = self.preprocess_image(small_img, small_mask)
                        curr_edges_count = np.sum(small_edges > 0)
                        curr_mask_count = np.sum(small_mask > 0)
                    except Exception:
                        curr_edges_count = 0
                        curr_mask_count = 1

                ratio = curr_edges_count / (curr_mask_count + epsilon)
                msc_ratios.append(ratio)

            msc = np.mean(msc_ratios) if msc_ratios else 0.0

            aii = 0.0
            try:
                contours, _ = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        aii = (4 * np.pi * area) / (perimeter * perimeter)
            except Exception:
                aii = 0.0

            result['sdi'] = float(sdi)
            result['eci'] = float(eci)
            result['ths'] = float(ths)
            result['msc'] = float(msc)
            result['aii'] = float(aii)

            return result

        except Exception as e:
            self.logger.error(f"分形维度计算失败: {str(e)}")
            raise RuntimeError(f"分形维度计算失败: {str(e)}")

    @monitor_memory(threshold_percent=60)
    def visualize_box_counting(self, image: np.ndarray, mask: np.ndarray,
                              save_path: Optional[str] = None) -> Dict:

        edges = self.preprocess_image(image, mask)
        box_sizes, counts = self.box_counting(edges)
        result = self.calculate_fd(box_sizes, counts)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title('边缘检测结果')
        axes[0, 1].axis('off')

        example_boxes = self._visualize_boxes(edges, box_sizes[0])
        axes[1, 0].imshow(example_boxes)
        axes[1, 0].set_title(f'盒计数示例 (尺寸={box_sizes[0]})')
        axes[1, 0].axis('off')

        log_sizes = np.log(box_sizes)
        log_counts = np.log(counts)

        axes[1, 1].scatter(log_sizes, log_counts, color='blue', s=50)
        axes[1, 1].plot(log_sizes,
                        result['slope'] * log_sizes + result['intercept'],
                        'r-', label=f'FD = {result["fd"]:.3f}')
        axes[1, 1].set_xlabel('log(Box Size)')
        axes[1, 1].set_ylabel('log(Count)')
        axes[1, 1].set_title(f'Box-Counting: FD = {result["fd"]:.3f}, R² = {result["r_squared"]:.3f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close()
        return result

    def _visualize_boxes(self, image: np.ndarray, box_size: int) -> np.ndarray:

        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        h, w = image.shape

        for i in range(0, h, box_size):
            cv2.line(vis_img, (0, i), (w, i), (0, 255, 0), 1)
        for j in range(0, w, box_size):
            cv2.line(vis_img, (j, 0), (j, h), (0, 255, 0), 1)

        for i in range(0, h, box_size):
            for j in range(0, w, box_size):
                box = image[i:min(i + box_size, h), j:min(j + box_size, w)]
                if np.any(box > 0):
                    cv2.rectangle(vis_img, (j, i),
                                  (min(j + box_size, w), min(i + box_size, h)),
                                  (255, 0, 0), 2)

        return vis_img

    def process_multi_slice(self, image_slices: List[np.ndarray],
                            disc_masks: List[np.ndarray]) -> Dict[str, float]:

        fd_results = []
        valid_slices = 0

        for i, (img, mask) in enumerate(zip(image_slices, disc_masks)):
            try:
                result = self.calculate(img, mask)
                fd_results.append(result)
                valid_slices += 1
            except Exception as e:
                self.logger.warning(f"切片{i}处理失败: {str(e)}")
                continue

        if valid_slices == 0:
            raise ValueError("没有成功处理的切片")

        avg_result = {
            'fd': np.mean([r['fd'] for r in fd_results]),
            'fd_std': np.std([r['fd'] for r in fd_results]),
            'r_squared': np.mean([r['r_squared'] for r in fd_results]),

            'sdi': np.mean([r.get('sdi', r['fd']) for r in fd_results]),
            'eci': np.mean([r.get('eci', 0.0) for r in fd_results]),
            'ths': np.mean([r.get('ths', 0.0) for r in fd_results]),
            'msc': np.mean([r.get('msc', 0.0) for r in fd_results]),
            'aii': np.mean([r.get('aii', 0.0) for r in fd_results]),

            'valid_slices': valid_slices
        }

        return avg_result

    @monitor_memory(threshold_percent=70)
    def process_multi_slice_parallel(self, image_slices: List[np.ndarray],
                                     disc_masks: List[np.ndarray]) -> Dict[str, float]:

        if not self.enable_parallel or len(image_slices) < 3:
            return self.process_multi_slice(image_slices, disc_masks)

        if not self._check_memory_availability(len(image_slices)):
            self.logger.warning("内存不足，使用串行处理")
            return self.process_multi_slice(image_slices, disc_masks)

        def process_single_slice(args):
            i, img, mask = args
            try:
                result = self.calculate(img, mask)
                return (i, result, None)
            except Exception as e:
                return (i, None, str(e))

        args_list = [(i, img, mask)
                     for i, (img, mask)
                     in enumerate(zip(image_slices, disc_masks))]

        max_workers = min(2, self.max_workers, len(args_list))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_slice, args_list))

        fd_results = []
        valid_slices = 0

        for i, result, error in sorted(results, key=lambda x: x[0]):
            if error:
                self.logger.warning(f"切片{i}处理失败: {error}")
            elif result:
                fd_results.append(result)
                valid_slices += 1

        if valid_slices == 0:
            raise ValueError("没有成功处理的切片")

        avg_result = {
            'fd': np.mean([r['fd'] for r in fd_results]),
            'fd_std': np.std([r['fd'] for r in fd_results]),
            'r_squared': np.mean([r['r_squared'] for r in fd_results]),

            'sdi': np.mean([r.get('sdi', r['fd']) for r in fd_results]),
            'eci': np.mean([r.get('eci', 0.0) for r in fd_results]),
            'ths': np.mean([r.get('ths', 0.0) for r in fd_results]),
            'msc': np.mean([r.get('msc', 0.0) for r in fd_results]),
            'aii': np.mean([r.get('aii', 0.0) for r in fd_results]),

            'valid_slices': valid_slices
        }

        return avg_result

    def _check_memory_availability(self, num_slices: int) -> bool:

        try:
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / (1024 ** 3)

            estimated_memory_per_slice = 0.1
            total_estimated = num_slices * estimated_memory_per_slice

            safe_available = available_gb - 1.0

            if total_estimated > safe_available:
                self.logger.warning(f"内存不足：需要约{total_estimated:.1f}GB，"
                                    f"可用{safe_available:.1f}GB")
                return False

            return True

        except Exception as e:
            self.logger.warning(f"无法检查内存：{str(e)}")
            return True

    def calculate_with_memory_check(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:

        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent

        if memory_percent > 80:
            self.logger.warning(f"内存使用率高：{memory_percent}%")
            import gc
            gc.collect()

        try:
            return self.calculate(image, mask)
        except MemoryError:
            self.logger.error("内存不足，尝试降低计算精度")
            scale_factor = 0.5
            scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
            scaled_mask = cv2.resize(mask.astype(np.uint8), None,
                                     fx=scale_factor, fy=scale_factor,
                                     interpolation=cv2.INTER_NEAREST)
            return self.calculate(scaled_image, scaled_mask)
