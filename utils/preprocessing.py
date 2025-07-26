import numpy as np
import cv2
from typing import Tuple, List, Optional, Union, Dict
import SimpleITK as sitk
from scipy import ndimage
from skimage import filters, morphology
import pywt
import logging


class Preprocessor:
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    
    def preprocess_for_texture(self, image: np.ndarray, mask: np.ndarray,
                            original_spacing: List[float],
                            target_spacing: List[float] = None,
                            target_size: List[int] = None,
                            bin_width: float = 16) -> Tuple[np.ndarray, np.ndarray]:

        self.logger.info("纹理特征预处理开始")

        if target_size is None:
            target_size = [512, 512]  

        resampled_image, _ = self.resample_image(
            image, original_spacing, target_size=target_size, 
            interpolation='linear', is_label=False
        )
        resampled_mask, _ = self.resample_image(
            mask, original_spacing, target_size=target_size,
            interpolation='nearest', is_label=True
        )

        normalized_image = self.normalize_intensity_zscore(
            resampled_image, resampled_mask
        )

        discretized_image = self.discretize_intensity(
            normalized_image, bin_width=bin_width
        )
        
        return discretized_image, resampled_mask
    
    def preprocess_for_fractal(self, image: np.ndarray, mask: np.ndarray,
                            original_spacing: List[float],
                            target_spacing: List[float] = None,
                            target_size: List[int] = None,
                            window_center: float = 128,
                            window_width: float = 255,
                            threshold_percentile: float = 65) -> Tuple[np.ndarray, np.ndarray]:

        self.logger.info("分形维度预处理开始")

        if target_size is None:
            target_size = [512, 512]  

        resampled_image, _ = self.resample_image(
            image, original_spacing, target_size=target_size,
            interpolation='linear', is_label=False
        )
        resampled_mask, _ = self.resample_image(
            mask, original_spacing, target_size=target_size,
            interpolation='nearest', is_label=True
        )
        
        image_8bit = self.convert_to_8bit(resampled_image)

        windowed_image = self.apply_windowing(
            image_8bit, window_center, window_width
        )

        binary_image = self.binarize(
            windowed_image, method='percentile', 
            percentile=threshold_percentile
        )

        binary_image = binary_image * (resampled_mask > 0)

        edges = self.detect_edges(binary_image, method='canny')
        
        return edges, resampled_mask
    
    def preprocess_for_signal_intensity(self, image: np.ndarray, mask: np.ndarray,
                                    original_spacing: List[float],
                                    target_spacing: List[float] = None,
                                    target_size: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:

        self.logger.info("信号强度预处理开始（仅重采样）")

        if target_size is None:
            target_size = [512, 512] 

        resampled_image, _ = self.resample_image(
            image, original_spacing, target_size=target_size,
            interpolation='linear', is_label=False
        )
        resampled_mask, _ = self.resample_image(
            mask, original_spacing, target_size=target_size,
            interpolation='nearest', is_label=True
        )
        
        return resampled_image, resampled_mask
    
    def preprocess_for_shape(self, mask: np.ndarray,
                        original_spacing: List[float],
                        target_spacing: List[float] = None,
                        target_size: List[int] = None) -> np.ndarray:

        self.logger.info("形状特征预处理开始")

        if target_size is None:
            target_size = [512, 512]  

        resampled_mask, _ = self.resample_image(
            mask, original_spacing, target_size=target_size,
            interpolation='nearest', is_label=True
        )

        binary_mask = self.binarize(
            resampled_mask, threshold=0.5, method='fixed'
        )
        
        return binary_mask
    
    def preprocess_with_filter(self, image: np.ndarray, mask: np.ndarray,
                            original_spacing: List[float],
                            target_spacing: List[float] = None,
                            target_size: List[int] = None,
                            filter_type: str = 'log',
                            filter_params: Dict = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

        if target_size is None:
            target_size = [512, 512]  

        resampled_image, _ = self.resample_image(
            image, original_spacing, target_size=target_size,
            interpolation='linear', is_label=False
        )
        resampled_mask, _ = self.resample_image(
            mask, original_spacing, target_size=target_size,
            interpolation='nearest', is_label=True
        )

        if filter_type == 'log':
            sigma_list = filter_params.get('sigma_list', [1, 3, 5]) if filter_params else [1, 3, 5]
            filtered_images = self.apply_log_filter(resampled_image, sigma_list)
        elif filter_type == 'wavelet':
            wavelet = filter_params.get('wavelet', 'db1') if filter_params else 'db1'
            level = filter_params.get('level', 1) if filter_params else 1
            filtered_images = {}
            if resampled_image.ndim == 2:
                filtered_images = self.apply_wavelet_transform(resampled_image, wavelet, level)
            else:
                for i in range(resampled_image.shape[2]):
                    slice_filtered = self.apply_wavelet_transform(
                        resampled_image[:, :, i], wavelet, level
                    )
                    for key, value in slice_filtered.items():
                        if key not in filtered_images:
                            filtered_images[key] = []
                        filtered_images[key].append(value)
        else:
            filtered_images = {'original': resampled_image}

        processed_images = {}
        for key, img in filtered_images.items():
            normalized = self.normalize_intensity_zscore(img, resampled_mask)
            discretized = self.discretize_intensity(normalized, bin_width=16)
            processed_images[key] = discretized
        
        return processed_images, resampled_mask

    
    def convert_to_8bit(self, image: np.ndarray, 
                       percentile_range: Tuple[float, float] = (1, 99)) -> np.ndarray:

        p_low, p_high = np.percentile(image, percentile_range)
        image_clipped = np.clip(image, p_low, p_high)
        
        if p_high > p_low:
            image_8bit = ((image_clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
        else:
            image_8bit = np.zeros_like(image, dtype=np.uint8)
        
        return image_8bit
    
    def apply_windowing(self, image: np.ndarray, 
                       window_center: float = 128, 
                       window_width: float = 255) -> np.ndarray:

        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        windowed = np.clip(image, window_min, window_max)
        
        if window_max > window_min:
            windowed = ((windowed - window_min) / (window_max - window_min) * 255)
        else:
            windowed = np.zeros_like(image)
        
        return windowed
    
    def discretize_intensity(self, image: np.ndarray, 
                           bin_width: Optional[float] = None,
                           n_bins: Optional[int] = None) -> np.ndarray:

        if bin_width is None and n_bins is None:
            bin_width = 16
        
        min_val = np.min(image)
        max_val = np.max(image)
        
        if bin_width is not None:
            bins = np.arange(min_val, max_val + bin_width, bin_width)
            n_bins = len(bins) - 1
        else:
            bins = np.linspace(min_val, max_val, n_bins + 1)
        
        discretized = np.digitize(image, bins) - 1
        discretized = np.clip(discretized, 0, n_bins - 1)
        
        return discretized
    
    def normalize_intensity_zscore(self, image: np.ndarray, 
                                mask: Optional[np.ndarray] = None,
                                robust: bool = False,  
                                exclude_percentile: float = 0.0) -> np.ndarray:  
 
        if mask is not None and np.any(mask > 0):
            mask = mask.astype(bool)
            valid_pixels = image[mask]
        else:
            valid_pixels = image.flatten()
        
        if len(valid_pixels) == 0:
            self.logger.warning("没有有效像素，返回原始图像")
            return image.copy()

        if exclude_percentile > 0:
            p_low = np.percentile(valid_pixels, exclude_percentile)
            p_high = np.percentile(valid_pixels, 100 - exclude_percentile)
            valid_pixels = valid_pixels[(valid_pixels >= p_low) & (valid_pixels <= p_high)]
        
        if robust:
            median_val = np.median(valid_pixels)
            q25 = np.percentile(valid_pixels, 25)
            q75 = np.percentile(valid_pixels, 75)
            iqr = q75 - q25
            std_val = iqr / 1.349 if iqr > 0 else 1.0
            mean_val = median_val
            
            self.logger.info(f"鲁棒统计量: median={mean_val:.3f}, IQR_std={std_val:.3f}")
        else:
            mean_val = np.mean(valid_pixels)
            std_val = np.std(valid_pixels)
            
            self.logger.info(f"传统统计量: mean={mean_val:.3f}, std={std_val:.3f}")
        
        if std_val <= 0:
            std_val = 1.0
            self.logger.warning("标准差为0，设置为1.0")

        normalized = (image - mean_val) / std_val
        
        return normalized
    
    def resample_image(self, image: Union[np.ndarray, sitk.Image], 
                        original_spacing: List[float],
                        target_size: Optional[List[int]] = None,
                        target_spacing: Optional[List[float]] = None,
                        interpolation: str = 'linear',
                        is_label: bool = False) -> Tuple[np.ndarray, List[float]]:

        if isinstance(image, np.ndarray):
            sitk_image = sitk.GetImageFromArray(image)
            sitk_spacing = original_spacing[::-1]  
            sitk_image.SetSpacing(sitk_spacing)
        else:
            sitk_image = image
            sitk_spacing = list(sitk_image.GetSpacing())  

        original_size = sitk_image.GetSize()  

        if target_size is not None:
            if isinstance(target_size, int):
                target_size = [target_size, target_size]

            if len(target_size) == 2:  
                if len(original_size) == 3:  
                    new_size = [target_size[1], target_size[0], original_size[2]]
                    
                    physical_size_x = original_size[0] * sitk_spacing[0]
                    physical_size_y = original_size[1] * sitk_spacing[1]

                    scale_x = target_size[1] / original_size[0]
                    scale_y = target_size[0] / original_size[1]
                    scale = min(scale_x, scale_y)  

                    actual_new_x = int(original_size[0] * scale)
                    actual_new_y = int(original_size[1] * scale)

                    if actual_new_x < target_size[1] or actual_new_y < target_size[0]:
                        new_size = [target_size[1], target_size[0], original_size[2]]
                        target_spacing_xyz = [
                            physical_size_x / new_size[0],
                            physical_size_y / new_size[1],
                            sitk_spacing[2]
                        ]
                    else:
                        new_size = [actual_new_x, actual_new_y, original_size[2]]
                        target_spacing_xyz = [
                            sitk_spacing[0] / scale,
                            sitk_spacing[1] / scale,
                            sitk_spacing[2]
                        ]
                else: 
                    new_size = [target_size[1], target_size[0]] 
                    physical_size_x = original_size[0] * sitk_spacing[0]
                    physical_size_y = original_size[1] * sitk_spacing[1]
                    
                    scale_x = target_size[1] / original_size[0]
                    scale_y = target_size[0] / original_size[1]
                    scale = min(scale_x, scale_y)
                    
                    actual_new_x = int(original_size[0] * scale)
                    actual_new_y = int(original_size[1] * scale)
                    
                    if actual_new_x < target_size[1] or actual_new_y < target_size[0]:
                        new_size = [target_size[1], target_size[0]]
                        target_spacing_xyz = [
                            physical_size_x / new_size[0],
                            physical_size_y / new_size[1]
                        ]
                    else:
                        new_size = [actual_new_x, actual_new_y]
                        target_spacing_xyz = [
                            sitk_spacing[0] / scale,
                            sitk_spacing[1] / scale
                        ]
            else: 
                new_size = [target_size[2], target_size[1], target_size[0]] 
                target_spacing_xyz = [
                    original_size[i] * sitk_spacing[i] / new_size[i]
                    for i in range(len(new_size))
                ]
        else:
            if target_spacing is None:
                target_spacing = [1.0, 1.0, 1.0]

            target_spacing_xyz = target_spacing[::-1]  

            new_size = []
            for i in range(len(original_size)):
                new_pixels = int(round(original_size[i] * sitk_spacing[i] / target_spacing_xyz[i]))
                new_size.append(new_pixels)

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing_xyz)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        resampler.SetTransform(sitk.Transform())

        if is_label or interpolation == 'nearest':
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        elif interpolation == 'linear':
            resampler.SetInterpolator(sitk.sitkLinear)
        elif interpolation == 'cubic':
            resampler.SetInterpolator(sitk.sitkBSpline)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)

        resampled = resampler.Execute(sitk_image)
        resampled_array = sitk.GetArrayFromImage(resampled)

        if target_size is not None and len(target_size) == 2:
            current_shape = resampled_array.shape
            if len(current_shape) == 3:  
                if current_shape[1] != target_size[0] or current_shape[2] != target_size[1]:
                    new_array = np.zeros((current_shape[0], target_size[0], target_size[1]), dtype=resampled_array.dtype)

                    start_y = (target_size[0] - current_shape[1]) // 2
                    start_x = (target_size[1] - current_shape[2]) // 2

                    src_y_start = max(0, -start_y)
                    src_x_start = max(0, -start_x)
                    src_y_end = min(current_shape[1], current_shape[1] + (target_size[0] - current_shape[1] - start_y))
                    src_x_end = min(current_shape[2], current_shape[2] + (target_size[1] - current_shape[2] - start_x))
                    
                    dst_y_start = max(0, start_y)
                    dst_x_start = max(0, start_x)
                    dst_y_end = dst_y_start + (src_y_end - src_y_start)
                    dst_x_end = dst_x_start + (src_x_end - src_x_start)
                    
                    new_array[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                        resampled_array[:, src_y_start:src_y_end, src_x_start:src_x_end]
                    
                    resampled_array = new_array
            elif len(current_shape) == 2:  
                if current_shape[0] != target_size[0] or current_shape[1] != target_size[1]:
                    new_array = np.zeros(target_size, dtype=resampled_array.dtype)

                    start_y = (target_size[0] - current_shape[0]) // 2
                    start_x = (target_size[1] - current_shape[1]) // 2

                    src_y_start = max(0, -start_y)
                    src_x_start = max(0, -start_x)
                    src_y_end = min(current_shape[0], current_shape[0] + (target_size[0] - current_shape[0] - start_y))
                    src_x_end = min(current_shape[1], current_shape[1] + (target_size[1] - current_shape[1] - start_x))
                    
                    dst_y_start = max(0, start_y)
                    dst_x_start = max(0, start_x)
                    dst_y_end = dst_y_start + (src_y_end - src_y_start)
                    dst_x_end = dst_x_start + (src_x_end - src_x_start)
                    
                    new_array[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                        resampled_array[src_y_start:src_y_end, src_x_start:src_x_end]
                    
                    resampled_array = new_array

        if target_size is not None:
            actual_spacing_xyz = list(resampled.GetSpacing())
            actual_spacing = actual_spacing_xyz[::-1]  
        else:
            actual_spacing = target_spacing_xyz[::-1]  

        self.logger.info(f"重采样: {original_size} -> {new_size}, "
                        f"间距: {sitk_spacing} -> {target_spacing_xyz}")

        return resampled_array, actual_spacing
        
    
    def apply_log_filter(self, image: np.ndarray, 
                        sigma_list: List[float] = [1, 3, 5]) -> Dict[float, np.ndarray]:

        filtered_images = {}
        
        for sigma in sigma_list:
            gaussian = ndimage.gaussian_filter(image.astype(float), sigma=sigma)
            laplacian = ndimage.laplace(gaussian)
            log_filtered = -(sigma ** 2) * laplacian
            filtered_images[sigma] = log_filtered
            self.logger.info(f"应用LoG滤波器，sigma={sigma}")
        
        return filtered_images
    
    def apply_wavelet_transform(self, image: np.ndarray, 
                              wavelet: str = 'db1',
                              level: int = 1) -> Dict[str, np.ndarray]:

        if image.ndim != 2:
            raise ValueError("小波变换需要2D图像输入")
        
        coeffs = pywt.wavedec2(image, wavelet, level=level)
        wavelet_images = {}
        
        wavelet_images['LL'] = coeffs[0]
        
        for i in range(1, len(coeffs)):
            (LH, HL, HH) = coeffs[i]
            suffix = '' if i == 1 else f'_{i}'
            wavelet_images[f'LH{suffix}'] = LH
            wavelet_images[f'HL{suffix}'] = HL
            wavelet_images[f'HH{suffix}'] = HH
        
        return wavelet_images
    
    def binarize(self, image: np.ndarray, 
                threshold: Optional[float] = None,
                method: str = 'percentile',
                percentile: float = 65) -> np.ndarray:

        if method == 'fixed' and threshold is not None:
            binary = (image > threshold).astype(np.uint8)
        elif method == 'percentile':
            threshold = np.percentile(image, percentile)
            binary = (image > threshold).astype(np.uint8)
        elif method == 'otsu':
            threshold = filters.threshold_otsu(image)
            binary = (image > threshold).astype(np.uint8)
        elif method == 'mean':
            threshold = np.mean(image)
            binary = (image > threshold).astype(np.uint8)
        else:
            threshold = np.percentile(image, 50)
            binary = (image > threshold).astype(np.uint8)
        
        return binary
    
    def detect_edges(self, binary_image: np.ndarray, 
                    method: str = 'canny') -> np.ndarray:
        
        if binary_image.dtype != np.uint8:
            binary_image = (binary_image * 255).astype(np.uint8)
        
        if method == 'canny':
            edges = cv2.Canny(binary_image, 50, 150)
        elif method == 'sobel':
            sobelx = cv2.Sobel(binary_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(binary_image, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = (edges > 0).astype(np.uint8) * 255
        elif method == 'morphological':
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(binary_image, kernel, iterations=1)
            eroded = cv2.erode(binary_image, kernel, iterations=1)
            edges = dilated - eroded
        else:
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            edges = np.zeros_like(binary_image)
            cv2.drawContours(edges, contours, -1, 255, 1)
        
        return edges
    
    def denoise(self, image: np.ndarray, 
               method: str = 'gaussian',
               **kwargs) -> np.ndarray:

        if method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            denoised = ndimage.gaussian_filter(image, sigma=sigma)
        elif method == 'median':
            size = kwargs.get('size', 3)
            denoised = ndimage.median_filter(image, size=size)
        elif method == 'bilateral':
            img_uint8 = self.convert_to_8bit(image)
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            denoised = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
            denoised = denoised.astype(float) / 255.0 * (image.max() - image.min()) + image.min()
        elif method == 'nlm':
            img_uint8 = self.convert_to_8bit(image)
            h = kwargs.get('h', 10)
            template_window_size = kwargs.get('template_window_size', 7)
            search_window_size = kwargs.get('search_window_size', 21)
            denoised = cv2.fastNlMeansDenoising(img_uint8, None, h, 
                                               template_window_size, search_window_size)
            denoised = denoised.astype(float) / 255.0 * (image.max() - image.min()) + image.min()
        else:
            denoised = image.copy()
        
        return denoised