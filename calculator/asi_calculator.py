import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import signal, ndimage
from typing import Dict, Tuple, Optional
import warnings
from .base_calculator import BaseCalculator
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from utils.memory_monitor import monitor_memory


class ASICalculator(BaseCalculator):

    def __init__(self, n_components: int = 2, scale_factor: float = 255.0,
                enable_parallel: bool = True, max_workers: Optional[int] = None):

        super().__init__("ASI Calculator", enable_parallel=enable_parallel)
        self.n_components = n_components
        self.scale_factor = scale_factor
        if max_workers is not None:
            self.max_workers = max_workers
        
    @monitor_memory(threshold_percent=80)
    def calculate(self, disc_image: np.ndarray, disc_mask: np.ndarray,
                  csf_image: np.ndarray, csf_mask: np.ndarray) -> Dict[str, float]:

        self.validate_input(disc_image, disc_mask)
        self.validate_input(csf_image, csf_mask)
        
        try:
            disc_intensities = self._extract_signal_intensities(disc_image, disc_mask)
            
            if len(disc_intensities) < 30: 
                raise ValueError("椎间盘区域像素太少，无法进行可靠的GMM拟合")
            
            gmm, gmm_data = self._fit_gaussian_mixture(disc_intensities)
            
            peak_diff = self._calculate_peak_difference(gmm)
            
            csf_intensity = self._calculate_csf_intensity(csf_image, csf_mask)
            
            asi = (peak_diff / csf_intensity)
            
            means = gmm.means_.flatten()
            weights = gmm.weights_
            
            return {
                'asi': asi,
                'peak_diff': peak_diff,
                'csf_intensity': csf_intensity,
                'peak1': float(min(means)),
                'peak2': float(max(means)),
                'weight1': float(weights[np.argmin(means)]),
                'weight2': float(weights[np.argmax(means)]),
                'convergence': gmm.converged_,
                'gmm_data': gmm_data
            }
            
        except Exception as e:
            raise RuntimeError(f"ASI计算失败: {str(e)}")
    
    def _extract_signal_intensities(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:

        mask = mask.astype(bool)
        
        intensities = image[mask].flatten()
        
        intensities = intensities[np.isfinite(intensities)]
        
        return intensities
    
    @monitor_memory(threshold_percent=75)
    def _fit_gaussian_mixture(self, intensities: np.ndarray) -> Tuple[GaussianMixture, Dict]:

        X = intensities.reshape(-1, 1)
        
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=100,
            n_init=10,
            random_state=42
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm.fit(X)
        
        gmm_data = self._prepare_gmm_visualization_data(intensities, gmm)
        
        return gmm, gmm_data
    
    def _calculate_peak_difference(self, gmm: GaussianMixture) -> float:
        
        means = gmm.means_.flatten()
        
        if len(means) != 2:
            self.logger.warning(f"GMM拟合得到{len(means)}个峰值，期望2个。将采用备用策略。")
            if len(means) > 2:
                peak_diff = abs(np.max(means) - np.min(means))
            elif len(means) == 1:
                peak_diff = float(np.sqrt(gmm.covariances_[0][0, 0]))
            else:
                return 0.0
        else:
            peak_diff = abs(means[1] - means[0])
        
        return float(peak_diff)
    
    def _calculate_csf_intensity(self, image: np.ndarray, csf_mask: np.ndarray) -> float:
        
        csf_intensities = self._extract_signal_intensities(image, csf_mask)
        
        if len(csf_intensities) == 0:
            self.logger.warning("CSF掩模中没有有效像素，使用全图95%分位数作为备用参考")
            valid_pixels = image[image > 0]
            if len(valid_pixels) > 0:
                return float(np.percentile(valid_pixels, 95))
            else:
                self.logger.error("图像中没有有效像素，无法计算CSF参考值")
                raise ValueError("图像中没有有效像素")
        
        if len(csf_intensities) > 20:
            lower_bound = np.percentile(csf_intensities, 25)
            pure_pixels = csf_intensities[csf_intensities > lower_bound]
            
            if len(pure_pixels) > 10:
                return float(np.mean(pure_pixels))

        return float(np.mean(csf_intensities))
    
    def _prepare_gmm_visualization_data(self, intensities: np.ndarray, 
                                      gmm: GaussianMixture) -> Dict:

        counts, bin_edges = np.histogram(intensities, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        x_range = np.linspace(intensities.min(), intensities.max(), 1000)
        X_range = x_range.reshape(-1, 1)
        
        log_prob = gmm.score_samples(X_range)
        pdf = np.exp(log_prob)
        
        components = []
        for i in range(self.n_components):
            single_gmm = GaussianMixture(n_components=1)
            single_gmm.means_ = gmm.means_[i:i+1]
            single_gmm.covariances_ = gmm.covariances_[i:i+1]
            single_gmm.weights_ = np.array([1.0])
            single_gmm.precisions_cholesky_ = gmm.precisions_cholesky_[i:i+1]
            
            comp_log_prob = single_gmm.score_samples(X_range)
            comp_pdf = np.exp(comp_log_prob) * gmm.weights_[i]
            
            components.append({
                'x': x_range,
                'pdf': comp_pdf,
                'mean': float(gmm.means_[i]),
                'weight': float(gmm.weights_[i])
            })
        
        return {
            'histogram': {'bin_centers': bin_centers, 'counts': counts},
            'gmm_fit': {'x': x_range, 'pdf': pdf},
            'components': components,
            'peaks': gmm.means_.flatten()
        }
    
    def process_multi_slice(self, image_slices: list, disc_masks: list,
                        csf_masks: list, use_parallel: Optional[bool] = None) -> Dict[str, float]:
        if use_parallel is None:
            use_parallel = self.enable_parallel
        
        if use_parallel:
            return self.process_multi_slice_parallel(image_slices, disc_masks, csf_masks)
        
        if not (len(image_slices) == len(disc_masks) == len(csf_masks)):
            raise ValueError("切片列表长度不一致")
        
        asi_results = []
        valid_slices = 0
        
        for i, (img, disc_mask, csf_mask) in enumerate(zip(image_slices, disc_masks, csf_masks)):
            try:
                result = self.calculate(img, disc_mask, img, csf_mask)
                asi_results.append(result)
                valid_slices += 1
            except Exception as e:
                print(f"切片{i}处理失败: {str(e)}")
                continue
        
        if valid_slices == 0:
            raise ValueError("没有成功处理的切片")
        
        avg_result = {
            'asi': np.mean([r['asi'] for r in asi_results]),
            'asi_std': np.std([r['asi'] for r in asi_results]),
            'peak_diff': np.mean([r['peak_diff'] for r in asi_results]),
            'csf_intensity': np.mean([r['csf_intensity'] for r in asi_results]),
            'peak1': np.mean([r['peak1'] for r in asi_results]),
            'peak2': np.mean([r['peak2'] for r in asi_results]),
            'valid_slices': valid_slices
        }
        
        return avg_result
    
    @monitor_memory(threshold_percent=70)
    def process_multi_slice_parallel(self, image_slices: list, disc_masks: list,
                                csf_masks: list) -> Dict[str, float]:

        if not self.enable_parallel or len(image_slices) < 3:
            return self.process_multi_slice(image_slices, disc_masks, csf_masks)
        
        def process_single_slice(args):
            i, img, disc_mask, csf_mask = args
            try:
                result = self.calculate(img, disc_mask, img, csf_mask)
                return (i, result, None)
            except Exception as e:
                return (i, None, str(e))
        
        args_list = [(i, img, disc_mask, csf_mask) 
                    for i, (img, disc_mask, csf_mask) 
                    in enumerate(zip(image_slices, disc_masks, csf_masks))]
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(args_list))) as executor:
            results = list(executor.map(process_single_slice, args_list))
        
        asi_results = []
        valid_slices = 0
        
        for i, result, error in sorted(results, key=lambda x: x[0]):
            if error:
                self.logger.warning(f"切片{i}处理失败: {error}")
            elif result:
                asi_results.append(result)
                valid_slices += 1
        
        if valid_slices == 0:
            raise ValueError("没有成功处理的切片")
        
        avg_result = {
            'asi': np.mean([r['asi'] for r in asi_results]),
            'asi_std': np.std([r['asi'] for r in asi_results]),
            'peak_diff': np.mean([r['peak_diff'] for r in asi_results]),
            'csf_intensity': np.mean([r['csf_intensity'] for r in asi_results]),
            'peak1': np.mean([r['peak1'] for r in asi_results]),
            'peak2': np.mean([r['peak2'] for r in asi_results]),
            'valid_slices': valid_slices
        }
        
        return avg_result
