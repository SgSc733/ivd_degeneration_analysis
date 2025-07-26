import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Tuple, List
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
import cv2


class Visualizer:
    
    def __init__(self, style: str = 'seaborn', dpi: int = 150):

        plt.style.use(style)
        self.dpi = dpi
        self.colors = {
            'disc': '#4ECDC4',
            'upper_vertebra': '#FF6B6B',
            'lower_vertebra': '#45B7D1',
            'csf': '#FFA07A',
            'overlay': '#95E1D3'
        }
    
    def visualize_dhi_result(self, image: np.ndarray, 
                           upper_mask: np.ndarray,
                           disc_mask: np.ndarray,
                           lower_mask: np.ndarray,
                           result: Dict,
                           save_path: Optional[str] = None):

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像')
        axes[0].axis('off')

        overlay = self._create_overlay(image, upper_mask, disc_mask, lower_mask)
        axes[1].imshow(overlay)
        axes[1].set_title('分割结果')
        axes[1].axis('off')

        axes[2].text(0.1, 0.9, f"DHI: {result['dhi']:.3f}", 
                    transform=axes[2].transAxes, fontsize=14)
        axes[2].text(0.1, 0.7, f"椎间盘高度: {result['disc_height']:.1f}",
                    transform=axes[2].transAxes, fontsize=12)
        axes[2].text(0.1, 0.5, f"上椎体高度: {result['upper_vh']:.1f}",
                    transform=axes[2].transAxes, fontsize=12)
        axes[2].text(0.1, 0.3, f"下椎体高度: {result['lower_vh']:.1f}",
                    transform=axes[2].transAxes, fontsize=12)
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def visualize_asi_result(self, image: np.ndarray,
                           disc_mask: np.ndarray,
                           csf_mask: np.ndarray,
                           result: Dict,
                           save_path: Optional[str] = None):

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(image, cmap='gray')
        self._add_roi_overlay(axes[0, 0], disc_mask, csf_mask)
        axes[0, 0].set_title('原始图像与ROI')
        axes[0, 0].axis('off')

        if 'gmm_data' in result:
            self._plot_gmm_fit(axes[0, 1], result['gmm_data'])
        axes[0, 1].set_title('信号强度分布与GMM拟合')

        self._plot_intensity_heatmap(axes[1, 0], image, disc_mask)
        axes[1, 0].set_title('椎间盘信号强度热图')

        axes[1, 1].text(0.1, 0.9, f"ASI: {result['asi']:.2f}", 
                       transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].text(0.1, 0.7, f"峰值差异: {result['peak_diff']:.2f}",
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.5, f"CSF强度: {result['csf_intensity']:.2f}",
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.3, f"峰值1: {result['peak1']:.2f}",
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.1, f"峰值2: {result['peak2']:.2f}",
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

    def visualize_t2si_result(self, image: np.ndarray,
                            disc_mask: np.ndarray,
                            csf_mask: np.ndarray,
                            result: Dict,
                            save_path: Optional[str] = None):

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(image, cmap='gray')
        roi_overlay = np.ma.masked_where(result['roi_mask'] == 0, result['roi_mask'])
        axes[0, 0].imshow(roi_overlay, alpha=0.5, cmap='Reds')
        axes[0, 0].set_title(f'{result["roi_method"]}-ROI')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(image, cmap='gray')
        csf_overlay = np.ma.masked_where(csf_mask == 0, csf_mask)
        axes[0, 1].imshow(csf_overlay, alpha=0.5, cmap='Blues')
        axes[0, 1].set_title('CSF参考区域')
        axes[0, 1].axis('off')

        disc_pixels = image[disc_mask > 0]
        roi_pixels = image[result['roi_mask'] > 0]
        
        axes[1, 0].hist(disc_pixels, bins=50, alpha=0.5, label='整个椎间盘', density=True)
        axes[1, 0].hist(roi_pixels, bins=30, alpha=0.7, label=f'{result["roi_method"]}-ROI', density=True)
        axes[1, 0].axvline(result['csf_si'], color='blue', linestyle='--', label=f'CSF SI: {result["csf_si"]:.1f}')
        axes[1, 0].set_xlabel('信号强度')
        axes[1, 0].set_ylabel('概率密度')
        axes[1, 0].legend()
        axes[1, 0].set_title('信号强度分布')

        axes[1, 1].text(0.1, 0.9, f"ROI方法: {result['roi_method']}", 
                    transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.7, f"SI比值: {result['si_ratio']:.3f}", 
                    transform=axes[1, 1].transAxes, fontsize=14, weight='bold')
        axes[1, 1].text(0.1, 0.5, f"ROI SI: {result['roi_si']:.1f}", 
                    transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.3, f"CSF SI: {result['csf_si']:.1f}", 
                    transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.1, f"ROI大小: {result['roi_size']} 像素", 
                    transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def visualize_combined_results(self, image: np.ndarray,
                                 dhi_result: Dict,
                                 asi_result: Dict,
                                 fd_result: Optional[Dict] = None,
                                 save_path: Optional[str] = None):

        num_analyses = 3 if fd_result else 2
        fig, axes = plt.subplots(1, num_analyses + 1, figsize=(5 * (num_analyses + 1), 5))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像')
        axes[0].axis('off')

        axes[1].bar(['DHI', 'DH', 'VH上', 'VH下'], 
                   [dhi_result['dhi'], dhi_result['disc_height'], 
                    dhi_result['upper_vh'], dhi_result['lower_vh']])
        axes[1].set_title('DHI分析')
        axes[1].set_ylabel('数值')

        axes[2].bar(['ASI', '峰值差', 'CSF强度'], 
                   [asi_result['asi'], asi_result['peak_diff'], 
                    asi_result['csf_intensity']])
        axes[2].set_title('ASI分析')
        axes[2].set_ylabel('数值')

        if fd_result and num_analyses == 3:
            axes[3].text(0.5, 0.5, f"FD: {fd_result.get('fd', 'N/A'):.3f}",
                        ha='center', va='center', fontsize=16,
                        transform=axes[3].transAxes)
            axes[3].set_title('分形维度')
            axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def _create_overlay(self, image: np.ndarray, upper: np.ndarray, 
                       disc: np.ndarray, lower: np.ndarray) -> np.ndarray:

        img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        overlay = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

        overlay[upper > 0] = overlay[upper > 0] * 0.5 + np.array(self._hex_to_rgb(self.colors['upper_vertebra'])) * 0.5
        overlay[disc > 0] = overlay[disc > 0] * 0.5 + np.array(self._hex_to_rgb(self.colors['disc'])) * 0.5
        overlay[lower > 0] = overlay[lower > 0] * 0.5 + np.array(self._hex_to_rgb(self.colors['lower_vertebra'])) * 0.5
        
        return overlay.astype(np.uint8)
    
    def _add_roi_overlay(self, ax, disc_mask: np.ndarray, csf_mask: np.ndarray):

        disc_overlay = np.ma.masked_where(disc_mask == 0, disc_mask)
        csf_overlay = np.ma.masked_where(csf_mask == 0, csf_mask)
        
        ax.imshow(disc_overlay, alpha=0.3, cmap='Greens')
        ax.imshow(csf_overlay, alpha=0.3, cmap='Blues')
    
    def _plot_gmm_fit(self, ax, gmm_data: Dict):

        if 'histogram' in gmm_data:
            hist = gmm_data['histogram']
            ax.bar(hist['bin_centers'], hist['counts'], 
                  width=hist['bin_centers'][1] - hist['bin_centers'][0],
                  alpha=0.6, color='gray', label='直方图')

        if 'gmm_fit' in gmm_data:
            gmm = gmm_data['gmm_fit']
            ax.plot(gmm['x'], gmm['pdf'], 'r-', linewidth=2, label='GMM拟合')

        if 'components' in gmm_data:
            for i, comp in enumerate(gmm_data['components']):
                ax.plot(comp['x'], comp['pdf'], '--', linewidth=1.5, 
                       label=f'组分{i+1}')

        if 'peaks' in gmm_data:
            for i, peak in enumerate(gmm_data['peaks']):
                ax.axvline(peak, color='green', linestyle=':', 
                          label=f'峰值{i+1}' if i < 2 else '')
        
        ax.set_xlabel('信号强度')
        ax.set_ylabel('概率密度')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_intensity_heatmap(self, ax, image: np.ndarray, mask: np.ndarray):

        masked_image = np.ma.masked_where(mask == 0, image)
        im = ax.imshow(masked_image, cmap='hot', aspect='auto')
        plt.colorbar(im, ax=ax, label='信号强度')
        ax.axis('off')
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:

        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def plot_feature_comparison(self, features_dict: Dict[str, List[float]], 
                               save_path: Optional[str] = None):

        fig, ax = plt.subplots(figsize=(10, 6))

        feature_names = list(features_dict.keys())
        data = [features_dict[name] for name in feature_names]

        box_plot = ax.boxplot(data, labels=feature_names, patch_artist=True)

        colors = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('特征值')
        ax.set_title('特征分布对比')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()