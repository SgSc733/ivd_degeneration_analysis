import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import SimpleITK as sitk
import cv2
from scipy import stats, ndimage
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import Preprocessor
from utils.image_io import ImageIO
from config import Config

HAS_PYDICOM = False
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    print("警告: pydicom未安装，将无法直接读取DICOM文件")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import Preprocessor
from utils.image_io import ImageIO
from config import Config

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PreprocessingValidator:
    
    def __init__(self, root):
        self.root = root
        self.root.title("MRI预处理效果验证工具")
        self.root.geometry("1200x800")

        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.preprocessor = Preprocessor()
        self.image_io = ImageIO()
        self.config = Config()

        self.current_image = None
        self.current_mask = None
        self.current_mask_slice = None 
        self.current_spacing = None
        self.preprocessing_results = {}
        self.validation_results = {}

        self.setup_gui()
        
    def setup_gui(self):

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        title_label = ttk.Label(main_frame, text="MRI预处理效果验证工具", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(file_frame, text="MRI图像:").grid(row=0, column=0, sticky=tk.W)
        self.image_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.image_path_var, width=50).grid(
            row=0, column=1, padx=5)
        ttk.Button(file_frame, text="选择", 
                  command=self.select_image).grid(row=0, column=2)
        ttk.Button(file_frame, text="选择文件", 
                command=self.select_image).grid(row=0, column=2)
        ttk.Button(file_frame, text="选择文件夹", 
                command=self.select_folder).grid(row=0, column=3)

        ttk.Label(file_frame, text="分割掩模(可选):").grid(row=1, column=0, sticky=tk.W)
        self.mask_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.mask_path_var, width=50).grid(
            row=1, column=1, padx=5)
        ttk.Button(file_frame, text="选择", 
                  command=self.select_mask).grid(row=1, column=2)

        mask_option_frame = ttk.Frame(file_frame)
        mask_option_frame.grid(row=2, column=0, columnspan=3, pady=5)

        self.use_mask_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mask_option_frame, text="使用掩模进行ROI分析", 
                    variable=self.use_mask_var,
                    command=self.update_checkbox_states).pack(side=tk.LEFT, padx=5)

        ttk.Button(file_frame, text="加载数据", 
                  command=self.load_data).grid(row=3, column=1, pady=10)

        process_frame = ttk.LabelFrame(main_frame, text="预处理步骤选择", padding="10")
        process_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        info_label = ttk.Label(process_frame, text="(标*的步骤需要掩模)", 
                            font=('Arial', 9, 'italic'))
        info_label.grid(row=0, column=0, columnspan=4, pady=(0, 5))

        self.preprocess_options = {
            'resample': tk.BooleanVar(value=True),
            'texture_norm': tk.BooleanVar(value=True),
            'signal_norm': tk.BooleanVar(value=True),
            'fractal_prep': tk.BooleanVar(value=True),
            'shape_prep': tk.BooleanVar(value=True),
            'filter_log': tk.BooleanVar(value=True),
            'filter_wavelet': tk.BooleanVar(value=True)
        }

        self.preprocess_checkboxes = {}

        cb1 = ttk.Checkbutton(process_frame, text="图像重采样", 
                            variable=self.preprocess_options['resample'])
        cb1.grid(row=1, column=0, sticky=tk.W, padx=5)
        self.preprocess_checkboxes['resample'] = cb1

        cb2 = ttk.Checkbutton(process_frame, text="纹理特征预处理*", 
                            variable=self.preprocess_options['texture_norm'])
        cb2.grid(row=1, column=1, sticky=tk.W, padx=5)
        self.preprocess_checkboxes['texture_norm'] = cb2

        cb3 = ttk.Checkbutton(process_frame, text="信号强度预处理", 
                            variable=self.preprocess_options['signal_norm'])
        cb3.grid(row=1, column=2, sticky=tk.W, padx=5)
        self.preprocess_checkboxes['signal_norm'] = cb3

        cb4 = ttk.Checkbutton(process_frame, text="分形维度预处理*", 
                            variable=self.preprocess_options['fractal_prep'])
        cb4.grid(row=2, column=0, sticky=tk.W, padx=5)
        self.preprocess_checkboxes['fractal_prep'] = cb4

        cb5 = ttk.Checkbutton(process_frame, text="形状特征预处理*", 
                            variable=self.preprocess_options['shape_prep'])
        cb5.grid(row=2, column=1, sticky=tk.W, padx=5)
        self.preprocess_checkboxes['shape_prep'] = cb5

        cb6 = ttk.Checkbutton(process_frame, text="LoG滤波", 
                            variable=self.preprocess_options['filter_log'])
        cb6.grid(row=2, column=2, sticky=tk.W, padx=5)
        self.preprocess_checkboxes['filter_log'] = cb6

        cb7 = ttk.Checkbutton(process_frame, text="小波滤波", 
                            variable=self.preprocess_options['filter_wavelet'])
        cb7.grid(row=2, column=3, sticky=tk.W, padx=5)
        self.preprocess_checkboxes['filter_wavelet'] = cb7

        self.update_checkbox_states()

        ttk.Button(process_frame, text="执行预处理验证", 
                  command=self.run_validation).grid(row=3, column=1, pady=10)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.visual_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visual_frame, text="可视化对比")

        self.quant_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.quant_frame, text="定量评估")

        self.report_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.report_frame, text="详细报告")

        self.report_text = tk.Text(self.report_frame, wrap=tk.WORD, height=25)
        report_scrollbar = ttk.Scrollbar(self.report_frame, orient="vertical", 
                                       command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=report_scrollbar.set)
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        report_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=4, column=0, columnspan=3, pady=5)

        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def update_checkbox_states(self):

        has_real_mask = (self.use_mask_var.get() and 
                        self.current_mask is not None and 
                        not np.all(self.current_mask == 1))

        mask_required_options = ['texture_norm', 'fractal_prep', 'shape_prep']
        
        for option_name in mask_required_options:
            if option_name in self.preprocess_options:
                checkbox = self.preprocess_checkboxes.get(option_name)
                if checkbox:
                    if has_real_mask:
                        checkbox.configure(state='normal')
                    else:
                        checkbox.configure(state='disabled')
                        self.preprocess_options[option_name].set(False)
        
    def select_image(self):

        filename = filedialog.askopenfilename(
            title="选择MRI图像",
            filetypes=[
                ("DICOM文件", "*.dcm"),
                ("NIfTI文件", "*.nii *.nii.gz"), 
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.image_path_var.set(filename)
            
    def select_mask(self):

        filename = filedialog.askopenfilename(
            title="选择分割掩模",
            filetypes=[
                ("DICOM文件", "*.dcm"), 
                ("NIfTI文件", "*.nii *.nii.gz"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.mask_path_var.set(filename)
        
    def load_data(self):

        try:
            image_path = self.image_path_var.get()
            
            if not image_path:
                messagebox.showerror("错误", "请选择图像文件")
                return
                
            self.status_var.set("加载数据中...")

            if os.path.isdir(image_path):
                dicom_files = self.find_dicom_files(image_path)
                if dicom_files:
                    image_path = dicom_files[0]  
                    self.logger.info(f"从文件夹中选择文件: {image_path}")
                else:
                    messagebox.showerror("错误", "文件夹中没有找到DICOM文件")
                    return

            if image_path.lower().endswith('.dcm'):
                if not HAS_PYDICOM:
                    messagebox.showerror("错误", "未安装pydicom，无法读取DICOM文件\n请运行: pip install pydicom")
                    return
                    
                ds = pydicom.dcmread(image_path)
                self.current_image = ds.pixel_array.astype(np.float64)

                if self.current_image.ndim == 2:
                    self.current_image = self.current_image[np.newaxis, :, :]

                if hasattr(ds, 'PixelSpacing'):
                    pixel_spacing = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]
                    self.current_spacing = pixel_spacing  # [y, x]
                else:
                    self.current_spacing = [1.0, 1.0]
                    self.logger.warning("DICOM文件没有PixelSpacing信息，使用默认值[1.0, 1.0]")
            else:
                image_sitk = sitk.ReadImage(image_path)
                self.current_image = self.image_io.sitk_to_numpy(image_sitk)

                spacing = list(image_sitk.GetSpacing())[::-1]  # xyz -> zyx
                self.current_spacing = spacing[:2] if len(spacing) >= 2 else spacing + [1.0]

            mask_path = self.mask_path_var.get()
            if mask_path and self.use_mask_var.get():
                if mask_path.lower().endswith('.dcm'):
                    if HAS_PYDICOM:
                        ds_mask = pydicom.dcmread(mask_path)
                        self.current_mask = ds_mask.pixel_array.astype(np.uint8)
                        if self.current_mask.ndim == 2:
                            self.current_mask = self.current_mask[np.newaxis, :, :]
                    else:
                        messagebox.showerror("错误", "无法读取DICOM掩模文件")
                        return
                else:
                    mask_sitk = sitk.ReadImage(mask_path)
                    self.current_mask = self.image_io.sitk_to_numpy(mask_sitk)

                if self.current_image.shape != self.current_mask.shape:
                    messagebox.showerror("错误", "图像和掩模尺寸不匹配")
                    return

                middle_idx = self.current_image.shape[0] // 2
                self.current_mask_slice = self.current_mask[middle_idx]
            else:
                self.current_mask = np.ones_like(self.current_image, dtype=np.uint8)
                self.use_mask_var.set(False)

                middle_idx = self.current_image.shape[0] // 2
                self.current_mask_slice = self.current_mask[middle_idx]

            middle_idx = self.current_image.shape[0] // 2
            self.current_slice = self.current_image[middle_idx]

            self.update_checkbox_states()
            
            mask_status = "使用掩模" if self.use_mask_var.get() else "全图分析"
            self.status_var.set(f"数据加载成功 - 图像形状: {self.current_image.shape}, "
                            f"间距: {[f'{s:.2f}' for s in self.current_spacing]}, {mask_status}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载数据失败: {str(e)}")
            self.status_var.set("加载失败")
            self.logger.error(f"加载数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())


    def select_folder(self):

        folder_path = filedialog.askdirectory(
            title="选择包含DICOM文件的文件夹"
        )
        if folder_path:
            self.image_path_var.set(folder_path)
            dicom_files = self.find_dicom_files(folder_path)
            if dicom_files:
                self.current_dicom_files = dicom_files
                self.status_var.set(f"找到 {len(dicom_files)} 个DICOM文件")
            else:
                messagebox.showwarning("警告", "该文件夹中没有找到DICOM文件")

    def find_dicom_files(self, directory: str) -> List[str]:

        dicom_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        return sorted(dicom_files)
            
    def run_validation(self):

        if self.current_image is None:
            messagebox.showerror("错误", "请先加载数据")
            return

        has_real_mask = (self.use_mask_var.get() and 
                        self.current_mask is not None and 
                        not np.all(self.current_mask == 1))
        
        if self.preprocess_options['shape_prep'].get() and not has_real_mask:
            messagebox.showwarning("警告", "形状特征预处理需要真实的掩模，已自动跳过")
            self.preprocess_options['shape_prep'].set(False)

        mask_required = ['texture_norm', 'fractal_prep']
        for option in mask_required:
            if self.preprocess_options[option].get() and not has_real_mask:
                messagebox.showwarning("警告", f"{self._get_process_display_name(option)}需要真实的掩模，将使用全图分析")
            
        try:
            self.status_var.set("执行预处理验证中...")
            self.preprocessing_results = {}
            self.validation_results = {}

            slice_spacing = self.current_spacing[:2] + [1.0]  # 2D切片的间距

            if not has_real_mask:
                self.status_var.set("执行预处理验证中（全图分析模式）...")

            if self.preprocess_options['resample'].get():
                self._validate_resampling(self.current_slice, self.current_mask_slice, 
                                        slice_spacing)

            if self.preprocess_options['texture_norm'].get():
                self._validate_texture_preprocessing(self.current_slice, 
                                                   self.current_mask_slice, 
                                                   slice_spacing)

            if self.preprocess_options['signal_norm'].get():
                self._validate_signal_preprocessing(self.current_slice, 
                                                  self.current_mask_slice, 
                                                  slice_spacing)

            if self.preprocess_options['fractal_prep'].get():
                self._validate_fractal_preprocessing(self.current_slice, 
                                                   self.current_mask_slice, 
                                                   slice_spacing)

            if self.preprocess_options['shape_prep'].get() and self.use_mask_var.get():
                self._validate_shape_preprocessing(self.current_mask_slice, slice_spacing)

            if self.preprocess_options['filter_log'].get():
                self._validate_log_filter(self.current_slice, self.current_mask_slice)
                
            if self.preprocess_options['filter_wavelet'].get():
                self._validate_wavelet_filter(self.current_slice)

            self._display_visual_results()
            self._display_quantitative_results()
            self._generate_report()
            
            self.status_var.set("验证完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"验证失败: {str(e)}")
            self.status_var.set("验证失败")
            
    def _validate_resampling(self, image, mask, spacing):

        target_size = [512, 512]  # 固定目标尺寸

        if len(spacing) < 2:
            spacing = spacing + [1.0] * (2 - len(spacing))

        resampled_image, actual_spacing = self.preprocessor.resample_image(
            image, spacing, target_size=target_size, 
            interpolation='linear', is_label=False
        )
        resampled_mask, _ = self.preprocessor.resample_image(
            mask, spacing, target_size=target_size,
            interpolation='nearest', is_label=True
        )

        self.preprocessing_results['resample'] = {
            'original_image': image,
            'original_mask': mask,
            'processed_image': resampled_image,
            'processed_mask': resampled_mask,
            'original_spacing': spacing,
            'target_size': target_size,
            'actual_spacing': actual_spacing
        }

        self.validation_results['resample'] = {
            'original_shape': image.shape,
            'resampled_shape': resampled_image.shape,
            'size_change': f"{image.shape} -> {resampled_image.shape}",
            'spacing_info': {
                'original': [f'{s:.2f}' for s in spacing[:2]],
                'actual': [f'{s:.2f}' for s in actual_spacing[:2]] if actual_spacing else ['N/A', 'N/A']
            },
            'interpolation_quality': self._assess_interpolation_quality(
                mask, resampled_mask, spacing, target_size
            )
        }
        
    def _validate_texture_preprocessing(self, image, mask, spacing):

        if len(spacing) < 2:
            spacing = spacing + [1.0] * (2 - len(spacing))

        if mask is None or np.all(mask == 0):
            self.log_message("警告：纹理预处理需要有效的掩模")
            return

        processed_image, processed_mask = self.preprocessor.preprocess_for_texture(
            image, mask, spacing, target_size=[512, 512], bin_width=16
        )

        self.preprocessing_results['texture'] = {
            'original_image': image,
            'processed_image': processed_image,
            'mask': processed_mask
        }

        original_stats = self._calculate_roi_statistics(image, mask)
        processed_stats = self._calculate_roi_statistics(processed_image, processed_mask)
        
        self.validation_results['texture'] = {
            'original_stats': original_stats,
            'processed_stats': processed_stats,
            'z_score_mean': processed_stats['mean'],
            'z_score_std': processed_stats['std'],
            'discretization_levels': len(np.unique(processed_image[processed_mask > 0])),
            'bin_width': 16
        }
        
    def _validate_signal_preprocessing(self, image, mask, spacing):

        if len(spacing) < 2:
            spacing = spacing + [1.0] * (2 - len(spacing))

        processed_image, processed_mask = self.preprocessor.preprocess_for_signal_intensity(
            image, mask, spacing, target_size=[512, 512]
        )

        self.preprocessing_results['signal'] = {
            'original_image': image,
            'processed_image': processed_image,
            'mask': processed_mask
        }

        original_stats = self._calculate_roi_statistics(image, mask)
        processed_stats = self._calculate_roi_statistics(processed_image, processed_mask)

        if image.shape != processed_image.shape:
            processed_resized = cv2.resize(processed_image, (image.shape[1], image.shape[0]))
            mask_resized = cv2.resize(processed_mask.astype(float), (mask.shape[1], mask.shape[0])) > 0.5
            correlation = np.corrcoef(
                image[mask > 0].flatten(), 
                processed_resized[mask > 0].flatten()
            )[0, 1]
        else:
            correlation = np.corrcoef(
                image[mask > 0].flatten(), 
                processed_image[processed_mask > 0].flatten()
            )[0, 1]
        
        self.validation_results['signal'] = {
            'original_stats': original_stats,
            'processed_stats': processed_stats,
            'intensity_preserved': correlation,
            'shape_change': f"{image.shape} -> {processed_image.shape}",
            'pixel_value_range': {
                'original': [float(np.min(image)), float(np.max(image))],
                'processed': [float(np.min(processed_image)), float(np.max(processed_image))]
            }
        }
        
    def _validate_fractal_preprocessing(self, image, mask, spacing):

        if len(spacing) < 2:
            spacing = spacing + [1.0] * (2 - len(spacing))
        edges, processed_mask = self.preprocessor.preprocess_for_fractal(
            image, mask, spacing, target_size=[512, 512],
            window_center=128, window_width=255, threshold_percentile=65
        )

        self.preprocessing_results['fractal'] = {
            'original_image': image,
            'edges': edges,
            'mask': processed_mask
        }

        edge_pixels = np.sum(edges > 0)
        total_pixels = np.sum(processed_mask > 0)
        
        self.validation_results['fractal'] = {
            'edge_pixels': edge_pixels,
            'total_pixels': total_pixels,
            'edge_ratio': edge_pixels / total_pixels if total_pixels > 0 else 0,
            'edge_continuity': self._assess_edge_continuity(edges)
        }
        
    def _validate_shape_preprocessing(self, mask, spacing):

        if len(spacing) < 2:
            spacing = spacing + [1.0] * (2 - len(spacing))
        binary_mask = self.preprocessor.preprocess_for_shape(
            mask, spacing, target_size=[512, 512]
        )

        self.preprocessing_results['shape'] = {
            'original_mask': mask,
            'binary_mask': binary_mask
        }
        
        original_area = np.sum(mask > 0)
        processed_area = np.sum(binary_mask > 0)
        
        self.validation_results['shape'] = {
            'original_area': original_area,
            'processed_area': processed_area,
            'area_preservation': processed_area / original_area if original_area > 0 else 0,
            'shape_similarity': self._calculate_dice_coefficient(mask > 0, binary_mask > 0)
        }
        
    def _validate_log_filter(self, image, mask):

        sigma_list = [1, 3, 5]
        filtered_images = self.preprocessor.apply_log_filter(image, sigma_list)

        self.preprocessing_results['log_filter'] = {
            'original_image': image,
            'filtered_images': filtered_images,
            'mask': mask
        }

        self.validation_results['log_filter'] = {
            'num_scales': len(sigma_list),
            'sigma_values': sigma_list,
            'response_stats': {
                f'sigma_{sigma}': self._calculate_roi_statistics(
                    filtered_images[sigma], mask
                ) for sigma in sigma_list
            }
        }
        
    def _validate_wavelet_filter(self, image):

        wavelet_images = self.preprocessor.apply_wavelet_transform(
            image, wavelet='db1', level=1
        )

        self.preprocessing_results['wavelet'] = {
            'original_image': image,
            'wavelet_images': wavelet_images
        }

        self.validation_results['wavelet'] = {
            'num_components': len(wavelet_images),
            'components': list(wavelet_images.keys()),
            'energy_distribution': {
                comp: np.sum(img**2) for comp, img in wavelet_images.items()
            }
        }
    
    def _calculate_roi_statistics(self, image, mask):

        roi_values = image[mask > 0]
        
        if len(roi_values) == 0:
            return {
                'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                'median': 0, 'q25': 0, 'q75': 0,
                'skewness': 0, 'kurtosis': 0
            }
        
        from scipy import stats
        
        return {
            'mean': float(np.mean(roi_values)),
            'std': float(np.std(roi_values)),
            'min': float(np.min(roi_values)),
            'max': float(np.max(roi_values)),
            'median': float(np.median(roi_values)),
            'q25': float(np.percentile(roi_values, 25)),
            'q75': float(np.percentile(roi_values, 75)),
            'skewness': float(stats.skew(roi_values)),
            'kurtosis': float(stats.kurtosis(roi_values))
        }
        
    def _assess_interpolation_quality(self, original_mask, resampled_mask, 
                                    original_spacing, target_size):

        expected_shape = target_size
        actual_shape = resampled_mask.shape
        
        size_accuracy = (actual_shape[0] == expected_shape[0] and 
                        actual_shape[1] == expected_shape[1])

        if np.all(original_mask > 0) and np.all(resampled_mask > 0):

            edge_similarity = 1.0  
            quality_score = 'Good' if size_accuracy else 'Fair'
        else:

            original_edges = cv2.Canny((original_mask > 0).astype(np.uint8) * 255, 50, 150)

            resampled_resized = cv2.resize(
                resampled_mask.astype(np.float32), 
                (original_mask.shape[1], original_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            resampled_edges = cv2.Canny((resampled_resized > 0).astype(np.uint8) * 255, 50, 150)

            if np.sum(original_edges | resampled_edges) > 0:
                edge_similarity = np.sum(original_edges & resampled_edges) / np.sum(original_edges | resampled_edges)
            else:
                edge_similarity = 1.0 
            
            quality_score = 'Good' if size_accuracy and edge_similarity > 0.8 else \
                        'Fair' if edge_similarity > 0.6 else 'Poor'
        
        return {
            'size_accuracy': size_accuracy,
            'edge_similarity': float(edge_similarity),
            'quality_score': quality_score,
            'expected_shape': expected_shape,
            'actual_shape': list(actual_shape),
            'is_full_image': np.all(original_mask > 0)
        }
        
    def _assess_edge_continuity(self, edges):

        num_labels, labels = cv2.connectedComponents(edges.astype(np.uint8))

        if num_labels > 1:
            component_sizes = [np.sum(labels == i) for i in range(1, num_labels)]
            max_component_ratio = max(component_sizes) / np.sum(edges > 0) if np.sum(edges > 0) > 0 else 0
        else:
            max_component_ratio = 0
            
        return {
            'num_components': num_labels - 1,  
            'largest_component_ratio': float(max_component_ratio),
            'continuity_score': 'Good' if max_component_ratio > 0.8 else 
                              'Fair' if max_component_ratio > 0.5 else 'Poor'
        }
        
    def _calculate_dice_coefficient(self, mask1, mask2):

        intersection = np.sum(mask1 & mask2)
        return 2 * intersection / (np.sum(mask1) + np.sum(mask2)) if (np.sum(mask1) + np.sum(mask2)) > 0 else 0
        
    def _display_visual_results(self):

        for widget in self.visual_frame.winfo_children():
            widget.destroy()
            
        container = ttk.Frame(self.visual_frame)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        v_scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
        scrollable_frame = ttk.Frame(canvas)

        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)

        fig = plt.figure(figsize=(20, 24))  

        fig.suptitle('MRI预处理效果可视化对比', fontsize=16, fontweight='bold', y=0.995)
        
        plot_idx = 1
        total_plots = len([k for k, v in self.preprocess_options.items() if v.get()]) + 2  
        
        def add_row_title(fig, row_idx, total_rows, title, y_offset=0.02):

            y_position = 1 - (row_idx - 0.5) / total_rows - y_offset
            fig.text(0.02, y_position, title, fontsize=12, fontweight='bold', 
                    rotation=90, verticalalignment='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
        
        current_row = 1

        if self.current_slice is not None:
            add_row_title(fig, current_row, total_plots, '8位转换')
            
            original_img = self.current_slice
            
            plt.subplot(total_plots, 4, plot_idx)
            plt.imshow(original_img, cmap='gray')
            plt.title(f'Original\nType: {original_img.dtype}\nRange: [{original_img.min():.1f}, {original_img.max():.1f}]')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 1)
            img_8bit = self.preprocessor.convert_to_8bit(original_img)
            plt.imshow(img_8bit, cmap='gray')
            plt.title(f'8-bit Converted\nType: {img_8bit.dtype}\nRange: [{img_8bit.min()}, {img_8bit.max()}]')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 2)
            plt.hist(original_img.flatten(), bins=50, alpha=0.7, density=True)
            plt.xlabel('Original Intensity')
            plt.ylabel('Density')
            plt.title('Original Histogram')
            
            plt.subplot(total_plots, 4, plot_idx + 3)
            plt.hist(img_8bit.flatten(), bins=50, alpha=0.7, density=True)
            plt.xlabel('8-bit Intensity')
            plt.ylabel('Density')
            plt.title('8-bit Histogram')
            plt.xlim(0, 255)
            
            plot_idx += 4
            current_row += 1

        if 'resample' in self.preprocessing_results:
            add_row_title(fig, current_row, total_plots, '图像重采样')
            
            data = self.preprocessing_results['resample']
            plt.subplot(total_plots, 4, plot_idx)
            plt.imshow(data['original_image'], cmap='gray')
            plt.title(f'Original Image\n{data["original_image"].shape}')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 1)
            plt.imshow(data['processed_image'], cmap='gray')
            plt.title(f'Resampled Image\n{data["processed_image"].shape}')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 2)
            plt.imshow(data['original_mask'], cmap='gray')
            plt.title('Original Mask')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 3)
            plt.imshow(data['processed_mask'], cmap='gray')
            plt.title('Resampled Mask')
            plt.axis('off')
            
            plot_idx += 4
            current_row += 1

        if 'texture' in self.preprocessing_results:
            add_row_title(fig, current_row, total_plots, '纹理特征预处理')
            
            data = self.preprocessing_results['texture']
            mask = data['mask']
            
            plt.subplot(total_plots, 4, plot_idx)
            plt.imshow(data['original_image'], cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 1)
            plt.imshow(data['processed_image'], cmap='gray')
            plt.title('Z-Score Normalized\n& Discretized')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 2)
            plt.hist(data['original_image'][mask > 0].flatten(), bins=50, alpha=0.7, 
                    density=True, label='Original')
            plt.xlabel('Intensity')
            plt.ylabel('Density')
            plt.title('Original Distribution')
            plt.legend()
            
            plt.subplot(total_plots, 4, plot_idx + 3)
            plt.hist(data['processed_image'][mask > 0].flatten(), bins=20, alpha=0.7, 
                    density=True, label='Processed')
            plt.xlabel('Bin Level')
            plt.ylabel('Density')
            plt.title('Discretized Distribution')
            plt.legend()
            
            plot_idx += 4
            current_row += 1

        if 'signal' in self.preprocessing_results:
            add_row_title(fig, current_row, total_plots, '信号强度预处理')
            
            data = self.preprocessing_results['signal']
            
            plt.subplot(total_plots, 4, plot_idx)
            plt.imshow(data['original_image'], cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 1)
            plt.imshow(data['processed_image'], cmap='gray')
            plt.title('Resampled Only\n(Intensity Preserved)')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 2)
            plt.hist(data['original_image'].flatten(), bins=50, alpha=0.5, 
                    density=True, label='Original')
            plt.hist(data['processed_image'].flatten(), bins=50, alpha=0.5, 
                    density=True, label='Processed')
            plt.xlabel('Intensity')
            plt.ylabel('Density')
            plt.title('Intensity Distribution')
            plt.legend()
            
            plt.subplot(total_plots, 4, plot_idx + 3)
            if data['original_image'].shape == data['processed_image'].shape:
                sample_indices = np.random.choice(data['original_image'].size, 
                                                min(1000, data['original_image'].size), 
                                                replace=False)
                orig_flat = data['original_image'].flatten()[sample_indices]
                proc_flat = data['processed_image'].flatten()[sample_indices]
                plt.scatter(orig_flat, proc_flat, alpha=0.5, s=1)
                plt.xlabel('Original Intensity')
                plt.ylabel('Processed Intensity')
                plt.title('Intensity Correlation')
            
            plot_idx += 4
            current_row += 1
        
        if 'fractal' in self.preprocessing_results or self.current_slice is not None:
            add_row_title(fig, current_row, total_plots, '窗位窗宽调整')
            
            original_img = self.preprocessing_results.get('fractal', {}).get('original_image', self.current_slice)
            
            plt.subplot(total_plots, 4, plot_idx)
            plt.imshow(original_img, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 1)

            img_8bit = self.preprocessor.convert_to_8bit(original_img)
            windowed = self.preprocessor.apply_windowing(img_8bit, window_center=128, window_width=255)
            plt.imshow(windowed, cmap='gray')
            plt.title('Windowed\n(C:128, W:255)')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 2)

            windowed_narrow = self.preprocessor.apply_windowing(img_8bit, window_center=128, window_width=128)
            plt.imshow(windowed_narrow, cmap='gray')
            plt.title('Narrow Window\n(C:128, W:128)')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 3)

            plt.hist(original_img.flatten(), bins=50, alpha=0.5, label='Original', density=True)
            plt.hist(windowed.flatten(), bins=50, alpha=0.5, label='Windowed', density=True)
            plt.xlabel('Intensity')
            plt.ylabel('Density')
            plt.title('Windowing Effect')
            plt.legend()
            
            plot_idx += 4
            current_row += 1

        if 'fractal' in self.preprocessing_results:
            add_row_title(fig, current_row, total_plots, '分形维度预处理')
            
            data = self.preprocessing_results['fractal']
            
            plt.subplot(total_plots, 4, plot_idx)
            plt.imshow(data['original_image'], cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 1)
            img_8bit = self.preprocessor.convert_to_8bit(data['original_image'])
            windowed = self.preprocessor.apply_windowing(img_8bit, 128, 255)
            binary = self.preprocessor.binarize(windowed, method='percentile', percentile=65)
            if data['mask'] is not None:
                binary = binary * (data['mask'] > 0)
            plt.imshow(binary, cmap='gray')
            plt.title('Binary (65% threshold)')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 2)
            plt.imshow(data['edges'], cmap='gray')
            plt.title('Edge Detection\n(Canny)')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 3)
            overlay = np.zeros((*data['edges'].shape, 3))
            overlay[:, :, 0] = data['original_image'] / data['original_image'].max()
            overlay[:, :, 1] = data['edges'] / 255
            plt.imshow(overlay)
            plt.title('Edge Overlay')
            plt.axis('off')
            
            plot_idx += 4
            current_row += 1

        if 'shape' in self.preprocessing_results:
            add_row_title(fig, current_row, total_plots, '形状特征预处理')
            
            data = self.preprocessing_results['shape']
            
            plt.subplot(total_plots, 4, plot_idx)
            plt.imshow(data['original_mask'], cmap='gray')
            plt.title('Original Mask')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 1)
            plt.imshow(data['binary_mask'], cmap='gray')
            plt.title('Binary Mask')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 2)

            diff = np.abs(data['original_mask'].astype(float) - data['binary_mask'].astype(float))
            plt.imshow(diff, cmap='hot')
            plt.title('Difference Map')
            plt.axis('off')
            
            plt.subplot(total_plots, 4, plot_idx + 3)

            original_contours = cv2.Canny((data['original_mask'] > 0).astype(np.uint8) * 255, 50, 150)
            binary_contours = cv2.Canny((data['binary_mask'] > 0).astype(np.uint8) * 255, 50, 150)
            contour_overlay = np.zeros((*original_contours.shape, 3))
            contour_overlay[:, :, 0] = original_contours / 255 
            contour_overlay[:, :, 1] = binary_contours / 255 
            plt.imshow(contour_overlay)
            plt.title('Contour Comparison\n(Red:Orig, Green:Binary)')
            plt.axis('off')
            
            plot_idx += 4
            current_row += 1

        if 'log_filter' in self.preprocessing_results:
            add_row_title(fig, current_row, total_plots, 'LoG滤波器')
            
            data = self.preprocessing_results['log_filter']
            
            plt.subplot(total_plots, 4, plot_idx)
            plt.imshow(data['original_image'], cmap='gray')
            plt.title('Original Image')
            plt.axis('off')

            sigma_values = list(data['filtered_images'].keys())
            for i, sigma in enumerate(sigma_values[:3]): 
                plt.subplot(total_plots, 4, plot_idx + i + 1)
                filtered = data['filtered_images'][sigma]

                vmin, vmax = np.percentile(filtered, [2, 98])
                plt.imshow(filtered, cmap='gray', vmin=vmin, vmax=vmax)
                plt.title(f'LoG σ={sigma}')
                plt.axis('off')
                
            plot_idx += 4
            current_row += 1

        if 'wavelet' in self.preprocessing_results:
            add_row_title(fig, current_row, total_plots, '小波变换')
            
            data = self.preprocessing_results['wavelet']
            wavelet_images = data['wavelet_images']

            plt.subplot(total_plots, 4, plot_idx)
            plt.imshow(data['original_image'], cmap='gray')
            plt.title('Original Image')
            plt.axis('off')

            if 'LL' in wavelet_images:
                plt.subplot(total_plots, 4, plot_idx + 1)
                plt.imshow(wavelet_images['LL'], cmap='gray')
                plt.title('LL (Approximation)')
                plt.axis('off')

            if 'LH' in wavelet_images:
                plt.subplot(total_plots, 4, plot_idx + 2)

                lh_norm = (wavelet_images['LH'] - wavelet_images['LH'].min()) / (wavelet_images['LH'].max() - wavelet_images['LH'].min() + 1e-8)
                plt.imshow(lh_norm, cmap='gray')
                plt.title('LH (Horizontal)')
                plt.axis('off')
            
            if 'HL' in wavelet_images:
                plt.subplot(total_plots, 4, plot_idx + 3)
                hl_norm = (wavelet_images['HL'] - wavelet_images['HL'].min()) / (wavelet_images['HL'].max() - wavelet_images['HL'].min() + 1e-8)
                plt.imshow(hl_norm, cmap='gray')
                plt.title('HL (Vertical)')
                plt.axis('off')
                
            plot_idx += 4
            current_row += 1

            if 'HH' in wavelet_images and plot_idx <= total_plots * 4:
                plt.subplot(total_plots, 4, plot_idx)
                hh_norm = (wavelet_images['HH'] - wavelet_images['HH'].min()) / (wavelet_images['HH'].max() - wavelet_images['HH'].min() + 1e-8)
                plt.imshow(hh_norm, cmap='gray')
                plt.title('HH (Diagonal)')
                plt.axis('off')
                plot_idx += 1

        plt.tight_layout(rect=[0.05, 0, 0.99, 0.99])

        canvas_fig = FigureCanvasTkAgg(fig, scrollable_frame)
        canvas_fig.draw()
        canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def on_shift_mousewheel(event):
            canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.bind_all("<Shift-MouseWheel>", on_shift_mousewheel)

        scrollable_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        
    def _display_quantitative_results(self):

        for widget in self.quant_frame.winfo_children():
            widget.destroy()

        tree = ttk.Treeview(self.quant_frame, columns=('Value',), show='tree headings')
        tree.heading('#0', text='评估项目')
        tree.heading('Value', text='结果')

        scrollbar = ttk.Scrollbar(self.quant_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        for process_name, results in self.validation_results.items():
            parent = tree.insert('', 'end', text=self._get_process_display_name(process_name), 
                               values=('',))

            for key, value in results.items():
                if isinstance(value, dict):
                    sub_parent = tree.insert(parent, 'end', text=key, values=('',))
                    for sub_key, sub_value in value.items():
                        tree.insert(sub_parent, 'end', text=sub_key, 
                                  values=(self._format_value(sub_value),))
                else:
                    tree.insert(parent, 'end', text=key, values=(self._format_value(value),))

        for item in tree.get_children():
            tree.item(item, open=True)
            for child in tree.get_children(item):
                tree.item(child, open=True)
                
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
    def _generate_resample_report(self, results):

        self.report_text.insert(tk.END, f"原始形状: {results['original_shape']}\n")
        self.report_text.insert(tk.END, f"重采样后形状: {results['resampled_shape']}\n")
        self.report_text.insert(tk.END, f"尺寸变化: {results['size_change']}\n")

        if 'spacing_info' in results:
            spacing_info = results['spacing_info']
            self.report_text.insert(tk.END, f"原始间距: {spacing_info['original']}\n")
            self.report_text.insert(tk.END, f"实际间距: {spacing_info['actual']}\n")
        
        quality = results['interpolation_quality']
        self.report_text.insert(tk.END, f"\n插值质量评估:\n")
        self.report_text.insert(tk.END, f"  - 尺寸准确性: {'通过' if quality['size_accuracy'] else '失败'}\n")

        if 'expected_shape' in quality:
            self.report_text.insert(tk.END, f"  - 目标形状: {quality['expected_shape']}\n")
            self.report_text.insert(tk.END, f"  - 实际形状: {quality['actual_shape']}\n")
        
        self.report_text.insert(tk.END, f"  - 边缘相似度: {quality['edge_similarity']:.3f}\n")
        self.report_text.insert(tk.END, f"  - 质量等级: {quality['quality_score']}\n")

        if quality['quality_score'] == 'Poor':
            self.report_text.insert(tk.END, "\n⚠️ 建议: 重采样质量较差，可能需要调整插值方法\n")
            if not quality['size_accuracy']:
                self.report_text.insert(tk.END, "   - 重采样后的尺寸与预期不符，请检查设置\n")
        elif quality['quality_score'] == 'Fair':
            self.report_text.insert(tk.END, "\n💡 建议: 重采样质量尚可，但仍有改进空间\n")
        else:
            self.report_text.insert(tk.END, "\n✅ 评价: 重采样效果良好，图像已成功重采样到512×512\n")
            
    def _generate_texture_report(self, results):

        self.report_text.insert(tk.END, "Z-Score标准化效果:\n")
        self.report_text.insert(tk.END, f"  - 标准化后均值: {results['z_score_mean']:.3f} (期望: 0.0)\n")
        self.report_text.insert(tk.END, f"  - 标准化后标准差: {results['z_score_std']:.3f} (期望: 1.0)\n")
        
        mean_error = abs(results['z_score_mean'])
        std_error = abs(results['z_score_std'] - 1.0)

        self.report_text.insert(tk.END, f"\n评估结果:\n")
        self.report_text.insert(tk.END, f"  - 均值偏差: {mean_error:.3f} ")
        if mean_error < 0.05:
            self.report_text.insert(tk.END, "✅ 优秀\n")
        elif mean_error < 0.15:
            self.report_text.insert(tk.END, "⚠️ 良好\n")
        else:
            self.report_text.insert(tk.END, "❌ 需改进\n")
        
        self.report_text.insert(tk.END, f"  - 标准差偏差: {std_error:.3f} ")
        if std_error < 0.1:
            self.report_text.insert(tk.END, "✅ 优秀\n")
        elif std_error < 0.2:
            self.report_text.insert(tk.END, "⚠️ 良好\n")
        else:
            self.report_text.insert(tk.END, "❌ 需改进\n")

        if mean_error < 0.05 and std_error < 0.1:
            self.report_text.insert(tk.END, "\n✅ 总体评价: Z-Score标准化效果优秀，不同图像间可直接比较\n")
        elif mean_error < 0.15 and std_error < 0.2:
            self.report_text.insert(tk.END, "\n⚠️ 总体评价: Z-Score标准化效果良好，基本满足要求\n")
        else:
            self.report_text.insert(tk.END, "\n❌ 总体评价: Z-Score标准化偏差较大，需要检查是否有背景干扰\n")
            
        self.report_text.insert(tk.END, f"\n离散化信息:\n")
        self.report_text.insert(tk.END, f"  - 离散化级别数: {results['discretization_levels']}\n")
        self.report_text.insert(tk.END, f"  - Bin宽度: {results['bin_width']}\n")
        
    def _generate_signal_report(self, results):

        self.report_text.insert(tk.END, f"信号强度保持度: {results['intensity_preserved']:.3f}\n")
        self.report_text.insert(tk.END, f"形状变化: {results['shape_change']}\n")

        orig_range = results['pixel_value_range']['original']
        proc_range = results['pixel_value_range']['processed']
        self.report_text.insert(tk.END, f"\n像素值范围:\n")
        self.report_text.insert(tk.END, f"  - 原始: [{orig_range[0]:.1f}, {orig_range[1]:.1f}]\n")
        self.report_text.insert(tk.END, f"  - 处理后: [{proc_range[0]:.1f}, {proc_range[1]:.1f}]\n")
        
        self.report_text.insert(tk.END, f"\nROI统计信息:\n")
        orig_stats = results['original_stats']
        proc_stats = results['processed_stats']
        
        self.report_text.insert(tk.END, f"  原始图像:\n")
        self.report_text.insert(tk.END, f"    - 均值: {orig_stats['mean']:.2f}\n")
        self.report_text.insert(tk.END, f"    - 标准差: {orig_stats['std']:.2f}\n")
        self.report_text.insert(tk.END, f"    - 中位数: {orig_stats['median']:.2f}\n")
        
        self.report_text.insert(tk.END, f"  处理后图像:\n")
        self.report_text.insert(tk.END, f"    - 均值: {proc_stats['mean']:.2f}\n")
        self.report_text.insert(tk.END, f"    - 标准差: {proc_stats['std']:.2f}\n")
        self.report_text.insert(tk.END, f"    - 中位数: {proc_stats['median']:.2f}\n")
        
        if results['intensity_preserved'] > 0.95:
            self.report_text.insert(tk.END, "\n✅ 评价: 信号强度保持良好，仅进行了空间重采样\n")
        else:
            self.report_text.insert(tk.END, "\n💡 注意: 信号强度有一定变化，可能是由于插值造成\n")
            
    def _generate_fractal_report(self, results):

        self.report_text.insert(tk.END, f"边缘检测结果:\n")
        self.report_text.insert(tk.END, f"  - 边缘像素数: {results['edge_pixels']}\n")
        self.report_text.insert(tk.END, f"  - 总像素数: {results['total_pixels']}\n")
        self.report_text.insert(tk.END, f"  - 边缘比例: {results['edge_ratio']:.3f}\n")
        
        continuity = results['edge_continuity']
        self.report_text.insert(tk.END, f"\n边缘连续性评估:\n")
        self.report_text.insert(tk.END, f"  - 连通组件数: {continuity['num_components']}\n")
        self.report_text.insert(tk.END, f"  - 最大组件占比: {continuity['largest_component_ratio']:.3f}\n")
        self.report_text.insert(tk.END, f"  - 连续性等级: {continuity['continuity_score']}\n")
        
        if continuity['continuity_score'] == 'Poor':
            self.report_text.insert(tk.END, "\n⚠️ 建议: 边缘过于破碎，可能需要调整阈值或使用形态学操作\n")
        elif continuity['continuity_score'] == 'Good':
            self.report_text.insert(tk.END, "\n✅ 评价: 边缘连续性良好，适合分形维度计算\n")
            
    def _generate_shape_report(self, results):

        self.report_text.insert(tk.END, f"形状保持度:\n")
        self.report_text.insert(tk.END, f"  - 原始面积: {results['original_area']} 像素\n")
        self.report_text.insert(tk.END, f"  - 处理后面积: {results['processed_area']} 像素\n")
        self.report_text.insert(tk.END, f"  - 面积保持率: {results['area_preservation']:.3f}\n")
        self.report_text.insert(tk.END, f"  - Dice系数: {results['shape_similarity']:.3f}\n")
        
        if results['shape_similarity'] > 0.95:
            self.report_text.insert(tk.END, "\n✅ 评价: 形状保持极好\n")
        elif results['shape_similarity'] > 0.90:
            self.report_text.insert(tk.END, "\n💡 评价: 形状保持良好\n")
        else:
            self.report_text.insert(tk.END, "\n⚠️ 警告: 形状变化较大\n")
            
    def _generate_log_filter_report(self, results):

        self.report_text.insert(tk.END, f"LoG滤波器参数:\n")
        self.report_text.insert(tk.END, f"  - 尺度数: {results['num_scales']}\n")
        self.report_text.insert(tk.END, f"  - Sigma值: {results['sigma_values']}\n")
        
        self.report_text.insert(tk.END, "\n各尺度响应统计:\n")
        for sigma, stats in results['response_stats'].items():
            self.report_text.insert(tk.END, f"  {sigma}: 均值={stats['mean']:.3f}, "
                                         f"标准差={stats['std']:.3f}\n")
            
    def _generate_wavelet_report(self, results):

        self.report_text.insert(tk.END, f"小波分解:\n")
        self.report_text.insert(tk.END, f"  - 组件数: {results['num_components']}\n")
        self.report_text.insert(tk.END, f"  - 组件列表: {', '.join(results['components'])}\n")
        
        self.report_text.insert(tk.END, "\n能量分布:\n")
        total_energy = sum(results['energy_distribution'].values())
        for comp, energy in results['energy_distribution'].items():
            ratio = energy / total_energy if total_energy > 0 else 0
            self.report_text.insert(tk.END, f"  - {comp}: {ratio:.1%}\n")
            
    def _generate_overall_assessment(self):

        total_tests = len(self.validation_results)
        good_results = 0
        warnings = []

        if 'resample' in self.validation_results:
            result = self.validation_results['resample']
            if result['interpolation_quality'].get('is_full_image', False):
                if result['interpolation_quality']['size_accuracy']:
                    good_results += 1
            elif result['interpolation_quality']['quality_score'] == 'Good':
                good_results += 1
            else:
                warnings.append("重采样质量需要改进")
                
        if 'texture' in self.validation_results:
            mean_error = abs(self.validation_results['texture']['z_score_mean'])
            std_error = abs(self.validation_results['texture']['z_score_std'] - 1.0)
            if mean_error < 0.2 and std_error < 0.2:
                good_results += 1
            else:
                warnings.append("Z-Score标准化效果不理想")
                
        if 'signal' in self.validation_results:
            if self.validation_results['signal']['intensity_preserved'] > 0.95:
                good_results += 1
            else:
                warnings.append("信号强度保持度较低")
                
        if 'fractal' in self.validation_results:
            if self.validation_results['fractal']['edge_continuity']['continuity_score'] != 'Poor':
                good_results += 1
            else:
                warnings.append("边缘检测连续性较差")

        if 'log_filter' in self.validation_results:
            good_results += 1 
            
        if 'wavelet' in self.validation_results:
            good_results += 1  

        score = good_results / total_tests if total_tests > 0 else 0
        
        self.report_text.insert(tk.END, f"\n验证项目数: {total_tests}\n")
        self.report_text.insert(tk.END, f"通过项目数: {good_results}\n")
        self.report_text.insert(tk.END, f"总体得分: {score:.1%}\n\n")
        
        if score >= 0.8:
            self.report_text.insert(tk.END, "✅ 总体评价: 预处理效果优秀，可以用于特征提取\n")
        elif score >= 0.6:
            self.report_text.insert(tk.END, "💡 总体评价: 预处理效果良好，但仍有改进空间\n")
        else:
            self.report_text.insert(tk.END, "⚠️ 总体评价: 预处理效果需要改进\n")
            
        if warnings:
            self.report_text.insert(tk.END, "\n需要注意的问题:\n")
            for warning in warnings:
                self.report_text.insert(tk.END, f"  - {warning}\n")

        self.report_text.insert(tk.END, "\n建议:\n")
        self.report_text.insert(tk.END, "1. 定期使用本工具验证预处理效果\n")
        self.report_text.insert(tk.END, "2. 针对不同的特征类型选择合适的预处理方法\n")
        self.report_text.insert(tk.END, "3. 保存验证通过的参数配置用于批量处理\n")

        if any(r.get('interpolation_quality', {}).get('is_full_image', False) 
            for r in self.validation_results.values()):
            self.report_text.insert(tk.END, "\n注意：当前为全图分析模式，边缘相似度评估不适用。\n")
            self.report_text.insert(tk.END, "建议使用实际的ROI掩模进行更准确的验证。\n")

    def _generate_zscore_validation_report(self):

        if 'texture' not in self.validation_results:
            return
            
        texture_result = self.validation_results['texture']
        processed_stats = texture_result['processed_stats']

        expected_mean = 0.0
        expected_std = 1.0

        actual_mean = processed_stats['mean']
        actual_std = processed_stats['std']

        mean_error = abs(actual_mean - expected_mean)
        std_error = abs(actual_std - expected_std)
        
        self.report_text.insert(tk.END, "【Z-score标准化质量指标】\n")
        self.report_text.insert(tk.END, "-"*40 + "\n")
        self.report_text.insert(tk.END, f"期望均值: {expected_mean:.3f}, 实际均值: {actual_mean:.3f}\n")
        self.report_text.insert(tk.END, f"期望标准差: {expected_std:.3f}, 实际标准差: {actual_std:.3f}\n")
        self.report_text.insert(tk.END, f"均值偏差: {mean_error:.3f}\n")
        self.report_text.insert(tk.END, f"标准差偏差: {std_error:.3f}\n\n")

        if mean_error < 0.05 and std_error < 0.1:
            quality_grade = "★★★ 优秀"
            quality_desc = "标准化效果非常理想，完全满足深度学习训练要求"
        elif mean_error < 0.15 and std_error < 0.2:
            quality_grade = "★★☆ 良好"
            quality_desc = "标准化效果良好，适合深度学习训练使用"
        else:
            quality_grade = "★☆☆ 需改进"
            quality_desc = "标准化效果有偏差，建议调整参数后重新处理"
        
        self.report_text.insert(tk.END, f"质量评级: {quality_grade}\n")
        self.report_text.insert(tk.END, f"评价: {quality_desc}\n\n")

        if 'skewness' in processed_stats:
            skewness = processed_stats.get('skewness', 0)
            kurtosis = processed_stats.get('kurtosis', 0)
            
            self.report_text.insert(tk.END, "【分布形状参数】\n")
            self.report_text.insert(tk.END, f"偏度 (Skewness): {skewness:.3f}\n")
            self.report_text.insert(tk.END, f"  - 说明: 0表示完全对称，正值表示右偏，负值表示左偏\n")
            self.report_text.insert(tk.END, f"峰度 (Kurtosis): {kurtosis:.3f}\n")
            self.report_text.insert(tk.END, f"  - 说明: 0表示正态分布，正值表示尖峰，负值表示平坦\n\n")

        self.report_text.insert(tk.END, "【标准化建议】\n")
        if quality_grade.startswith("★★★") or quality_grade.startswith("★★☆"):
            self.report_text.insert(tk.END, "✓ 标准化质量合格，可以用于跨图像的一致性分析\n")
            self.report_text.insert(tk.END, "✓ 不同图像经过相同的标准化处理后，信号强度将具有可比性\n")
        else:
            self.report_text.insert(tk.END, "⚠ 标准化质量需要改进\n")
            if mean_error > 0.1:
                self.report_text.insert(tk.END, "  - 均值偏差较大，建议检查背景排除方法\n")
            if std_error > 0.15:
                self.report_text.insert(tk.END, "  - 标准差偏差较大，建议调整离群值处理参数\n")
        
    def _get_process_display_name(self, process_name):

        name_map = {
            'resample': '图像重采样',
            'texture': '纹理特征预处理',
            'signal': '信号强度预处理',
            'fractal': '分形维度预处理',
            'shape': '形状特征预处理',
            'log_filter': 'LoG滤波器',
            'wavelet': '小波变换'
        }
        return name_map.get(process_name, process_name)
        
    def _format_value(self, value):

        if isinstance(value, float):
            return f"{value:.4f}"
        elif isinstance(value, list):
            return ', '.join(str(v) for v in value)
        elif isinstance(value, dict):
            parts = []
            for k, v in value.items():
                if isinstance(v, list):
                    parts.append(f"{k}: {', '.join(str(x) for x in v)}")
                else:
                    parts.append(f"{k}: {v}")
            return '; '.join(parts)
        else:
            return str(value)
        
    def _generate_report(self):

        self.report_text.delete(1.0, tk.END)

        self.report_text.insert(tk.END, "="*60 + "\n")
        self.report_text.insert(tk.END, "MRI预处理效果验证报告\n")
        self.report_text.insert(tk.END, "="*60 + "\n\n")

        self.report_text.insert(tk.END, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.report_text.insert(tk.END, f"图像路径: {self.image_path_var.get()}\n")
        if self.use_mask_var.get() and self.mask_path_var.get():
            self.report_text.insert(tk.END, f"掩模路径: {self.mask_path_var.get()}\n")
        else:
            self.report_text.insert(tk.END, f"掩模状态: {'未使用掩模' if not self.use_mask_var.get() else '未加载掩模文件'}\n")
        self.report_text.insert(tk.END, f"原始图像形状: {self.current_image.shape}\n")
        self.report_text.insert(tk.END, f"原始间距: {[f'{s:.2f}' for s in self.current_spacing]}\n\n")

        self.report_text.insert(tk.END, "预处理步骤评估标准:\n")
        self.report_text.insert(tk.END, "-"*40 + "\n")
        self.report_text.insert(tk.END, "1. 图像重采样\n")
        self.report_text.insert(tk.END, "   • 评估指标：尺寸准确性、边缘相似度\n")
        self.report_text.insert(tk.END, "   • 优秀标准：边缘相似度>0.8，尺寸完全匹配\n")
        self.report_text.insert(tk.END, "   • 用途：统一图像尺寸，便于批量处理\n\n")
        
        self.report_text.insert(tk.END, "2. 纹理特征预处理（含Z-score标准化）*\n")
        self.report_text.insert(tk.END, "   • 评估指标：标准化后均值偏差、标准差偏差\n")
        self.report_text.insert(tk.END, "   • 优秀标准：均值偏差<0.05，标准差偏差<0.1\n")
        self.report_text.insert(tk.END, "   • 用途：确保不同图像间的纹理特征可比性\n\n")
        
        self.report_text.insert(tk.END, "3. 信号强度预处理\n")
        self.report_text.insert(tk.END, "   • 评估指标：强度保持度（相关系数）\n")
        self.report_text.insert(tk.END, "   • 优秀标准：相关系数>0.95\n")
        self.report_text.insert(tk.END, "   • 用途：保持原始强度值，用于ASI/T2SI计算\n\n")
        
        self.report_text.insert(tk.END, "4. 分形维度预处理*\n")
        self.report_text.insert(tk.END, "   • 评估指标：边缘连续性、最大连通组件占比\n")
        self.report_text.insert(tk.END, "   • 优秀标准：最大连通组件>80%\n")
        self.report_text.insert(tk.END, "   • 用途：生成适合分形维度计算的边缘图像\n\n")
        
        self.report_text.insert(tk.END, "5. 形状特征预处理*\n")
        self.report_text.insert(tk.END, "   • 评估指标：Dice系数、面积保持率\n")
        self.report_text.insert(tk.END, "   • 优秀标准：Dice系数>0.95\n")
        self.report_text.insert(tk.END, "   • 用途：生成二值掩模用于形状特征计算\n\n")
        
        self.report_text.insert(tk.END, "注：标*的步骤需要掩模\n\n")

        for process_name, results in self.validation_results.items():
            self.report_text.insert(tk.END, "-"*40 + "\n")
            self.report_text.insert(tk.END, f"{self._get_process_display_name(process_name)}\n")
            self.report_text.insert(tk.END, "-"*40 + "\n")

            if process_name == 'resample':
                self._generate_resample_report(results)
            elif process_name == 'texture':
                self._generate_texture_report(results)
            elif process_name == 'signal':
                self._generate_signal_report(results)
            elif process_name == 'fractal':
                self._generate_fractal_report(results)
            elif process_name == 'shape':
                self._generate_shape_report(results)
            elif process_name == 'log_filter':
                self._generate_log_filter_report(results)
            elif process_name == 'wavelet':
                self._generate_wavelet_report(results)
                
            self.report_text.insert(tk.END, "\n")

        self.report_text.insert(tk.END, "="*60 + "\n")
        self.report_text.insert(tk.END, "总体评估\n")
        self.report_text.insert(tk.END, "="*60 + "\n")
        self._generate_overall_assessment()

        if 'texture' in self.validation_results:
            self.report_text.insert(tk.END, "\n")
            self.report_text.insert(tk.END, "="*60 + "\n")
            self.report_text.insert(tk.END, "Z-score标准化效果验证\n")
            self.report_text.insert(tk.END, "="*60 + "\n")
            self._generate_zscore_validation_report()

    def _generate_zscore_validation_report(self):

        if 'texture' not in self.validation_results:
            return
            
        texture_result = self.validation_results['texture']
        processed_stats = texture_result['processed_stats']

        expected_mean = 0.0
        expected_std = 1.0

        actual_mean = processed_stats['mean']
        actual_std = processed_stats['std']

        mean_error = abs(actual_mean - expected_mean)
        std_error = abs(actual_std - expected_std)
        
        self.report_text.insert(tk.END, "【Z-score标准化质量指标】\n")
        self.report_text.insert(tk.END, "-"*40 + "\n")
        self.report_text.insert(tk.END, f"期望均值: {expected_mean:.3f}, 实际均值: {actual_mean:.3f}\n")
        self.report_text.insert(tk.END, f"期望标准差: {expected_std:.3f}, 实际标准差: {actual_std:.3f}\n")
        self.report_text.insert(tk.END, f"均值偏差: {mean_error:.3f}\n")
        self.report_text.insert(tk.END, f"标准差偏差: {std_error:.3f}\n\n")

        if mean_error < 0.05 and std_error < 0.1:
            quality_grade = "★★★ 优秀"
            quality_desc = "标准化效果非常理想，完全满足深度学习训练要求"
        elif mean_error < 0.15 and std_error < 0.2:
            quality_grade = "★★☆ 良好"
            quality_desc = "标准化效果良好，适合深度学习训练使用"
        else:
            quality_grade = "★☆☆ 需改进"
            quality_desc = "标准化效果有偏差，建议调整参数后重新处理"
        
        self.report_text.insert(tk.END, f"质量评级: {quality_grade}\n")
        self.report_text.insert(tk.END, f"评价: {quality_desc}\n\n")

        if 'skewness' in processed_stats:
            skewness = processed_stats.get('skewness', 0)
            kurtosis = processed_stats.get('kurtosis', 0)
            
            self.report_text.insert(tk.END, "【分布形状参数】\n")
            self.report_text.insert(tk.END, f"偏度 (Skewness): {skewness:.3f}\n")
            self.report_text.insert(tk.END, f"  - 说明: 0表示完全对称，正值表示右偏，负值表示左偏\n")
            self.report_text.insert(tk.END, f"峰度 (Kurtosis): {kurtosis:.3f}\n")
            self.report_text.insert(tk.END, f"  - 说明: 0表示正态分布，正值表示尖峰，负值表示平坦\n\n")

        self.report_text.insert(tk.END, "【标准化建议】\n")
        if quality_grade.startswith("★★★") or quality_grade.startswith("★★☆"):
            self.report_text.insert(tk.END, "✓ 标准化质量合格，可以用于跨图像的一致性分析\n")
            self.report_text.insert(tk.END, "✓ 不同图像经过相同的标准化处理后，信号强度将具有可比性\n")
        else:
            self.report_text.insert(tk.END, "⚠ 标准化质量需要改进\n")
            if mean_error > 0.1:
                self.report_text.insert(tk.END, "  - 均值偏差较大，建议检查背景排除方法\n")
            if std_error > 0.15:
                self.report_text.insert(tk.END, "  - 标准差偏差较大，建议调整离群值处理参数\n")
            

def main():
    root = tk.Tk()
    app = PreprocessingValidator(root)
    root.mainloop()


if __name__ == "__main__":
    main()