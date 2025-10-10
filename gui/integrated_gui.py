import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
import pandas as pd
import SimpleITK as sitk
import numpy as np
import logging
from pathlib import Path
import json
import yaml
from datetime import datetime
import ctypes
from typing import List, Dict, Optional, Tuple, Any
import multiprocessing as mp
from .perturbation_gui import PerturbationGUI, PerturbationWorker
from .robustness_gui import RobustnessGUI

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculator import (
    DHICalculator, ASICalculator, FractalDimensionCalculator,
    T2SignalIntensityCalculator, GaborCalculator, HuMomentsCalculator,
    TextureFeaturesCalculator, DSCRCalculator
)
from utils import ImageIO, Preprocessor
from config import Config

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

PYRADIOMICS_AVAILABLE = False
PYRADIOMICS_ERROR = None

try:
    import numpy
    if numpy.__version__.startswith('2.'):
        PYRADIOMICS_ERROR = "NumPy 2.x不兼容。请运行: pip install 'numpy<2.0'"
    else:
        from radiomics import featureextractor
        import radiomics
        PYRADIOMICS_AVAILABLE = True
except ImportError as e:
    PYRADIOMICS_ERROR = f"导入失败: {str(e)}\n请安装依赖：pip install 'numpy<2.0' pyradiomics"
except Exception as e:
    PYRADIOMICS_ERROR = f"PyRadiomics加载错误: {str(e)}"

LANG_DICT = {
    'cn': {
        'title': '椎间盘退变分析系统',
        'other_features': '其他特征',
        'enable_dhi': '椎间盘高度指数 (DHI)',
        'enable_asi': '峰值信号强度差 (ASI)',
        'enable_fd': '分形维度 (FD)',
        'enable_t2si': 'T2信号强度',
        'enable_gabor': 'Gabor纹理特征',
        'enable_hu': 'Hu不变矩',
        'enable_texture': '扩展纹理特征',
        'enable_dscr': '椎管狭窄率 (DSCR)',
        'dural_sac_label': '硬脊膜囊标签值:',
        'other_feature_settings': '其他特征设置',
        'dural_sac_label': '椎管/CSF标签值:',
        'processing_other': '🔄 处理其他特征...',
        'other_complete': '✅ 其他特征提取完成',
        'feature_type': '特征类型:',
        'pyradiomics_features': 'PyRadiomics特征',
        'other_features_option': '其他特征',
        'both_features': '两者都提取',
        'file_selection': '📁 文件选择',
        'process_mode': '处理模式:',
        'batch_mode': '📊 批量处理',
        'single_mode': '🔍 单个案例',
        'input_path': '输入路径:',
        'mask_path': '掩码路径:',
        'output_path': '输出路径:',
        'select': '选择',
        'basic_settings': '🔧 基本设置',
        'parameter_settings': '参数设置',
        'bin_width': '分箱宽度:',
        'bin_count': '分箱数量:',
        'resample_spacing': '重采样间距:',
        'interpolator': '插值方法:',
        'normalize': '标准化强度',
        'scale': '尺度:',
        'remove_outliers': '移除离群值(nσ):',
        'correct_mask': '自动校正掩码',
        'label': '标签值:',
        'feature_classes': '特征类别',
        'shape_3d': '形状特征 (3D)',
        'shape_2d': '形状特征 (2D)',
        'firstorder': '一阶统计特征',
        'glcm': '灰度共生矩阵 (GLCM)',
        'glrlm': '灰度游程矩阵 (GLRLM)',
        'glszm': '灰度大小区域矩阵 (GLSZM)',
        'gldm': '灰度依赖矩阵 (GLDM)',
        'ngtdm': '邻域灰度差分矩阵 (NGTDM)',
        'advanced_params': '高级参数',
        'pad_distance': '边距填充:',
        'geometry_tolerance': '几何容差:',
        'min_roi_dimensions': '最小ROI维度:',
        'min_roi_size': '最小ROI大小:',
        'additional_info': '包含诊断信息',
        'enable_c_extensions': '启用C扩展',
        'filter_settings': '🔧 滤波器设置',
        'log_filter': 'LoG滤波器',
        'enable_log': '启用LoG滤波器',
        'sigma_values': 'Sigma值:',
        'wavelet_filter': '小波滤波器',
        'enable_wavelet': '启用小波滤波器',
        'wavelet_type': '小波类型:',
        'decomposition_level': '分解层级:',
        'start_level': '起始层级:',
        'simple_filters': '简单滤波器',
        'square': '平方',
        'square_root': '平方根',
        'logarithm': '对数',
        'exponential': '指数',
        'gradient_filter': '梯度滤波器',
        'enable_gradient': '启用梯度滤波器',
        'use_spacing': '使用间距计算',
        'lbp_filter': 'LBP滤波器',
        'enable_lbp2d': '启用LBP 2D',
        'radius': '半径:',
        'samples': '采样数:',
        'enable_lbp3d': '启用LBP 3D',
        'levels': '层级:',
        'advanced_settings': '🔧 高级设置',
        'resegmentation_settings': '重分割设置',
        'resegment_range': '重分割范围:',
        'resegment_mode': '重分割模式:',
        'resegment_shape': '重分割形状计算',
        '2d_settings': '2D设置',
        'force_2d': '强制2D提取',
        '2d_dimension': '2D维度:',
        'force2d_aggregator': '2D聚合方式:',
        'aggregator_mean': '平均值',
        'aggregator_max': '最大值',
        'aggregator_min': '最小值',
        'aggregator_std': '标准差',
        'aggregator_sum': '求和',
        'texture_matrix_settings': '纹理矩阵设置',
        'weighting_norm': '加权范数:',
        'distances': '距离值:',
        'symmetrical_glcm': '对称GLCM',
        'gldm_alpha': 'GLDM α值:',
        'other_settings': '其他设置',
        'voxel_array_shift': '体素数组偏移:',
        'pre_crop': '预裁剪',
        'voxel_settings': '🔧 体素级设置',
        'voxel_based_settings': '体素级别设置',
        'kernel_radius': '核半径:',
        'masked_kernel': '掩码核',
        'init_value': '初始值:',
        'voxel_batch': '体素批次:',
        'parameter_management': '参数管理',
        'save_params': '💾 保存参数',
        'load_params': '📂 加载参数',
        'reset_defaults': '🔄 重置为默认',
        'execution_control': '执行控制',
        'start_extraction': '🚀 开始提取',
        'stop': '⏹ 停止',
        'run_log': '📝 运行日志',
        'language': '语言',
        'chinese': '中文',
        'english': 'English',
        'select_input_file': '选择图像文件',
        'select_batch_csv': '选择批处理CSV文件',
        'select_mask_file': '选择掩码文件',
        'save_results': '保存结果',
        'error': '错误',
        'warning': '警告',
        'info': '提示',
        'parallel_settings': '并行处理设置',
        'enable_parallel': '启用并行处理',
        'worker_processes': '工作进程数:',
        'cpu_cores': 'CPU核心数:',
        'welcome_msg': """
🎯 椎间盘退变特征提取系统已就绪！
        """,
        'example_range': '(例: -50,100)',
        'axial_coronal_sagittal': '(0=轴向, 1=冠状, 2=矢状)',
        'negative_nan_note': '(负值返回原图, nan透明)',
        'negative_all_note': '(负值处理所有体素)'
    }
}

class IntegratedFeatureExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(LANG_DICT['cn']['title'])
        self.root.geometry("1920x1080")

        self.config = Config()

        self.image_io = ImageIO()
        self.preprocessor = Preprocessor()
        
        self.dhi_calculator = DHICalculator(**self.config.DHI_PARAMS)
        self.asi_calculator = ASICalculator(**self.config.ASI_PARAMS)
        self.fd_calculator = FractalDimensionCalculator(**self.config.FD_PARAMS)
        self.t2si_calculator = T2SignalIntensityCalculator(**self.config.T2SI_PARAMS)
        self.gabor_calculator = GaborCalculator(**self.config.GABOR_PARAMS)
        self.hu_calculator = HuMomentsCalculator(**self.config.HU_MOMENTS_PARAMS)
        self.texture_calculator = TextureFeaturesCalculator(**self.config.TEXTURE_PARAMS)
        self.dscr_calculator = DSCRCalculator(**self.config.DSCR_PARAMS)

        import queue
        import threading
        self.log_queue = queue.Queue()
        self._stop_log_processing = False

        style = ttk.Style()
        try:
            available_themes = style.theme_names()
            if 'vista' in available_themes:
                style.theme_use('vista')
            elif 'xpnative' in available_themes:
                style.theme_use('xpnative')
            else:
                style.theme_use('clam')
        except:
            style.theme_use('clam')

        style.configure('TCheckbutton', 
                        focuscolor='none',
                        focusthickness=1,
                        indicatorbackground='black',
                        indicatorforeground='black',
                        background='#f0f0f0',
                        foreground='black',
                        borderwidth=1)

        style.map('TCheckbutton',
                foreground=[('focus', 'black'),
                            ('selected', 'black'),
                            ('active', 'black'),
                            ('pressed', 'black'),
                            ('disabled', '#999999')],
                background=[('focus', '#f0f0f0'),
                            ('active', '#e8e8e8'),
                            ('pressed', '#d0d0d0'),
                            ('disabled', '#f5f5f5')],
                focuscolor=[('focus', 'black')],
                relief=[('focus', 'solid'),
                        ('!focus', 'flat')])

        style.configure('TRadiobutton',
                        focuscolor='none',
                        focusthickness=1,
                        indicatorbackground='black',
                        indicatorforeground='black',
                        background='#f0f0f0',
                        foreground='black',
                        borderwidth=1)

        style.map('TRadiobutton',
                foreground=[('focus', 'black'),
                            ('selected', 'black'),
                            ('active', 'black'),
                            ('pressed', 'black'),
                            ('disabled', '#999999')],
                background=[('focus', '#f0f0f0'),
                            ('active', '#e8e8e8'),
                            ('pressed', '#d0d0d0'),
                            ('disabled', '#f5f5f5')],
                focuscolor=[('focus', 'black')],
                relief=[('focus', 'solid'),
                        ('!focus', 'flat')])

        style.configure('TLabel', font=('Segoe UI', 9))
        style.configure('TButton', font=('Segoe UI', 9))
        style.configure('TCheckbutton', font=('Segoe UI', 9))
        style.configure('Title.TLabel', font=('Segoe UI', 20, 'bold'))
        style.configure('Heading.TLabel', font=('Segoe UI', 11, 'bold'))

        self.current_lang = tk.StringVar(value="cn")
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.mask_path = tk.StringVar()
        self.input_type = tk.StringVar(value="batch")

        self.feature_type = tk.StringVar(value="both")

        self.enable_other_dhi = tk.BooleanVar(value=True)
        self.enable_other_asi = tk.BooleanVar(value=True)
        self.enable_other_fd = tk.BooleanVar(value=True)
        self.enable_other_t2si = tk.BooleanVar(value=True) 
        self.enable_other_gabor = tk.BooleanVar(value=True)  
        self.enable_other_hu = tk.BooleanVar(value=True)     
        self.enable_other_texture = tk.BooleanVar(value=True)
        self.enable_other_dscr = tk.BooleanVar(value=True) 
        self.dural_sac_label = tk.IntVar(value=20)


        self._init_pyradiomics_variables()

        self.widgets = {}

        self.setup_gui()

        self._start_log_processor()

        if not PYRADIOMICS_AVAILABLE and PYRADIOMICS_ERROR:
            messagebox.showwarning("PyRadiomics不可用", 
                                 f"PyRadiomics功能将被禁用:\n\n{PYRADIOMICS_ERROR}\n\n仍可以使用自定义特征提取功能。")
            
    def _start_log_processor(self):
        self._process_log_queue()
        
    def _process_log_queue(self):
        if self._stop_log_processing:
            return
        
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Log processing error: {e}")
        
        self.root.after(100, self._process_log_queue)
    
    def _init_pyradiomics_variables(self):
        self.bin_width = tk.DoubleVar(value=16)
        self.bin_count = tk.IntVar(value=0)
        self.interpolator = tk.StringVar(value="sitkBSpline")
        self.resample_spacing = tk.StringVar(value="0.5,0.5,0")
        self.normalize = tk.BooleanVar(value=False)
        self.normalize_scale = tk.DoubleVar(value=1.0)
        self.remove_outliers = tk.DoubleVar(value=0.0)
        self.correct_mask = tk.BooleanVar(value=False)
        self.label = tk.IntVar(value=3)
        self.pad_distance = tk.IntVar(value=5)
        self.geometry_tolerance = tk.StringVar(value="1e-16")
        self.additional_info = tk.BooleanVar(value=True)
        self.enable_c_extensions = tk.BooleanVar(value=True)
        self.minimum_roi_dimensions = tk.IntVar(value=2)
        self.minimum_roi_size = tk.IntVar(value=50)
        self.preCrop = tk.BooleanVar(value=False)
        self.voxel_array_shift = tk.IntVar(value=0)
        self.force2D = tk.BooleanVar(value=False)
        self.force2D_dimension = tk.IntVar(value=0)
        self.force2D_aggregator = tk.StringVar(value="mean")
        self.distances = tk.StringVar(value="1")

        self.resegment_range = tk.StringVar(value="")
        self.resegment_mode = tk.StringVar(value="absolute")
        self.resegment_shape = tk.BooleanVar(value=False)
        self.weighting_norm = tk.StringVar(value="no_weighting")
        self.symmetrical_glcm = tk.BooleanVar(value=True)
        self.gldm_a = tk.DoubleVar(value=0.0)

        self.kernel_radius = tk.IntVar(value=1)
        self.masked_kernel = tk.BooleanVar(value=True)
        self.init_value = tk.StringVar(value="0")
        self.voxel_batch = tk.IntVar(value=-1)

        self.enable_log = tk.BooleanVar(value=True)
        self.log_sigma = tk.StringVar(value="1.0,3.0,5.0")
        self.enable_wavelet = tk.BooleanVar(value=True)
        self.wavelet_level = tk.IntVar(value=1)
        self.wavelet_start_level = tk.IntVar(value=0)
        self.wavelet_type = tk.StringVar(value="db1")
        self.enable_square = tk.BooleanVar(value=False)
        self.enable_square_root = tk.BooleanVar(value=False)
        self.enable_logarithm = tk.BooleanVar(value=False)
        self.enable_exponential = tk.BooleanVar(value=False)
        self.enable_gradient = tk.BooleanVar(value=True)
        self.gradient_sigma = tk.BooleanVar(value=True)
        self.enable_lbp2d = tk.BooleanVar(value=False)
        self.lbp2d_radius = tk.DoubleVar(value=1.0)
        self.lbp2d_samples = tk.IntVar(value=9)
        self.lbp2d_method = tk.StringVar(value="uniform")
        self.enable_lbp3d = tk.BooleanVar(value=False)
        self.lbp3d_levels = tk.IntVar(value=2)
        self.lbp3d_icosphere_radius = tk.DoubleVar(value=1.0)
        self.lbp3d_icosphere_subdivision = tk.IntVar(value=1)

        self.enable_shape = tk.BooleanVar(value=True)
        self.enable_shape2d = tk.BooleanVar(value=True)
        self.enable_firstorder = tk.BooleanVar(value=True)
        self.enable_glcm = tk.BooleanVar(value=True)
        self.enable_glrlm = tk.BooleanVar(value=True)
        self.enable_glszm = tk.BooleanVar(value=True)
        self.enable_gldm = tk.BooleanVar(value=True)
        self.enable_ngtdm = tk.BooleanVar(value=True)

    def get_text(self, key):
        lang = self.current_lang.get()
        return LANG_DICT[lang].get(key, key)
    
    def update_language(self):
        self.root.title(self.get_text('title'))

        if hasattr(self, 'widgets'):
            if 'main_title_label' in self.widgets:
                if self.current_lang.get() == "cn":
                    self.widgets['main_title_label'].config(text="椎间盘退变分析系统")
                else:
                    self.widgets['main_title_label'].config(text="IVD Degeneration Analysis System")

            if 'title_label' in self.widgets:
                self.widgets['title_label'].config(text=self.get_text('title'))

            if 'file_frame' in self.widgets:
                self.widgets['file_frame'].config(text=self.get_text('file_selection'))
            if 'type_label' in self.widgets:
                self.widgets['type_label'].config(text=self.get_text('process_mode'))
            if 'batch_radio' in self.widgets:
                self.widgets['batch_radio'].config(text=self.get_text('batch_mode'))
            if 'single_radio' in self.widgets:
                self.widgets['single_radio'].config(text=self.get_text('single_mode'))
            if 'input_label' in self.widgets:
                self.widgets['input_label'].config(text=self.get_text('input_path'))
            if 'mask_label' in self.widgets:
                self.widgets['mask_label'].config(text=self.get_text('mask_path'))
            if 'output_label' in self.widgets:
                self.widgets['output_label'].config(text=self.get_text('output_path'))
            if 'input_btn' in self.widgets:
                self.widgets['input_btn'].config(text="📂 " + self.get_text('select'))
            if 'mask_btn' in self.widgets:
                self.widgets['mask_btn'].config(text="🎯 " + self.get_text('select'))
            if 'output_btn' in self.widgets:
                self.widgets['output_btn'].config(text="💾 " + self.get_text('select'))

            if 'feature_type_frame' in self.widgets:
                self.widgets['feature_type_frame'].config(text=self.get_text('feature_type'))
            if 'pyrad_radio' in self.widgets:
                self.widgets['pyrad_radio'].config(text=self.get_text('pyradiomics_features'))
            if 'other_radio' in self.widgets:
                self.widgets['other_radio'].config(text=self.get_text('other_features_option'))
            if 'both_radio' in self.widgets:
                self.widgets['both_radio'].config(text=self.get_text('both_features'))

            if 'other_feature_group' in self.widgets:
                self.widgets['other_feature_group'].config(text=self.get_text('other_features'))
            if 'other_param_group' in self.widgets:
                self.widgets['other_param_group'].config(text=self.get_text('other_feature_settings'))
            if 'csf_label_label' in self.widgets:
                self.widgets['csf_label_label'].config(text=self.get_text('dural_sac_label'))

            if 'basic_group' in self.widgets:
                self.widgets['basic_group'].config(text=self.get_text('parameter_settings'))
            if 'bin_width_label' in self.widgets:
                self.widgets['bin_width_label'].config(text=self.get_text('bin_width'))
            if 'bin_count_label' in self.widgets:
                self.widgets['bin_count_label'].config(text=self.get_text('bin_count'))
            if 'resample_label' in self.widgets:
                self.widgets['resample_label'].config(text=self.get_text('resample_spacing'))
            if 'interp_label' in self.widgets:
                self.widgets['interp_label'].config(text=self.get_text('interpolator'))
            if 'normalize_cb' in self.widgets:
                self.widgets['normalize_cb'].config(text=self.get_text('normalize'))
            if 'scale_label' in self.widgets:
                self.widgets['scale_label'].config(text=self.get_text('scale'))
            if 'outlier_label' in self.widgets:
                self.widgets['outlier_label'].config(text=self.get_text('remove_outliers'))
            if 'correct_cb' in self.widgets:
                self.widgets['correct_cb'].config(text=self.get_text('correct_mask'))
            if 'label_label' in self.widgets:
                self.widgets['label_label'].config(text=self.get_text('label'))

            if 'feature_group' in self.widgets:
                self.widgets['feature_group'].config(text=self.get_text('feature_classes'))

            if 'advanced_group' in self.widgets:
                self.widgets['advanced_group'].config(text=self.get_text('advanced_params'))
            if 'info_cb' in self.widgets:
                self.widgets['info_cb'].config(text=self.get_text('additional_info'))
            if 'c_ext_cb' in self.widgets:
                self.widgets['c_ext_cb'].config(text=self.get_text('enable_c_extensions'))

            if 'log_group' in self.widgets:
                self.widgets['log_group'].config(text=self.get_text('log_filter'))
            if 'log_cb' in self.widgets:
                self.widgets['log_cb'].config(text=self.get_text('enable_log'))
            if 'sigma_label' in self.widgets:
                self.widgets['sigma_label'].config(text=self.get_text('sigma_values'))
            if 'wavelet_group' in self.widgets:
                self.widgets['wavelet_group'].config(text=self.get_text('wavelet_filter'))
            if 'wavelet_cb' in self.widgets:
                self.widgets['wavelet_cb'].config(text=self.get_text('enable_wavelet'))
            if 'simple_group' in self.widgets:
                self.widgets['simple_group'].config(text=self.get_text('simple_filters'))
            if 'gradient_group' in self.widgets:
                self.widgets['gradient_group'].config(text=self.get_text('gradient_filter'))
            if 'grad_cb' in self.widgets:
                self.widgets['grad_cb'].config(text=self.get_text('enable_gradient'))
            if 'grad_spacing_cb' in self.widgets:
                self.widgets['grad_spacing_cb'].config(text=self.get_text('use_spacing'))
            if 'lbp_group' in self.widgets:
                self.widgets['lbp_group'].config(text=self.get_text('lbp_filter'))
            if 'lbp2d_cb' in self.widgets:
                self.widgets['lbp2d_cb'].config(text=self.get_text('enable_lbp2d'))
            if 'radius2d_label' in self.widgets:
                self.widgets['radius2d_label'].config(text=self.get_text('radius'))
            if 'samples2d_label' in self.widgets:
                self.widgets['samples2d_label'].config(text=self.get_text('samples'))
            if 'lbp3d_cb' in self.widgets:
                self.widgets['lbp3d_cb'].config(text=self.get_text('enable_lbp3d'))
            if 'levels3d_label' in self.widgets:
                self.widgets['levels3d_label'].config(text=self.get_text('levels'))

            if 'reseg_group' in self.widgets:
                self.widgets['reseg_group'].config(text=self.get_text('resegmentation_settings'))
            if 'range_hint' in self.widgets:
                self.widgets['range_hint'].config(text=self.get_text('example_range'))
            if 'shape_cb' in self.widgets:
                self.widgets['shape_cb'].config(text=self.get_text('resegment_shape'))
            if 'd2_group' in self.widgets:
                self.widgets['d2_group'].config(text=self.get_text('2d_settings'))
            if 'force2d_cb' in self.widgets:
                self.widgets['force2d_cb'].config(text=self.get_text('force_2d'))
            if 'd2_dim_label' in self.widgets:
                self.widgets['d2_dim_label'].config(text=self.get_text('2d_dimension'))
            if 'd2_hint' in self.widgets:
                self.widgets['d2_hint'].config(text=self.get_text('axial_coronal_sagittal'))
            if 'd2_aggregator_label' in self.widgets:
                self.widgets['d2_aggregator_label'].config(text=self.get_text('force2d_aggregator'))
            if 'texture_group' in self.widgets:
                self.widgets['texture_group'].config(text=self.get_text('texture_matrix_settings'))
            if 'sym_cb' in self.widgets:
                self.widgets['sym_cb'].config(text=self.get_text('symmetrical_glcm'))
            if 'other_group' in self.widgets:
                self.widgets['other_group'].config(text=self.get_text('other_settings'))
            if 'vshift_label' in self.widgets:
                self.widgets['vshift_label'].config(text=self.get_text('voxel_array_shift'))
            if 'precrop_cb' in self.widgets:
                self.widgets['precrop_cb'].config(text=self.get_text('pre_crop'))

            if hasattr(self, 'feature_checkboxes'):
                for cb, key in self.feature_checkboxes:
                    cb.configure(text=self.get_text(key))

            if hasattr(self, 'advanced_labels'):
                for label, key in self.advanced_labels:
                    label.configure(text=self.get_text(key))

            if hasattr(self, 'wavelet_labels'):
                for label, key in self.wavelet_labels:
                    label.configure(text=self.get_text(key))

            if hasattr(self, 'simple_filters'):
                for cb, key in self.simple_filters:
                    cb.configure(text=self.get_text(key))

            if hasattr(self, 'reseg_labels'):
                for label, key in self.reseg_labels:
                    label.configure(text=self.get_text(key))

            if hasattr(self, 'texture_labels'):
                for label, key in self.texture_labels:
                    label.configure(text=self.get_text(key))

            if 'param_frame' in self.widgets:
                self.widgets['param_frame'].config(text=self.get_text('parameter_management'))
            if 'save_btn' in self.widgets:
                self.widgets['save_btn'].config(text=self.get_text('save_params'))
            if 'load_btn' in self.widgets:
                self.widgets['load_btn'].config(text=self.get_text('load_params'))
            if 'reset_btn' in self.widgets:
                self.widgets['reset_btn'].config(text=self.get_text('reset_defaults'))

            if 'exec_frame' in self.widgets:
                self.widgets['exec_frame'].config(text=self.get_text('execution_control'))
            if 'start_btn' in self.widgets:
                self.widgets['start_btn'].config(text=self.get_text('start_extraction'))
            if 'stop_btn' in self.widgets:
                self.widgets['stop_btn'].config(text=self.get_text('stop'))

            if 'log_frame' in self.widgets:
                self.widgets['log_frame'].config(text=self.get_text('run_log'))

            if 'parallel_frame' in self.widgets:
                self.widgets['parallel_frame'].config(text=self.get_text('parallel_settings'))
            if 'parallel_cb' in self.widgets:
                self.widgets['parallel_cb'].config(text=self.get_text('enable_parallel'))
            if 'worker_label' in self.widgets:
                self.widgets['worker_label'].config(text=self.get_text('worker_processes'))
            if 'cpu_label' in self.widgets:
                self.widgets['cpu_label'].config(text=f"({self.get_text('cpu_cores')} {mp.cpu_count()})")

        if hasattr(self, 'other_feature_checkboxes'):
            texts = [
                self.get_text('enable_dhi'),
                self.get_text('enable_asi'),
                self.get_text('enable_fd'),
                self.get_text('enable_t2si'),
                self.get_text('enable_gabor'),
                self.get_text('enable_hu'),
                self.get_text('enable_texture'),
                self.get_text('enable_dscr')  
            ]
            for i, cb in enumerate(self.other_feature_checkboxes):
                if i < len(texts):
                    cb.config(text=texts[i])

        if hasattr(self, 'main_notebook'):
            tabs = self.main_notebook.tabs()
            if len(tabs) >= 3:
                if self.current_lang.get() == "cn":
                    self.main_notebook.tab(tabs[0], text="特征提取")
                    self.main_notebook.tab(tabs[1], text="图像扰动")
                    self.main_notebook.tab(tabs[2], text="稳健性相关性分析")
                else:
                    self.main_notebook.tab(tabs[0], text="Feature Extraction")
                    self.main_notebook.tab(tabs[1], text="Image Perturbation")
                    self.main_notebook.tab(tabs[2], text="Robustness & Correlation Analysis")

        if hasattr(self, 'notebook'):
            tabs = self.notebook.tabs()
            if len(tabs) > 0:
                self.notebook.tab(tabs[0], text=self.get_text('other_features'))
            if PYRADIOMICS_AVAILABLE:
                if len(tabs) > 1:
                    self.notebook.tab(tabs[1], text=self.get_text('basic_settings'))
                if len(tabs) > 2:
                    self.notebook.tab(tabs[2], text=self.get_text('filter_settings'))
                if len(tabs) > 3:
                    self.notebook.tab(tabs[3], text=self.get_text('advanced_settings'))

        if hasattr(self, 'log_text'):
            current_content = self.log_text.get(1.0, tk.END).strip()
            if not current_content or current_content.startswith('🎯'):
                self.log_text.delete(1.0, tk.END)
                self.log_text.insert(tk.END, self.get_text('welcome_msg').strip())

    def create_logo(self):
        if PIL_AVAILABLE:
            try:
                possible_paths = [
                    os.path.join(os.path.dirname(__file__), "..", "logo", "fudan_logo.png"),
                    os.path.join(os.getcwd(), "logo", "fudan_logo.png"),
                    "logo/fudan_logo.png"
                ]
                
                for logo_path in possible_paths:
                    if os.path.exists(logo_path):
                        img = Image.open(logo_path)
                        img = img.resize((80, 80), Image.Resampling.LANCZOS)
                        return ImageTk.PhotoImage(img)

                return None
                
            except Exception as e:
                self.logger.warning(f"加载logo失败: {str(e)}")
                return None
        else:
            return None
        
    def setup_gui(self):
        self.root.configure(bg='#f0f0f0')
        
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True)
        
        self._setup_main_header(main_container)
        
        self.main_notebook = ttk.Notebook(main_container)
        self.main_notebook.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        
        style = ttk.Style()
        style.configure('TNotebook.Tab', padding=[20, 10])
        style.map('TNotebook.Tab',
                padding=[('selected', [20, 10])],
                background=[('selected', '#0078d4')],
                foreground=[('selected', 'black')])
        
        style.configure('TNotebook.Tab', font=('Segoe UI', 10, 'bold'))
        
        extraction_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(extraction_frame, text="特征提取")
        self._setup_extraction_tab(extraction_frame)
        
        perturbation_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(perturbation_frame, text="图像扰动")
        self.perturbation_gui = PerturbationGUI(perturbation_frame)
        
        robustness_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(robustness_frame, text="稳健性相关性分析")
        self.robustness_gui = RobustnessGUI(robustness_frame)
        
        self.main_notebook.bind("<<NotebookTabChanged>>", self._on_main_tab_changed)


    def _setup_main_header(self, parent):
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill="x", pady=(10, 10))
        
        logo_img = self.create_logo()
        if logo_img:
            logo_label = ttk.Label(header_frame, image=logo_img)
            logo_label.image = logo_img
            logo_label.pack(side="left", padx=(20, 20))
        
        title_label = ttk.Label(header_frame, text="椎间盘退变分析系统", 
                            style="Title.TLabel")
        title_label.pack(side="left")
        self.widgets['main_title_label'] = title_label
        

    def _setup_extraction_tab(self, parent):
        canvas = tk.Canvas(parent, bg='#f0f0f0', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        parent.bind_all("<MouseWheel>", _on_mousewheel)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        main_frame = ttk.Frame(scrollable_frame, padding="15")
        main_frame.pack(fill="both", expand=True)
        
        self.extraction_canvas = canvas

        self._setup_file_selection(main_frame)
        self._setup_feature_type(main_frame)
        self._setup_parameters_notebook(main_frame)
        self._setup_controls(main_frame)
        self._setup_log_display(main_frame)

    def _on_main_tab_changed(self, event):
        def _on_mousewheel(event, canvas):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        selected_tab_index = self.main_notebook.index(self.main_notebook.select())
        
        self.root.unbind_all("<MouseWheel>")

        if selected_tab_index == 0:
            if hasattr(self, 'extraction_canvas'):
                self.root.bind_all("<MouseWheel>", 
                                   lambda e, canvas=self.extraction_canvas: _on_mousewheel(e, canvas))
        elif selected_tab_index == 1:
            if hasattr(self.perturbation_gui, 'canvas'):
                self.root.bind_all("<MouseWheel>", 
                                   lambda e, canvas=self.perturbation_gui.canvas: _on_mousewheel(e, canvas))
        elif selected_tab_index == 2:
            if hasattr(self.robustness_gui, 'canvas'):
                self.root.bind_all("<MouseWheel>", 
                                   lambda e, canvas=self.robustness_gui.canvas: _on_mousewheel(e, canvas))

        tab_text = self.main_notebook.tab(self.main_notebook.select(), "text")
    
    def _setup_header(self, parent):
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill="x", pady=(0, 20))

        logo_img = self.create_logo()
        if logo_img:
            logo_label = ttk.Label(header_frame, image=logo_img)
            logo_label.image = logo_img
            logo_label.pack(side="left", padx=(0, 20))

        title_label = ttk.Label(header_frame, text=self.get_text('title'), 
                            style="Title.TLabel")
        title_label.pack(side="left")
        self.widgets['title_label'] = title_label

        lang_frame = ttk.Frame(header_frame)
        lang_frame.pack(side="right", padx=(20, 0))
        
        lang_label = ttk.Label(lang_frame, text="语言/Language:")
        lang_label.pack(side="left", padx=(0, 5))
        self.widgets['lang_label'] = lang_label

        lang_combo = ttk.Combobox(lang_frame, 
                                values=["中文", "English"], 
                                width=10, 
                                state="readonly")
        lang_combo.pack(side="left")
        lang_combo.set("中文" if self.current_lang.get() == "cn" else "English")
        
        def on_lang_change(event):
            if lang_combo.get() == "中文":
                self.current_lang.set("cn")
            else:
                self.current_lang.set("en")
            self.update_language()
        
        lang_combo.bind("<<ComboboxSelected>>", on_lang_change)

        self.widgets['lang_combo'] = lang_combo
    
    def _setup_file_selection(self, parent):
        file_frame = ttk.LabelFrame(parent, text=self.get_text('file_selection'), padding="10")
        file_frame.pack(fill="x", pady=5)
        self.widgets['file_frame'] = file_frame

        type_frame = ttk.Frame(file_frame)
        type_frame.pack(fill="x", pady=(0, 10))
        
        type_label = ttk.Label(type_frame, text=self.get_text('process_mode'))
        type_label.pack(side="left", padx=(0, 10))
        self.widgets['type_label'] = type_label
        
        batch_radio = ttk.Radiobutton(type_frame, text=self.get_text('batch_mode'), 
                                    variable=self.input_type, value="batch")
        batch_radio.pack(side="left", padx=(0, 20))
        self.widgets['batch_radio'] = batch_radio
        
        single_radio = ttk.Radiobutton(type_frame, text=self.get_text('single_mode'), 
                                    variable=self.input_type, value="single")
        single_radio.pack(side="left")
        self.widgets['single_radio'] = single_radio

        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill="x", pady=2)
        input_label = ttk.Label(input_frame, text=self.get_text('input_path'), width=10)
        input_label.pack(side="left")
        self.widgets['input_label'] = input_label
        
        input_entry = ttk.Entry(input_frame, textvariable=self.input_path)
        input_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.widgets['input_entry'] = input_entry
        
        input_btn = ttk.Button(input_frame, text="📂 " + self.get_text('select'), 
                            command=self.select_input)
        input_btn.pack(side="left")
        self.widgets['input_btn'] = input_btn

        mask_frame = ttk.Frame(file_frame)
        mask_frame.pack(fill="x", pady=2)
        mask_label = ttk.Label(mask_frame, text=self.get_text('mask_path'), width=10)
        mask_label.pack(side="left")
        self.widgets['mask_label'] = mask_label
        
        mask_entry = ttk.Entry(mask_frame, textvariable=self.mask_path)
        mask_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.widgets['mask_entry'] = mask_entry
        
        mask_btn = ttk.Button(mask_frame, text="🎯 " + self.get_text('select'), 
                            command=self.select_mask)
        mask_btn.pack(side="left")
        self.widgets['mask_btn'] = mask_btn

        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill="x", pady=2)
        output_label = ttk.Label(output_frame, text=self.get_text('output_path'), width=10)
        output_label.pack(side="left")
        self.widgets['output_label'] = output_label
        
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path)
        output_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.widgets['output_entry'] = output_entry
        
        output_btn = ttk.Button(output_frame, text="💾 " + self.get_text('select'), 
                            command=self.select_output)
        output_btn.pack(side="left")
        self.widgets['output_btn'] = output_btn

    def _setup_feature_type(self, parent):
        type_frame = ttk.LabelFrame(parent, text=self.get_text('feature_type'), padding="10")
        type_frame.pack(fill="x", pady=5)
        self.widgets['feature_type_frame'] = type_frame
        
        pyrad_radio = ttk.Radiobutton(type_frame, text=self.get_text('pyradiomics_features'), 
                                    variable=self.feature_type, value="pyradiomics",
                                    state="normal" if PYRADIOMICS_AVAILABLE else "disabled",
                                    command=self._on_feature_type_change)
        pyrad_radio.pack(side="left", padx=10)
        self.widgets['pyrad_radio'] = pyrad_radio
        
        other_radio = ttk.Radiobutton(type_frame, text=self.get_text('other_features_option'), 
                                    variable=self.feature_type, value="other",
                                    command=self._on_feature_type_change)
        other_radio.pack(side="left", padx=10)
        self.widgets['other_radio'] = other_radio
        
        both_radio = ttk.Radiobutton(type_frame, text=self.get_text('both_features'), 
                                    variable=self.feature_type, value="both",
                                    command=self._on_feature_type_change)
        both_radio.pack(side="left", padx=10)
        self.widgets['both_radio'] = both_radio

    def _on_feature_type_change(self):
        feature_type = self.feature_type.get()

        if hasattr(self, 'other_features_tab'):
            try:
                self.notebook.forget(self.other_features_tab)
            except tk.TclError:
                pass
        
        if hasattr(self, 'pyrad_tab') and PYRADIOMICS_AVAILABLE:
            try:
                self.notebook.forget(self.pyrad_tab)
            except tk.TclError:
                pass

        if feature_type == "other":
            self.notebook.add(self.other_features_tab, text="其他特征")
            self.notebook.select(self.other_features_tab)
        elif feature_type == "pyradiomics":
            if PYRADIOMICS_AVAILABLE:
                self.notebook.add(self.pyrad_tab, text="PyRadiomic特征")
                self.notebook.select(self.pyrad_tab)
        elif feature_type == "both":
            self.notebook.add(self.other_features_tab, text="其他特征")
            if PYRADIOMICS_AVAILABLE:
                self.notebook.add(self.pyrad_tab, text="PyRadiomic特征")
            self.notebook.select(self.other_features_tab)
    
    def _setup_parameters_notebook(self, parent):
        param_container = ttk.LabelFrame(parent, text="参数设置", padding="10")
        param_container.pack(fill="x", pady=5)

        self.notebook = ttk.Notebook(param_container)
        self.notebook.pack(fill="x", expand=True, pady=5)

        self.other_features_tab = ttk.Frame(self.notebook, padding="10")
        self._setup_other_features(self.other_features_tab)

        if PYRADIOMICS_AVAILABLE:
            self.pyrad_tab = ttk.Frame(self.notebook, padding="10")
            pyrad_main_frame = ttk.Frame(self.pyrad_tab)
            pyrad_main_frame.pack(fill="both", expand=True)
            self._setup_basic_settings(pyrad_main_frame)
            self._setup_filter_settings(pyrad_main_frame)
            self._setup_advanced_settings(pyrad_main_frame)
        
        self._on_feature_type_change()
    
    def _setup_basic_settings(self, parent):
        basic_left = ttk.Frame(parent)
        basic_left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        basic_right = ttk.Frame(parent)
        basic_right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        basic_group = ttk.LabelFrame(basic_left, text=self.get_text('parameter_settings'), padding="10")
        basic_group.pack(fill="x", pady=5)
        self.widgets['basic_group'] = basic_group

        row = 0
        bin_width_label = ttk.Label(basic_group, text=self.get_text('bin_width'))
        bin_width_label.grid(row=row, column=0, sticky="w", pady=2)
        self.widgets['bin_width_label'] = bin_width_label
        ttk.Entry(basic_group, textvariable=self.bin_width, width=15).grid(row=row, column=1, sticky="w", pady=2, padx=5)

        row += 1
        bin_count_label = ttk.Label(basic_group, text=self.get_text('bin_count'))
        bin_count_label.grid(row=row, column=0, sticky="w", pady=2)
        self.widgets['bin_count_label'] = bin_count_label
        ttk.Entry(basic_group, textvariable=self.bin_count, width=15).grid(row=row, column=1, sticky="w", pady=2, padx=5)

        row += 1
        resample_label = ttk.Label(basic_group, text=self.get_text('resample_spacing'))
        resample_label.grid(row=row, column=0, sticky="w", pady=2)
        self.widgets['resample_label'] = resample_label
        ttk.Entry(basic_group, textvariable=self.resample_spacing, width=15).grid(row=row, column=1, sticky="w", pady=2, padx=5)

        row += 1
        interp_label = ttk.Label(basic_group, text=self.get_text('interpolator'))
        interp_label.grid(row=row, column=0, sticky="w", pady=2)
        self.widgets['interp_label'] = interp_label
        interp_combo = ttk.Combobox(basic_group, textvariable=self.interpolator, 
                                values=["sitkBSpline", "sitkLinear", "sitkNearestNeighbor", 
                                        "sitkGaussian", "sitkLabelGaussian"],
                                width=20, state="readonly")
        interp_combo.grid(row=row, column=1, sticky="w", pady=2, padx=5)

        row += 1
        normalize_frame = ttk.Frame(basic_group)
        normalize_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        normalize_cb = ttk.Checkbutton(normalize_frame, text=self.get_text('normalize'), variable=self.normalize)
        normalize_cb.pack(side="left")
        self.widgets['normalize_cb'] = normalize_cb
        scale_label = ttk.Label(normalize_frame, text=self.get_text('scale'))
        scale_label.pack(side="left", padx=(10, 5))
        self.widgets['scale_label'] = scale_label
        ttk.Entry(normalize_frame, textvariable=self.normalize_scale, width=8).pack(side="left")

        row += 1
        outlier_label = ttk.Label(basic_group, text=self.get_text('remove_outliers'))
        outlier_label.grid(row=row, column=0, sticky="w", pady=2)
        self.widgets['outlier_label'] = outlier_label
        ttk.Entry(basic_group, textvariable=self.remove_outliers, width=15).grid(row=row, column=1, sticky="w", pady=2, padx=5)

        row += 1
        correct_cb = ttk.Checkbutton(basic_group, text=self.get_text('correct_mask'), variable=self.correct_mask)
        correct_cb.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        self.widgets['correct_cb'] = correct_cb

        row += 1
        label_label = ttk.Label(basic_group, text=self.get_text('label'))
        label_label.grid(row=row, column=0, sticky="w", pady=2)
        self.widgets['label_label'] = label_label
        ttk.Entry(basic_group, textvariable=self.label, width=5).grid(row=row, column=1, sticky="w", pady=2, padx=5)

        feature_group = ttk.LabelFrame(basic_right, text=self.get_text('feature_classes'), padding="10")
        feature_group.pack(fill="x", pady=5)
        self.widgets['feature_group'] = feature_group
        
        feature_vars = [
            (self.enable_shape, 'shape_3d'),
            (self.enable_shape2d, 'shape_2d'),
            (self.enable_firstorder, 'firstorder'),
            (self.enable_glcm, 'glcm'),
            (self.enable_glrlm, 'glrlm'),
            (self.enable_glszm, 'glszm'),
            (self.enable_gldm, 'gldm'),
            (self.enable_ngtdm, 'ngtdm')
        ]
        
        self.feature_checkboxes = []
        for i, (var, key) in enumerate(feature_vars):
            cb = ttk.Checkbutton(feature_group, text=self.get_text(key), variable=var)
            cb.grid(row=i, column=0, sticky="w", pady=1)
            self.feature_checkboxes.append((cb, key))

        advanced_group = ttk.LabelFrame(basic_right, text=self.get_text('advanced_params'), padding="10")
        advanced_group.pack(fill="x", pady=5)
        self.widgets['advanced_group'] = advanced_group
        
        self.advanced_labels = []
        row = 0
        pad_label = ttk.Label(advanced_group, text=self.get_text('pad_distance'))
        pad_label.grid(row=row, column=0, sticky="w", pady=2)
        self.advanced_labels.append((pad_label, 'pad_distance'))
        ttk.Entry(advanced_group, textvariable=self.pad_distance, width=5).grid(row=row, column=1, sticky="w", pady=2, padx=5)
        
        row += 1
        geo_label = ttk.Label(advanced_group, text=self.get_text('geometry_tolerance'))
        geo_label.grid(row=row, column=0, sticky="w", pady=2)
        self.advanced_labels.append((geo_label, 'geometry_tolerance'))
        ttk.Entry(advanced_group, textvariable=self.geometry_tolerance, width=10).grid(row=row, column=1, sticky="w", pady=2, padx=5)
        
        row += 1
        min_dim_label = ttk.Label(advanced_group, text=self.get_text('min_roi_dimensions'))
        min_dim_label.grid(row=row, column=0, sticky="w", pady=2)
        self.advanced_labels.append((min_dim_label, 'min_roi_dimensions'))
        ttk.Entry(advanced_group, textvariable=self.minimum_roi_dimensions, width=5).grid(row=row, column=1, sticky="w", pady=2, padx=5)
        
        row += 1
        min_size_label = ttk.Label(advanced_group, text=self.get_text('min_roi_size'))
        min_size_label.grid(row=row, column=0, sticky="w", pady=2)
        self.advanced_labels.append((min_size_label, 'min_roi_size'))
        ttk.Entry(advanced_group, textvariable=self.minimum_roi_size, width=5).grid(row=row, column=1, sticky="w", pady=2, padx=5)
        
        row += 1
        info_cb = ttk.Checkbutton(advanced_group, text=self.get_text('additional_info'), variable=self.additional_info)
        info_cb.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        self.widgets['info_cb'] = info_cb
        
        row += 1
        c_ext_cb = ttk.Checkbutton(advanced_group, text=self.get_text('enable_c_extensions'), variable=self.enable_c_extensions)
        c_ext_cb.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        self.widgets['c_ext_cb'] = c_ext_cb

    def _setup_filter_settings(self, parent):
        filter_left = ttk.Frame(parent)
        filter_left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        log_group = ttk.LabelFrame(filter_left, text=self.get_text('log_filter'), padding="10")
        log_group.pack(fill="x", pady=5)
        self.widgets['log_group'] = log_group
        
        log_cb = ttk.Checkbutton(log_group, text=self.get_text('enable_log'), variable=self.enable_log)
        log_cb.pack(anchor="w", pady=2)
        self.widgets['log_cb'] = log_cb
        
        sigma_frame = ttk.Frame(log_group)
        sigma_frame.pack(fill="x", pady=2)
        sigma_label = ttk.Label(sigma_frame, text=self.get_text('sigma_values'))
        sigma_label.pack(side="left")
        self.widgets['sigma_label'] = sigma_label
        ttk.Entry(sigma_frame, textvariable=self.log_sigma, width=20).pack(side="left", padx=5)

        wavelet_group = ttk.LabelFrame(filter_left, text=self.get_text('wavelet_filter'), padding="10")
        wavelet_group.pack(fill="x", pady=5)
        self.widgets['wavelet_group'] = wavelet_group
        
        wavelet_cb = ttk.Checkbutton(wavelet_group, text=self.get_text('enable_wavelet'), variable=self.enable_wavelet)
        wavelet_cb.pack(anchor="w", pady=2)
        self.widgets['wavelet_cb'] = wavelet_cb
        
        wavelet_settings = ttk.Frame(wavelet_group)
        wavelet_settings.pack(fill="x", pady=2)
        
        self.wavelet_labels = []
        row = 0
        wtype_label = ttk.Label(wavelet_settings, text=self.get_text('wavelet_type'))
        wtype_label.grid(row=row, column=0, sticky="w", pady=2)
        self.wavelet_labels.append((wtype_label, 'wavelet_type'))
        wavelet_combo = ttk.Combobox(wavelet_settings, textvariable=self.wavelet_type,
                                    values=["coif1", "db1", "db2", "db3", "db4", "db5",
                                        "haar", "sym2", "sym3", "bior1.1", "rbio1.1"],
                                    width=15, state="readonly")
        wavelet_combo.grid(row=row, column=1, sticky="w", pady=2, padx=5)
        
        row += 1
        wlevel_label = ttk.Label(wavelet_settings, text=self.get_text('decomposition_level'))
        wlevel_label.grid(row=row, column=0, sticky="w", pady=2)
        self.wavelet_labels.append((wlevel_label, 'decomposition_level'))
        ttk.Entry(wavelet_settings, textvariable=self.wavelet_level, width=5).grid(row=row, column=1, sticky="w", pady=2, padx=5)
        
        row += 1
        wstart_label = ttk.Label(wavelet_settings, text=self.get_text('start_level'))
        wstart_label.grid(row=row, column=0, sticky="w", pady=2)
        self.wavelet_labels.append((wstart_label, 'start_level'))
        ttk.Entry(wavelet_settings, textvariable=self.wavelet_start_level, width=5).grid(row=row, column=1, sticky="w", pady=2, padx=5)

        filter_right = ttk.Frame(parent)
        filter_right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        simple_group = ttk.LabelFrame(filter_right, text=self.get_text('simple_filters'), padding="10")
        simple_group.pack(fill="x", pady=5)
        self.widgets['simple_group'] = simple_group
        
        self.simple_filters = []
        filters = [
            (self.enable_square, 'square'),
            (self.enable_square_root, 'square_root'),
            (self.enable_logarithm, 'logarithm'),
            (self.enable_exponential, 'exponential')
        ]
        
        for var, key in filters:
            cb = ttk.Checkbutton(simple_group, text=self.get_text(key), variable=var)
            cb.pack(anchor="w", pady=1)
            self.simple_filters.append((cb, key))

        gradient_group = ttk.LabelFrame(filter_right, text=self.get_text('gradient_filter'), padding="10")
        gradient_group.pack(fill="x", pady=5)
        self.widgets['gradient_group'] = gradient_group
        
        grad_cb = ttk.Checkbutton(gradient_group, text=self.get_text('enable_gradient'), variable=self.enable_gradient)
        grad_cb.pack(anchor="w", pady=2)
        self.widgets['grad_cb'] = grad_cb
        
        grad_spacing_cb = ttk.Checkbutton(gradient_group, text=self.get_text('use_spacing'), variable=self.gradient_sigma)
        grad_spacing_cb.pack(anchor="w", pady=2)
        self.widgets['grad_spacing_cb'] = grad_spacing_cb

        lbp_group = ttk.LabelFrame(filter_right, text=self.get_text('lbp_filter'), padding="10")
        lbp_group.pack(fill="x", pady=5)
        self.widgets['lbp_group'] = lbp_group

        lbp2d_frame = ttk.Frame(lbp_group)
        lbp2d_frame.pack(fill="x", pady=2)
        lbp2d_cb = ttk.Checkbutton(lbp2d_frame, text=self.get_text('enable_lbp2d'), variable=self.enable_lbp2d)
        lbp2d_cb.pack(side="left")
        self.widgets['lbp2d_cb'] = lbp2d_cb
        
        radius2d_label = ttk.Label(lbp2d_frame, text=self.get_text('radius'))
        radius2d_label.pack(side="left", padx=(10, 5))
        self.widgets['radius2d_label'] = radius2d_label
        ttk.Entry(lbp2d_frame, textvariable=self.lbp2d_radius, width=5).pack(side="left")
        
        samples2d_label = ttk.Label(lbp2d_frame, text=self.get_text('samples'))
        samples2d_label.pack(side="left", padx=(10, 5))
        self.widgets['samples2d_label'] = samples2d_label
        ttk.Entry(lbp2d_frame, textvariable=self.lbp2d_samples, width=5).pack(side="left")

        lbp3d_frame = ttk.Frame(lbp_group)
        lbp3d_frame.pack(fill="x", pady=2)
        lbp3d_cb = ttk.Checkbutton(lbp3d_frame, text=self.get_text('enable_lbp3d'), variable=self.enable_lbp3d)
        lbp3d_cb.pack(side="left")
        self.widgets['lbp3d_cb'] = lbp3d_cb
        
        levels3d_label = ttk.Label(lbp3d_frame, text=self.get_text('levels'))
        levels3d_label.pack(side="left", padx=(10, 5))
        self.widgets['levels3d_label'] = levels3d_label
        ttk.Entry(lbp3d_frame, textvariable=self.lbp3d_levels, width=5).pack(side="left")

    def _setup_advanced_settings(self, parent):
        adv_left = ttk.Frame(parent)
        adv_left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        reseg_group = ttk.LabelFrame(adv_left, text=self.get_text('resegmentation_settings'), padding="10")
        reseg_group.pack(fill="x", pady=5)
        self.widgets['reseg_group'] = reseg_group
        
        self.reseg_labels = []
        row = 0
        range_label = ttk.Label(reseg_group, text=self.get_text('resegment_range'))
        range_label.grid(row=row, column=0, sticky="w", pady=2)
        self.reseg_labels.append((range_label, 'resegment_range'))
        ttk.Entry(reseg_group, textvariable=self.resegment_range, width=15).grid(row=row, column=1, sticky="w", pady=2, padx=5)
        range_hint = ttk.Label(reseg_group, text="(例: -50,100)", font=('Segoe UI', 8))
        range_hint.grid(row=row, column=2, sticky="w", pady=2)
        self.widgets['range_hint'] = range_hint
        
        row += 1
        mode_label = ttk.Label(reseg_group, text=self.get_text('resegment_mode'))
        mode_label.grid(row=row, column=0, sticky="w", pady=2)
        self.reseg_labels.append((mode_label, 'resegment_mode'))
        mode_combo = ttk.Combobox(reseg_group, textvariable=self.resegment_mode,
                                values=["absolute", "relative", "sigma"],
                                width=12, state="readonly")
        mode_combo.grid(row=row, column=1, sticky="w", pady=2, padx=5)
        
        row += 1
        shape_cb = ttk.Checkbutton(reseg_group, text=self.get_text('resegment_shape'), variable=self.resegment_shape)
        shape_cb.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        self.widgets['shape_cb'] = shape_cb

        d2_group = ttk.LabelFrame(adv_left, text=self.get_text('2d_settings'), padding="10")
        d2_group.pack(fill="x", pady=5)
        self.widgets['d2_group'] = d2_group
        
        force2d_cb = ttk.Checkbutton(d2_group, text=self.get_text('force_2d'), variable=self.force2D)
        force2d_cb.pack(anchor="w", pady=2)
        self.widgets['force2d_cb'] = force2d_cb
        
        d2_dim_frame = ttk.Frame(d2_group)
        d2_dim_frame.pack(fill="x", pady=2)
        d2_dim_label = ttk.Label(d2_dim_frame, text=self.get_text('2d_dimension'))
        d2_dim_label.pack(side="left")
        self.widgets['d2_dim_label'] = d2_dim_label
        ttk.Entry(d2_dim_frame, textvariable=self.force2D_dimension, width=5).pack(side="left", padx=5)
        d2_hint = ttk.Label(d2_dim_frame, text="(0=轴向, 1=冠状, 2=矢状)", font=('Segoe UI', 8))
        d2_hint.pack(side="left", padx=5)
        self.widgets['d2_hint'] = d2_hint
        
        aggregator_frame = ttk.Frame(d2_group)
        aggregator_frame.pack(fill="x", pady=2)
        aggregator_label = ttk.Label(aggregator_frame, text=self.get_text('force2d_aggregator'))
        aggregator_label.pack(side="left")
        self.widgets['d2_aggregator_label'] = aggregator_label
        aggregator_combo = ttk.Combobox(aggregator_frame, textvariable=self.force2D_aggregator,
                                    values=["mean", "max", "min", "std", "sum"],
                                    width=10, state="readonly")
        aggregator_combo.pack(side="left", padx=5)

        adv_right = ttk.Frame(parent)
        adv_right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        texture_group = ttk.LabelFrame(adv_right, text=self.get_text('texture_matrix_settings'), padding="10")
        texture_group.pack(fill="x", pady=5)
        self.widgets['texture_group'] = texture_group
        
        self.texture_labels = []
        row = 0
        weight_label = ttk.Label(texture_group, text=self.get_text('weighting_norm'))
        weight_label.grid(row=row, column=0, sticky="w", pady=2)
        self.texture_labels.append((weight_label, 'weighting_norm'))
        weight_combo = ttk.Combobox(texture_group, textvariable=self.weighting_norm,
                                values=["manhattan", "euclidean", "infinity", "no_weighting"],
                                width=15, state="readonly")
        weight_combo.grid(row=row, column=1, sticky="w", pady=2, padx=5)
        
        row += 1
        dist_label = ttk.Label(texture_group, text=self.get_text('distances'))
        dist_label.grid(row=row, column=0, sticky="w", pady=2)
        self.texture_labels.append((dist_label, 'distances'))
        ttk.Entry(texture_group, textvariable=self.distances, width=15).grid(row=row, column=1, sticky="w", pady=2, padx=5)
        
        row += 1
        sym_cb = ttk.Checkbutton(texture_group, text=self.get_text('symmetrical_glcm'), variable=self.symmetrical_glcm)
        sym_cb.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        self.widgets['sym_cb'] = sym_cb
        
        row += 1
        gldm_label = ttk.Label(texture_group, text=self.get_text('gldm_alpha'))
        gldm_label.grid(row=row, column=0, sticky="w", pady=2)
        self.texture_labels.append((gldm_label, 'gldm_alpha'))
        ttk.Entry(texture_group, textvariable=self.gldm_a, width=10).grid(row=row, column=1, sticky="w", pady=2, padx=5)

        other_group = ttk.LabelFrame(adv_right, text=self.get_text('other_settings'), padding="10")
        other_group.pack(fill="x", pady=5)
        self.widgets['other_group'] = other_group
        
        row = 0
        vshift_label = ttk.Label(other_group, text=self.get_text('voxel_array_shift'))
        vshift_label.grid(row=row, column=0, sticky="w", pady=2)
        self.widgets['vshift_label'] = vshift_label
        ttk.Entry(other_group, textvariable=self.voxel_array_shift, width=10).grid(row=row, column=1, sticky="w", pady=2, padx=5)
        
        row += 1
        precrop_cb = ttk.Checkbutton(other_group, text=self.get_text('pre_crop'), variable=self.preCrop)
        precrop_cb.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        self.widgets['precrop_cb'] = precrop_cb

    def _setup_other_features(self, parent):
        feature_group = ttk.LabelFrame(parent, text=self.get_text('other_features'), padding="10")
        feature_group.pack(fill="x", pady=5)
        self.widgets['other_feature_group'] = feature_group

        self.other_feature_checkboxes = []

        left_frame = ttk.Frame(feature_group)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 20))
        
        cb1 = ttk.Checkbutton(left_frame, text=self.get_text('enable_dhi'), 
                            variable=self.enable_other_dhi)
        cb1.pack(anchor="w", pady=2)
        self.other_feature_checkboxes.append(cb1)
        
        cb2 = ttk.Checkbutton(left_frame, text=self.get_text('enable_asi'), 
                            variable=self.enable_other_asi)
        cb2.pack(anchor="w", pady=2)
        self.other_feature_checkboxes.append(cb2)
        
        cb3 = ttk.Checkbutton(left_frame, text=self.get_text('enable_fd'), 
                            variable=self.enable_other_fd)
        cb3.pack(anchor="w", pady=2)
        self.other_feature_checkboxes.append(cb3)
        
        cb4 = ttk.Checkbutton(left_frame, text=self.get_text('enable_t2si'), 
                            variable=self.enable_other_t2si)
        cb4.pack(anchor="w", pady=2)
        self.other_feature_checkboxes.append(cb4)

        right_frame = ttk.Frame(feature_group)
        right_frame.pack(side="left", fill="both", expand=True)
        
        cb5 = ttk.Checkbutton(right_frame, text=self.get_text('enable_gabor'), 
                            variable=self.enable_other_gabor)
        cb5.pack(anchor="w", pady=2)
        self.other_feature_checkboxes.append(cb5)
        
        cb6 = ttk.Checkbutton(right_frame, text=self.get_text('enable_hu'), 
                            variable=self.enable_other_hu)
        cb6.pack(anchor="w", pady=2)
        self.other_feature_checkboxes.append(cb6)
        
        cb7 = ttk.Checkbutton(right_frame, text=self.get_text('enable_texture'), 
                            variable=self.enable_other_texture)
        cb7.pack(anchor="w", pady=2)
        self.other_feature_checkboxes.append(cb7)

        cb8 = ttk.Checkbutton(right_frame, text=self.get_text('enable_dscr'), 
                            variable=self.enable_other_dscr)
        cb8.pack(anchor="w", pady=2)
        self.other_feature_checkboxes.append(cb8)


        parallel_frame = ttk.LabelFrame(parent, text=self.get_text('parallel_settings'), padding="10")
        parallel_frame.pack(fill="x", pady=5)
        self.widgets['parallel_frame'] = parallel_frame

        self.enable_parallel = tk.BooleanVar(value=True)
        parallel_cb = ttk.Checkbutton(parallel_frame, text=self.get_text('enable_parallel'), 
                                    variable=self.enable_parallel)
        parallel_cb.pack(anchor="w")
        self.widgets['parallel_cb'] = parallel_cb

        worker_frame = ttk.Frame(parallel_frame)
        worker_frame.pack(fill="x", pady=5)

        worker_label = ttk.Label(worker_frame, text=self.get_text('worker_processes'))
        worker_label.pack(side="left")
        self.widgets['worker_label'] = worker_label

        self.max_workers = tk.IntVar(value=mp.cpu_count())
        worker_spinbox = ttk.Spinbox(worker_frame, from_=1, to=mp.cpu_count()*2,
                                    textvariable=self.max_workers, width=10)
        worker_spinbox.pack(side="left", padx=5)

        cpu_label = ttk.Label(worker_frame, text=f"({self.get_text('cpu_cores')} {mp.cpu_count()})")
        cpu_label.pack(side="left")
        self.widgets['cpu_label'] = cpu_label
    
    def _setup_controls(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", pady=10)

        start_btn = ttk.Button(control_frame, text="🚀 开始提取", 
                                command=self.start_extraction)
        
        start_btn.pack(anchor="center")
        
        self.widgets['start_btn'] = start_btn

    def _setup_log_display(self, parent):
        log_frame = ttk.LabelFrame(parent, text=self.get_text('run_log'), padding="5")
        log_frame.pack(fill="both", expand=True, pady=5)
        self.widgets['log_frame'] = log_frame
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, font=('Consolas', 9), wrap=tk.WORD)
        self.log_text.pack(fill="both", expand=True)
        
        self.log_text.insert(tk.END, "🎯 椎间盘退变特征提取系统已就绪\n")
    
    def log_message(self, message):
        if threading.current_thread() == threading.main_thread():
            if '\n' in message:
                lines = message.split('\n')
                for line in lines:
                    if line.strip():
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        self.log_text.insert(tk.END, f"[{timestamp}] {line}\n")
            else:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            
            self.log_text.see(tk.END)
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.update_idletasks()
        else:
            self.log_queue.put(message)
    
    def start_extraction(self):
        if not self.input_path.get():
            messagebox.showerror("错误", "请选择输入路径")
            return
        
        if not self.output_path.get():
            messagebox.showerror("错误", "请选择输出路径")
            return
        
        if self.input_type.get() == "single" and not self.mask_path.get():
            messagebox.showerror("错误", "单文件模式需要选择掩码文件")
            return

        self.extraction_thread = threading.Thread(target=self.run_extraction)
        self.extraction_thread.daemon = True
        self.extraction_thread.start()
    
    def extract_other_features(self, matched_pairs=None):
        results = {}
        
        try:
            if self.input_type.get() == "single":
                image_path = self.input_path.get()
                mask_path = self.mask_path.get()
                
                if not image_path or not mask_path:
                    raise ValueError("请选择图像和掩码文件")
                
                self.log_message("加载图像和掩码...")

                image, mask = self.image_io.load_image_and_mask(image_path, mask_path)

                spacing = list(image.GetSpacing())[::-1]
                self.log_message(f"图像间距: {spacing}")

                image_array = self.image_io.sitk_to_numpy(image)
                mask_array = self.image_io.sitk_to_numpy(mask)

                p = Path(image_path)
                base_name = p.name
                while Path(base_name).suffix:
                    base_name = Path(base_name).stem
                case_id = base_name

                result = {
                    'case_id': case_id,
                    'image_path': image_path,
                    'mask_path': mask_path
                }

                image_slices = self._extract_middle_slices(
                    image_array, self.config.NUM_SLICES, self.config.SLICE_AXIS
                )
                mask_slices = self._extract_middle_slices(
                    mask_array, self.config.NUM_SLICES, self.config.SLICE_AXIS
                )

                roi_masks = []
                for mask_slice in mask_slices:
                    roi_mask = (mask_slice > 0).astype(np.uint8)
                    roi_masks.append(roi_mask)

                if not any(np.any(mask) for mask in roi_masks):
                    self.log_message("⚠️ 没有找到有效的ROI区域")
                    result['status'] = 'no_roi_found'
                    return {'results': [result]}

                csf_masks = []
                if self.enable_other_asi.get() or self.enable_other_t2si.get():
                    csf_label = self.dural_sac_label.get()
                    for mask_slice in mask_slices:
                        csf_mask = (mask_slice == csf_label).astype(np.uint8)
                        csf_masks.append(csf_mask)

                    if not any(np.any(mask) for mask in csf_masks):
                        self.log_message("⚠️ 没有找到CSF区域，ASI/T2SI将使用全局标准化")
                        csf_masks = None

                if self.enable_other_dhi.get():
                    try:
                        self.log_message("计算DHI...")
                        
                        dhi_results_all_levels = []
                        
                        for level_name, labels in self.config.DISC_LABELS.items():
                            self.log_message(f"处理{level_name}层级")
                            
                            upper_masks = []
                            disc_masks_level = []
                            lower_masks = []
                            
                            for mask_slice in mask_slices:
                                upper_mask = (mask_slice == labels['upper']).astype(np.uint8)
                                upper_masks.append(upper_mask)
                                
                                disc_mask = (mask_slice == labels['disc']).astype(np.uint8)
                                disc_masks_level.append(disc_mask)
                                
                                lower_mask = (mask_slice == labels['lower']).astype(np.uint8)
                                lower_masks.append(lower_mask)
                            
                            if not any(np.any(mask) for mask in disc_masks_level):
                                self.log_message(f"{level_name}层级没有找到椎间盘区域，跳过")
                                continue
                                
                            is_l5_s1 = (level_name == 'L5-S1')
                            dhi_result = self.dhi_calculator.process_multi_slice(
                                upper_masks, disc_masks_level, lower_masks, is_l5_s1
                            )
                            
                            for key, value in dhi_result.items():
                                result[f'dhi_{level_name}_{key}'] = value
                            
                            self.log_message(f"{level_name} DHI = {dhi_result.get('dhi', 'N/A'):.3f}")
                            dhi_results_all_levels.append(dhi_result)
                        
                        if dhi_results_all_levels:
                            avg_dhi = np.mean([r['dhi'] for r in dhi_results_all_levels])
                            result['dhi_average'] = avg_dhi
                            self.log_message(f"平均DHI = {avg_dhi:.3f}")
                        
                    except Exception as e:
                        self.log_message(f"❌ DHI计算失败: {str(e)}")
                        result['dhi_error'] = str(e)

                processed_image_slices_for_si = None
                processed_mask_slices_for_si = None

                if self.enable_other_asi.get() or self.enable_other_t2si.get():

                    processed_image_slices_for_si = []
                    processed_mask_slices_for_si = []
                    
                    for i, img_slice in enumerate(image_slices):
                        slice_spacing = spacing[:2] + [1.0]
                        
                        processed_img, processed_mask = self.preprocessor.preprocess_for_signal_intensity(
                            img_slice, mask_slices[i], slice_spacing
                        )
                        unique_labels = np.unique(processed_mask)


                        processed_image_slices_for_si.append(processed_img)
                        processed_mask_slices_for_si.append(processed_mask.astype(np.int32))


                if self.enable_other_asi.get() and processed_image_slices_for_si:
                    self.log_message("计算ASI...")
                    for level_name, labels in self.config.DISC_LABELS.items():
                        disc_masks_level = [(p_mask.astype(np.int32) == int(labels['disc'])).astype(np.uint8) for p_mask in processed_mask_slices_for_si]
                        
                        if not any(np.any(mask) for mask in disc_masks_level):
                            continue
                            
                        csf_label = self.dural_sac_label.get()
                        csf_label_int = int(csf_label)
                        csf_masks_level = [(p_mask.astype(np.int32) == csf_label_int).astype(np.uint8) for p_mask in processed_mask_slices_for_si]
                        
                        try:
                            self.log_message(f"  -> 处理 {level_name} ASI...")
                            asi_result = self.asi_calculator.process_multi_slice(
                                processed_image_slices_for_si, disc_masks_level, csf_masks_level
                            )
                            result.update({f'asi_{level_name}_{k}': v for k, v in asi_result.items()})
                            self.log_message(f"  -> {level_name} ASI = {asi_result.get('asi', 'N/A'):3f}")
                        except Exception as e:
                            self.log_message(f"❌ {level_name} ASI计算失败: {str(e)}")
                            result[f'asi_{level_name}_error'] = str(e)

                if self.enable_other_t2si.get() and processed_image_slices_for_si:
                    self.log_message("计算T2信号强度...")
                    for level_name, labels in self.config.DISC_LABELS.items():
                        disc_masks_level = [(p_mask.astype(np.int32) == int(labels['disc'])).astype(np.uint8) for p_mask in processed_mask_slices_for_si]
                        
                        if not any(np.any(mask) for mask in disc_masks_level):
                            continue
                            
                        csf_label = self.dural_sac_label.get()
                        csf_label_int = int(csf_label)
                        csf_masks_level = [(p_mask.astype(np.int32) == csf_label_int).astype(np.uint8) for p_mask in processed_mask_slices_for_si]
                        
                        try:
                            self.log_message(f"  -> 处理 {level_name} T2SI...")
                            t2si_result = self.t2si_calculator.process_multi_slice(
                                processed_image_slices_for_si, 
                                disc_masks_level,
                                csf_masks_level
                            )
                            serializable_t2si_result = {k: v for k, v in t2si_result.items() if k != 'slice_results'}
                            result.update({f't2si_{level_name}_{k}': v for k, v in serializable_t2si_result.items()})
                            self.log_message(f"  -> {level_name} T2SI比率 = {t2si_result.get('si_ratio', 'N/A'):.3f}")
                        except Exception as e:
                            self.log_message(f"❌ {level_name} T2SI计算失败: {str(e)}")
                            result[f't2si_{level_name}_error'] = str(e)
            
                if self.enable_other_fd.get():
                    try:
                        self.log_message("计算分形维度...")
                        for level_name, labels in self.config.DISC_LABELS.items():
                            self.log_message(f"  -> 处理 {level_name} FD...")

                            disc_masks_level = []
                            for mask_slice in mask_slices:
                                disc_mask = (mask_slice == labels['disc']).astype(np.uint8)
                                disc_masks_level.append(disc_mask)
                            
                            if not any(np.any(mask) for mask in disc_masks_level):
                                self.log_message(f"  -> 在 {level_name} 未找到掩码，跳过")
                                continue

                            fd_slices = []
                            fd_masks = []
                            
                            for i, img_slice in enumerate(image_slices):
                                slice_spacing = spacing[:2] + [1.0]
                                edges, processed_mask = self.preprocessor.preprocess_for_fractal(
                                    img_slice, disc_masks_level[i], slice_spacing
                                )
                                fd_slices.append(edges)
                                fd_masks.append(processed_mask)
                            
                            fd_result = self.fd_calculator.process_multi_slice(
                                fd_slices, fd_masks
                            )
                            result.update({f'fd_{level_name}_{k}': v for k, v in fd_result.items()})
                            self.log_message(f"  -> {level_name} FD = {fd_result.get('fd', 'N/A'):.3f}")

                    except Exception as e:
                        self.log_message(f"❌ FD计算失败: {str(e)}")
                        result['fd_error'] = str(e)

                if self.enable_other_gabor.get():
                    try:
                        self.log_message("计算Gabor特征...")
                        for level_name, labels in self.config.DISC_LABELS.items():
                            self.log_message(f"  -> 处理 {level_name} Gabor...")

                            disc_masks_level = []
                            for mask_slice in mask_slices:
                                disc_mask = (mask_slice == labels['disc']).astype(np.uint8)
                                disc_masks_level.append(disc_mask)

                            if not any(np.any(mask) for mask in disc_masks_level):
                                self.log_message(f"  -> 在 {level_name} 未找到掩码，跳过")
                                continue

                            gabor_slices_level = []
                            gabor_masks_level = []

                            for i, img_slice in enumerate(image_slices):
                                slice_spacing = spacing[:2] + [1.0]
                                processed_img, processed_mask = self.preprocessor.preprocess_for_texture(
                                    img_slice, disc_masks_level[i], slice_spacing
                                )
                                gabor_slices_level.append(processed_img)
                                gabor_masks_level.append(processed_mask)

                            gabor_features = {}
                            for i, (img, mask) in enumerate(zip(gabor_slices_level, gabor_masks_level)):
                                if not np.any(mask): continue 
                                slice_features = self.gabor_calculator.calculate(img, mask)
                                for k, v in slice_features.items():
                                    if k in gabor_features:
                                        gabor_features[k].append(v)
                                    else:
                                        gabor_features[k] = [v]

                            if not gabor_features: continue

                            gabor_result = {k: np.mean(v) for k, v in gabor_features.items()}
                            result.update({f'gabor_{level_name}_{k}': v for k, v in gabor_result.items()})
                            self.log_message(f"  -> {level_name} 提取了 {len(gabor_result)} 个Gabor特征")

                    except Exception as e:
                        self.log_message(f"❌ Gabor计算失败: {str(e)}")
                        result['gabor_error'] = str(e)

                if self.enable_other_hu.get():
                    try:
                        self.log_message("计算Hu不变矩...")
                        for level_name, labels in self.config.DISC_LABELS.items():
                            self.log_message(f"  -> 处理 {level_name} Hu矩...")

                            disc_masks_level = []
                            for mask_slice in mask_slices:
                                disc_mask = (mask_slice == labels['disc']).astype(np.uint8)
                                disc_masks_level.append(disc_mask)

                            if not any(np.any(mask) for mask in disc_masks_level):
                                self.log_message(f"  -> 在 {level_name} 未找到掩码，跳过")
                                continue

                            hu_masks_level = []
                            for roi_mask in disc_masks_level:
                                slice_spacing = spacing[:2] + [1.0]
                                binary_mask = self.preprocessor.preprocess_for_shape(
                                    roi_mask, slice_spacing
                                )
                                hu_masks_level.append(binary_mask)

                            hu_features = {}
                            for i, mask in enumerate(hu_masks_level):
                                if not np.any(mask): continue
                                slice_features = self.hu_calculator.calculate(mask, mask)
                                for k, v in slice_features.items():
                                    if k in hu_features:
                                        hu_features[k].append(v)
                                    else:
                                        hu_features[k] = [v]
                            
                            if not hu_features: continue

                            hu_result = {k: np.mean(v) for k, v in hu_features.items()}
                            result.update({f'hu_{level_name}_{k}': v for k, v in hu_result.items()})
                            self.log_message(f"  -> {level_name} 提取了 {len(hu_result)} 个Hu矩特征")

                    except Exception as e:
                        self.log_message(f"❌ Hu矩计算失败: {str(e)}")
                        result['hu_error'] = str(e)

                if self.enable_other_texture.get():
                    try:
                        self.log_message("计算扩展纹理特征 (按椎间盘级别)...")
                        for level_name, labels in self.config.DISC_LABELS.items():
                            self.log_message(f"  -> 处理 {level_name} 扩展纹理...")

                            disc_masks_level = []
                            for mask_slice in mask_slices:
                                disc_mask = (mask_slice == labels['disc']).astype(np.uint8)
                                disc_masks_level.append(disc_mask)

                            if not any(np.any(mask) for mask in disc_masks_level):
                                self.log_message(f"  -> 在 {level_name} 未找到掩码，跳过")
                                continue

                            texture_slices_level = []
                            texture_masks_level = []
                            for i, img_slice in enumerate(image_slices):
                                slice_spacing = spacing[:2] + [1.0]
                                processed_img, processed_mask = self.preprocessor.preprocess_for_texture(
                                    img_slice, disc_masks_level[i], slice_spacing
                                )
                                texture_slices_level.append(processed_img)
                                texture_masks_level.append(processed_mask)

                            texture_features = {}
                            for i, (img, mask) in enumerate(zip(texture_slices_level, texture_masks_level)):
                                if not np.any(mask): continue
                                slice_features = self.texture_calculator.calculate(img, mask)
                                for k, v in slice_features.items():
                                    if k in texture_features:
                                        texture_features[k].append(v)
                                    else:
                                        texture_features[k] = [v]
                            
                            if not texture_features: continue

                            texture_result = {k: np.mean(v) for k, v in texture_features.items()}
                            result.update({f'texture_{level_name}_{k}': v for k, v in texture_result.items()})
                            self.log_message(f"  -> {level_name} 提取了 {len(texture_result)} 个扩展纹理特征")

                    except Exception as e:
                        self.log_message(f"❌ 扩展纹理特征计算失败: {str(e)}")
                        result['texture_error'] = str(e)


                if self.enable_other_dscr.get():
                    try:
                        self.log_message("计算椎管狭窄率DSCR...")
                        
                        dural_sac_masks = []
                        landmark_masks = []
                        
                        for mask_slice in mask_slices:
                            dural_sac_mask = (mask_slice == self.dural_sac_label.get()).astype(np.uint8)
                            dural_sac_masks.append(dural_sac_mask)
                            
                            landmark_mask = np.zeros_like(mask_slice, dtype=np.uint8)
                            if hasattr(self.config, 'LANDMARK_LABELS'):
                                for landmark_name, landmark_label in self.config.LANDMARK_LABELS.items():
                                    landmark_mask |= (mask_slice == landmark_label).astype(np.uint8)
                            else:
                                for label_value in [31, 32, 33, 34, 35, 36]:
                                    landmark_mask |= (mask_slice == label_value).astype(np.uint8)
                            
                            landmark_masks.append(landmark_mask)
                        
                        if not any(np.any(mask) for mask in dural_sac_masks):
                            self.log_message("⚠️ 没有找到硬脊膜囊区域")
                            result['dscr_note'] = "需要硬脊膜囊标注"
                        elif not any(np.any(mask) for mask in landmark_masks):
                            self.log_message("⚠️ 没有找到地标点")
                            result['dscr_note'] = "需要椎体地标点标注"
                        else:
                            for level_name in self.config.DISC_LABELS.keys():
                                disc_masks_for_dscr = []
                                for mask_slice in mask_slices:
                                    disc_mask = (mask_slice == self.config.DISC_LABELS[level_name]['disc']).astype(np.uint8)
                                    disc_masks_for_dscr.append(disc_mask)
                                
                                if any(np.any(mask) for mask in disc_masks_for_dscr):
                                    dscr_result = self.dscr_calculator.process_multi_slice(
                                        disc_masks_for_dscr, dural_sac_masks, landmark_masks, level_name
                                    )
                                    
                                    for key, value in dscr_result.items():
                                        result[f'dscr_{level_name}_{key}'] = value
                                    
                                    self.log_message(f"{level_name} DSCR = {dscr_result.get('dscr', 'N/A'):.1f}%")
                            
                    except Exception as e:
                        self.log_message(f"❌ DSCR计算失败: {str(e)}")
                        result['dscr_error'] = str(e)
                
                result['status'] = 'success'
                results = {'results': [result]}

            else:
                input_dir = self.input_path.get()
                mask_dir = self.mask_path.get()
                
                if not input_dir or not mask_dir:
                    raise ValueError("请选择输入文件夹和掩码文件夹")
                
                self.log_message("开始批量处理...")
                
                batch_results = []
                
                for idx, (case_id, image_path, mask_path, rel_path) in enumerate(matched_pairs):
                    self.log_message(f"\n处理病例 {idx+1}/{len(matched_pairs)}: {case_id}")

                    try:
                        self.log_message("加载图像和掩码...")

                        image, mask = self.image_io.load_image_and_mask(image_path, mask_path)

                        spacing = list(image.GetSpacing())[::-1]
                        self.log_message(f"图像间距: {spacing}")

                        image_array = self.image_io.sitk_to_numpy(image)
                        mask_array = self.image_io.sitk_to_numpy(mask)


                        result = {
                            'case_id': case_id,
                            'image_path': image_path,
                            'mask_path': mask_path,
                            'relative_path': rel_path
                        }

                        image_slices = self._extract_middle_slices(
                            image_array, self.config.NUM_SLICES, self.config.SLICE_AXIS
                        )
                        mask_slices = self._extract_middle_slices(
                            mask_array, self.config.NUM_SLICES, self.config.SLICE_AXIS
                        )

                        roi_masks = []
                        for mask_slice in mask_slices:
                            roi_mask = (mask_slice > 0).astype(np.uint8)
                            roi_masks.append(roi_mask)

                        if not any(np.any(mask) for mask in roi_masks):
                            self.log_message("⚠️ 没有找到有效的ROI区域")
                            result['status'] = 'no_roi_found'
                            return {'results': [result]}

                        csf_masks = []
                        if self.enable_other_asi.get() or self.enable_other_t2si.get():
                            csf_label = self.dural_sac_label.get()
                            for mask_slice in mask_slices:
                                csf_mask = (mask_slice == csf_label).astype(np.uint8)
                                csf_masks.append(csf_mask)

                            if not any(np.any(mask) for mask in csf_masks):
                                self.log_message("⚠️ 没有找到CSF区域，ASI/T2SI将使用全局标准化")
                                csf_masks = None

                        if self.enable_other_dhi.get():
                            try:
                                self.log_message("计算DHI...")
                                
                                dhi_results_all_levels = []
                                
                                for level_name, labels in self.config.DISC_LABELS.items():
                                    self.log_message(f"处理{level_name}层级")
                                    
                                    upper_masks = []
                                    disc_masks_level = []
                                    lower_masks = []
                                    
                                    for mask_slice in mask_slices:
                                        upper_mask = (mask_slice == labels['upper']).astype(np.uint8)
                                        upper_masks.append(upper_mask)
                                        
                                        disc_mask = (mask_slice == labels['disc']).astype(np.uint8)
                                        disc_masks_level.append(disc_mask)
                                        
                                        lower_mask = (mask_slice == labels['lower']).astype(np.uint8)
                                        lower_masks.append(lower_mask)
                                    
                                    if not any(np.any(mask) for mask in disc_masks_level):
                                        self.log_message(f"{level_name}层级没有找到椎间盘区域，跳过")
                                        continue
                                        
                                    is_l5_s1 = (level_name == 'L5-S1')
                                    dhi_result = self.dhi_calculator.process_multi_slice(
                                        upper_masks, disc_masks_level, lower_masks, is_l5_s1
                                    )
                                    
                                    for key, value in dhi_result.items():
                                        result[f'dhi_{level_name}_{key}'] = value
                                    
                                    self.log_message(f"{level_name} DHI = {dhi_result.get('dhi', 'N/A'):.3f}")
                                    dhi_results_all_levels.append(dhi_result)
                                
                                if dhi_results_all_levels:
                                    avg_dhi = np.mean([r['dhi'] for r in dhi_results_all_levels])
                                    result['dhi_average'] = avg_dhi
                                    self.log_message(f"平均DHI = {avg_dhi:.3f}")
                                
                            except Exception as e:
                                self.log_message(f"❌ DHI计算失败: {str(e)}")
                                result['dhi_error'] = str(e)

                        processed_image_slices_for_si = None
                        processed_mask_slices_for_si = None

                        if self.enable_other_asi.get() or self.enable_other_t2si.get():

                            processed_image_slices_for_si = []
                            processed_mask_slices_for_si = []
                            
                            for i, img_slice in enumerate(image_slices):
                                slice_spacing = spacing[:2] + [1.0]
                                
                                processed_img, processed_mask = self.preprocessor.preprocess_for_signal_intensity(
                                    img_slice, mask_slices[i], slice_spacing
                                )
                                unique_labels = np.unique(processed_mask)


                                processed_image_slices_for_si.append(processed_img)
                                processed_mask_slices_for_si.append(processed_mask.astype(np.int32))


                        if self.enable_other_asi.get() and processed_image_slices_for_si:
                            self.log_message("计算ASI...")
                            for level_name, labels in self.config.DISC_LABELS.items():
                                disc_masks_level = [(p_mask.astype(np.int32) == int(labels['disc'])).astype(np.uint8) for p_mask in processed_mask_slices_for_si]
                                
                                if not any(np.any(mask) for mask in disc_masks_level):
                                    continue
                                    
                                csf_label = self.dural_sac_label.get()
                                csf_label_int = int(csf_label)
                                csf_masks_level = [(p_mask.astype(np.int32) == csf_label_int).astype(np.uint8) for p_mask in processed_mask_slices_for_si]
                                
                                try:
                                    self.log_message(f"  -> 处理 {level_name} ASI...")
                                    asi_result = self.asi_calculator.process_multi_slice(
                                        processed_image_slices_for_si, disc_masks_level, csf_masks_level
                                    )
                                    result.update({f'asi_{level_name}_{k}': v for k, v in asi_result.items()})
                                    self.log_message(f"  -> {level_name} ASI = {asi_result.get('asi', 'N/A'):3f}")
                                except Exception as e:
                                    self.log_message(f"❌ {level_name} ASI计算失败: {str(e)}")
                                    result[f'asi_{level_name}_error'] = str(e)

                        if self.enable_other_t2si.get() and processed_image_slices_for_si:
                            self.log_message("计算T2信号强度...")
                            for level_name, labels in self.config.DISC_LABELS.items():
                                disc_masks_level = [(p_mask.astype(np.int32) == int(labels['disc'])).astype(np.uint8) for p_mask in processed_mask_slices_for_si]
                                
                                if not any(np.any(mask) for mask in disc_masks_level):
                                    continue
                                    
                                csf_label = self.dural_sac_label.get()
                                csf_label_int = int(csf_label)
                                csf_masks_level = [(p_mask.astype(np.int32) == csf_label_int).astype(np.uint8) for p_mask in processed_mask_slices_for_si]
                                
                                try:
                                    self.log_message(f"  -> 处理 {level_name} T2SI...")
                                    t2si_result = self.t2si_calculator.process_multi_slice(
                                        processed_image_slices_for_si, 
                                        np_masks_level,
                                        csf_masks_level
                                    )

                                    serializable_t2si_result = {k: v for k, v in t2si_result.items() if k != 'slice_results'}
                                    result.update({f't2si_{level_name}_{k}': v for k, v in serializable_t2si_result.items()})
                                    self.log_message(f"  -> {level_name} T2SI比率 = {t2si_result.get('si_ratio', 'N/A'):.3f}")
                                except Exception as e:
                                    self.log_message(f"❌ {level_name} T2SI计算失败: {str(e)}")
                                    result[f't2si_{level_name}_error'] = str(e)
                        
                        if self.enable_other_fd.get():
                            try:
                                self.log_message("计算分形维度...")
                                for level_name, labels in self.config.DISC_LABELS.items():
                                    self.log_message(f"  -> 处理 {level_name} FD...")

                                    disc_masks_level = []
                                    for mask_slice in mask_slices:
                                        disc_mask = (mask_slice == labels['disc']).astype(np.uint8)
                                        disc_masks_level.append(disc_mask)
                                    
                                    if not any(np.any(mask) for mask in disc_masks_level):
                                        self.log_message(f"  -> 在 {level_name} 未找到掩码，跳过")
                                        continue

                                    fd_slices = []
                                    fd_masks = []
                                    
                                    for i, img_slice in enumerate(image_slices):
                                        slice_spacing = spacing[:2] + [1.0]
                                        edges, processed_mask = self.preprocessor.preprocess_for_fractal(
                                            img_slice, disc_masks_level[i], slice_spacing
                                        )
                                        fd_slices.append(edges)
                                        fd_masks.append(processed_mask)
                                    
                                    fd_result = self.fd_calculator.process_multi_slice(
                                        fd_slices, fd_masks
                                    )
                                    result.update({f'fd_{level_name}_{k}': v for k, v in fd_result.items()})
                                    self.log_message(f"  -> {level_name} FD = {fd_result.get('fd', 'N/A'):.3f}")

                            except Exception as e:
                                self.log_message(f"❌ FD计算失败: {str(e)}")
                                result['fd_error'] = str(e)

                        if self.enable_other_gabor.get():
                            try:
                                self.log_message("计算Gabor特征...")
                                for level_name, labels in self.config.DISC_LABELS.items():
                                    self.log_message(f"  -> 处理 {level_name} Gabor...")

                                    disc_masks_level = []
                                    for mask_slice in mask_slices:
                                        disc_mask = (mask_slice == labels['disc']).astype(np.uint8)
                                        disc_masks_level.append(disc_mask)

                                    if not any(np.any(mask) for mask in disc_masks_level):
                                        self.log_message(f"  -> 在 {level_name} 未找到掩码，跳过")
                                        continue

                                    gabor_slices_level = []
                                    gabor_masks_level = []

                                    for i, img_slice in enumerate(image_slices):
                                        slice_spacing = spacing[:2] + [1.0]
                                        processed_img, processed_mask = self.preprocessor.preprocess_for_texture(
                                            img_slice, disc_masks_level[i], slice_spacing
                                        )
                                        gabor_slices_level.append(processed_img)
                                        gabor_masks_level.append(processed_mask)

                                    gabor_features = {}
                                    for i, (img, mask) in enumerate(zip(gabor_slices_level, gabor_masks_level)):
                                        if not np.any(mask): continue 
                                        slice_features = self.gabor_calculator.calculate(img, mask)
                                        for k, v in slice_features.items():
                                            if k in gabor_features:
                                                gabor_features[k].append(v)
                                            else:
                                                gabor_features[k] = [v]

                                    if not gabor_features: continue

                                    gabor_result = {k: np.mean(v) for k, v in gabor_features.items()}
                                    result.update({f'gabor_{level_name}_{k}': v for k, v in gabor_result.items()})
                                    self.log_message(f"  -> {level_name} 提取了 {len(gabor_result)} 个Gabor特征")

                            except Exception as e:
                                self.log_message(f"❌ Gabor计算失败: {str(e)}")
                                result['gabor_error'] = str(e)

                        if self.enable_other_hu.get():
                            try:
                                self.log_message("计算Hu不变矩...")
                                for level_name, labels in self.config.DISC_LABELS.items():
                                    self.log_message(f"  -> 处理 {level_name} Hu矩...")

                                    disc_masks_level = []
                                    for mask_slice in mask_slices:
                                        disc_mask = (mask_slice == labels['disc']).astype(np.uint8)
                                        disc_masks_level.append(disc_mask)

                                    if not any(np.any(mask) for mask in disc_masks_level):
                                        self.log_message(f"  -> 在 {level_name} 未找到掩码，跳过")
                                        continue

                                    hu_masks_level = []
                                    for roi_mask in disc_masks_level:
                                        slice_spacing = spacing[:2] + [1.0]
                                        binary_mask = self.preprocessor.preprocess_for_shape(
                                            roi_mask, slice_spacing
                                        )
                                        hu_masks_level.append(binary_mask)

                                    hu_features = {}
                                    for i, mask in enumerate(hu_masks_level):
                                        if not np.any(mask): continue
                                        slice_features = self.hu_calculator.calculate(mask, mask)
                                        for k, v in slice_features.items():
                                            if k in hu_features:
                                                hu_features[k].append(v)
                                            else:
                                                hu_features[k] = [v]
                                    
                                    if not hu_features: continue

                                    hu_result = {k: np.mean(v) for k, v in hu_features.items()}
                                    result.update({f'hu_{level_name}_{k}': v for k, v in hu_result.items()})
                                    self.log_message(f"  -> {level_name} 提取了 {len(hu_result)} 个Hu矩特征")

                            except Exception as e:
                                self.log_message(f"❌ Hu矩计算失败: {str(e)}")
                                result['hu_error'] = str(e)


                        if self.enable_other_texture.get():
                            try:
                                self.log_message("计算扩展纹理特征 (按椎间盘级别)...")
                                for level_name, labels in self.config.DISC_LABELS.items():
                                    self.log_message(f"  -> 处理 {level_name} 扩展纹理...")

                                    disc_masks_level = []
                                    for mask_slice in mask_slices:
                                        disc_mask = (mask_slice == labels['disc']).astype(np.uint8)
                                        disc_masks_level.append(disc_mask)

                                    if not any(np.any(mask) for mask in disc_masks_level):
                                        self.log_message(f"  -> 在 {level_name} 未找到掩码，跳过")
                                        continue

                                    texture_slices_level = []
                                    texture_masks_level = []
                                    for i, img_slice in enumerate(image_slices):
                                        slice_spacing = spacing[:2] + [1.0]
                                        processed_img, processed_mask = self.preprocessor.preprocess_for_texture(
                                            img_slice, disc_masks_level[i], slice_spacing
                                        )
                                        texture_slices_level.append(processed_img)
                                        texture_masks_level.append(processed_mask)

                                    texture_features = {}
                                    for i, (img, mask) in enumerate(zip(texture_slices_level, texture_masks_level)):
                                        if not np.any(mask): continue
                                        slice_features = self.texture_calculator.calculate(img, mask)
                                        for k, v in slice_features.items():
                                            if k in texture_features:
                                                texture_features[k].append(v)
                                            else:
                                                texture_features[k] = [v]
                                    
                                    if not texture_features: continue

                                    texture_result = {k: np.mean(v) for k, v in texture_features.items()}
                                    result.update({f'texture_{level_name}_{k}': v for k, v in texture_result.items()})
                                    self.log_message(f"  -> {level_name} 提取了 {len(texture_result)} 个扩展纹理特征")

                            except Exception as e:
                                self.log_message(f"❌ 扩展纹理特征计算失败: {str(e)}")
                                result['texture_error'] = str(e)

                        if self.enable_other_dscr.get():
                            try:
                                self.log_message("计算椎管狭窄率DSCR...")
                                
                                dural_sac_masks = []
                                landmark_masks = []
                                
                                for mask_slice in mask_slices:
                                    dural_sac_mask = (mask_slice == self.dural_sac_label.get()).astype(np.uint8)
                                    dural_sac_masks.append(dural_sac_mask)
                                    
                                    landmark_mask = np.zeros_like(mask_slice, dtype=np.uint8)
                                    if hasattr(self.config, 'LANDMARK_LABELS'):
                                        for landmark_name, landmark_label in self.config.LANDMARK_LABELS.items():
                                            landmark_mask |= (mask_slice == landmark_label).astype(np.uint8)
                                    else:
                                        for label_value in [31, 32, 33, 34, 35, 36]:
                                            landmark_mask |= (mask_slice == label_value).astype(np.uint8)
                                    
                                    landmark_masks.append(landmark_mask)
                                
                                if not any(np.any(mask) for mask in dural_sac_masks):
                                    self.log_message("⚠️ 没有找到硬脊膜囊区域")
                                    result['dscr_note'] = "需要硬脊膜囊标注"
                                elif not any(np.any(mask) for mask in landmark_masks):
                                    self.log_message("⚠️ 没有找到地标点")
                                    result['dscr_note'] = "需要椎体地标点标注"
                                else:
                                    for level_name in self.config.DISC_LABELS.keys():
                                        disc_masks_for_dscr = []
                                        for mask_slice in mask_slices:
                                            disc_mask = (mask_slice == self.config.DISC_LABELS[level_name]['disc']).astype(np.uint8)
                                            disc_masks_for_dscr.append(disc_mask)
                                        
                                        if any(np.any(mask) for mask in disc_masks_for_dscr):
                                            dscr_result = self.dscr_calculator.process_multi_slice(
                                                disc_masks_for_dscr, dural_sac_masks, landmark_masks, level_name
                                            )
                                            
                                            for key, value in dscr_result.items():
                                                result[f'dscr_{level_name}_{key}'] = value
                                            
                                            self.log_message(f"{level_name} DSCR = {dscr_result.get('dscr', 'N/A'):.1f}%")
                                    
                            except Exception as e:
                                self.log_message(f"❌ DSCR计算失败: {str(e)}")
                                result['dscr_error'] = str(e)

                        
                        result['status'] = 'success'
                        batch_results.append(result)

                    except Exception as e:
                        self.log_message(f"❌ 处理失败: {str(e)}")
                        batch_results.append({
                            'case_id': case_id,
                            'status': 'failed',
                            'error': str(e),
                            'relative_path': rel_path
                        })
                
                results = {
                    'batch_mode': True,
                    'total_cases': len(matched_pairs),
                    'results': batch_results
                }
            
            return results
            
        except Exception as e:
            self.log_message(f"❌ 特征提取失败: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
            return {'error': str(e)}
        
    def run_extraction(self):
        try:
            self.log_message("🚀 开始特征提取...")
            
            source = self.feature_type.get()
            all_results = []
            matched_pairs = None

            if self.input_type.get() == "batch":
                input_dir = self.input_path.get()
                mask_dir = self.mask_path.get()
                self.log_message(f"扫描图像文件夹: {input_dir}")
                self.log_message(f"扫描掩码文件夹: {mask_dir}")
                
                image_files = self._scan_image_files(input_dir)
                mask_files = self._scan_mask_files(mask_dir)
                matched_pairs = self._match_files(image_files, mask_files, input_dir, mask_dir)

                if not matched_pairs:
                    self.log_message("⚠️ 未找到任何匹配的 图像/掩码 文件对。处理中止。")
                    messagebox.showwarning("警告", "未找到任何匹配的 图像/掩码 文件对。")
                    return
                self.log_message(f"成功匹配 {len(matched_pairs)} 对文件。")

            if source in ["other", "both"]:
                self.log_message(self.get_text('processing_other'))
                other_results = self.extract_other_features(matched_pairs)
                
                if 'error' in other_results:
                    self.log_message(f"❌ 自定义特征提取失败: {other_results['error']}")
                    if source == "other": 
                        messagebox.showerror("错误", f"特征提取失败: {other_results['error']}")
                        return
                elif 'results' in other_results:
                    all_results.extend(other_results['results'])
                    self.log_message(self.get_text('other_complete'))

            if source in ["pyradiomics", "both"] and PYRADIOMICS_AVAILABLE:
                self.log_message("🔄 处理PyRadiomics特征...")
                pyrad_results = self.extract_pyradiomics_features(matched_pairs)
                
                if 'error' in pyrad_results:
                    self.log_message(f"❌ PyRadiomics特征提取失败: {pyrad_results['error']}")
                    if source == "pyradiomics":  
                        messagebox.showerror("错误", f"特征提取失败: {pyrad_results['error']}")
                        return
                elif 'results' in pyrad_results:
                    if source == "both" and all_results:
                        self._merge_results(all_results, pyrad_results['results'])
                    else:
                        all_results.extend(pyrad_results['results'])
                    self.log_message("✅ PyRadiomics特征提取完成")

            if all_results:
                is_batch = self.input_type.get() == "batch"
                self.save_results({'results': all_results, 'batch_mode': is_batch, 'total_cases': len(all_results)})
                self.log_message("🎉 所有特征提取完成！")
            else:
                self.log_message("⚠️ 没有提取到任何特征")
                messagebox.showwarning("警告", "没有提取到任何特征")
                
        except Exception as e:
            self.log_message(f"❌ 错误: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
            messagebox.showerror("错误", f"特征提取失败: {str(e)}")

    def _merge_results(self, other_results: List[Dict], pyrad_results: List[Dict]):
        other_dict = {r['case_id']: r for r in other_results}
        
        for pyrad_result in pyrad_results:
            case_id = pyrad_result['case_id']
            if case_id in other_dict:
                other_dict[case_id].update({
                    k: v for k, v in pyrad_result.items() 
                    if k not in ['case_id', 'image_path', 'mask_path', 'status']
                })
                if 'num_features' in pyrad_result:
                    other_dict[case_id]['num_pyradiomics_features'] = pyrad_result['num_features']
            else:
                other_results.append(pyrad_result)

    def _convert_to_long_format(self, wide_df: pd.DataFrame) -> pd.DataFrame:

        long_results = []
        
        for _, row in wide_df.iterrows():
            case_id = row.get('case_id', 'unknown')
            
            for level_name in self.config.DISC_LABELS.keys():
                level_row = {
                    'Sample_ID': case_id,
                    'Disc_Level': level_name
                }
                
                for col in row.index:
                    if col == 'case_id' or col == 'status':
                        continue
                        
                    if level_name in col:
                        feature_name = col.replace(f'_{level_name}', '').replace(f'{level_name}_', '')
                        level_row[feature_name] = row[col]
                    elif not any(lvl in col for lvl in self.config.DISC_LABELS.keys()):
                        level_row[col] = row[col]
                
                long_results.append(level_row)
        
        return pd.DataFrame(long_results)

    def save_results(self, results):
        output_path = self.output_path.get()
        
        if not output_path:
            messagebox.showerror("错误", "请选择输出路径")
            return
        
        if 'error' in results:
            messagebox.showerror("错误", f"无法保存结果: {results['error']}")
            return
        
        if 'results' not in results or not results['results']:
            messagebox.showwarning("警告", "没有结果可以保存")
            return
        
        try:
            df = pd.DataFrame(results['results'])

            cols_to_remove = [
                'image_path', 'mask_path', 'relative_path', 'status',
                'num_features', 'num_pyradiomics_features', 'diagnostics',
                't2si_roi_method', 'fd_r_squared', 't2si_num_slices',
                't2si_mean_roi_size'
            ]
            
            for col in df.columns:
                if (col.endswith('_error') or 
                    col.endswith('_note') or 
                    col.endswith('_valid_slices')):
                    if col not in cols_to_remove:
                        cols_to_remove.append(col)

            df_features_only = df.drop(columns=cols_to_remove, errors='ignore')


            if hasattr(self, 'output_format') and self.output_format.get() == 'long':
                df_features_only = self._convert_to_long_format(df_features_only)

            if not os.path.isdir(output_path):
                output_dir = os.path.dirname(output_path)
            else:
                output_dir = output_path

            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            new_filename = f"features_{timestamp}.csv"

            final_csv_path = os.path.join(output_dir, new_filename)

            df_features_only.to_csv(final_csv_path, index=False)

            self.log_message(f"💾 结果已保存到: {final_csv_path}")

            messagebox.showinfo("完成", "特征提取和保存完成！")
            
        except Exception as e:
            self.log_message(f"❌ 保存结果失败: {str(e)}")
            messagebox.showerror("错误", f"保存结果失败: {str(e)}")

    def _generate_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:

        stats = []

        feature_types = {
            'DHI特征': [col for col in df.columns if col.startswith('dhi_')],
            'ASI特征': [col for col in df.columns if col.startswith('asi_')],
            'T2SI特征': [col for col in df.columns if col.startswith('t2si_')],
            'FD特征': [col for col in df.columns if col.startswith('fd_')],
            'Gabor特征': [col for col in df.columns if col.startswith('gabor_')],
            'Hu矩特征': [col for col in df.columns if col.startswith('hu_')],
            '纹理特征': [col for col in df.columns if col.startswith('texture_')],
            'PyRadiomics特征': [col for col in df.columns if not any(col.startswith(prefix) 
                            for prefix in ['dhi_', 'asi_', 't2si_', 'fd_', 'gabor_', 'hu_', 'texture_', 'case_', 'status', 'image_', 'mask_'])]
        }
        
        for feature_type, columns in feature_types.items():
            if columns:
                stats.append({
                    '特征类型': feature_type,
                    '特征数量': len(columns),
                    '示例特征': ', '.join(columns[:3]) + ('...' if len(columns) > 3 else '')
                })
        
        return pd.DataFrame(stats) if stats else None

    def _generate_batch_statistics(self, df: pd.DataFrame) -> pd.DataFrame:

        stats = []

        stats.append({
            '统计项': '总病例数',
            '数值': len(df)
        })
        
        if 'status' in df.columns:
            stats.append({
                '统计项': '成功病例数',
                '数值': len(df[df['status'] == 'success'])
            })
            stats.append({
                '统计项': '失败病例数',
                '数值': len(df[df['status'] == 'failed'])
            })

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        stats.append({
            '统计项': '总特征数',
            '数值': len(numeric_columns)
        })

        key_features = ['dhi_dhi', 'asi_asi', 'fd_fd', 't2si_si_ratio']
        for feature in key_features:
            if feature in df.columns:
                valid_values = df[feature].dropna()
                if len(valid_values) > 0:
                    stats.append({
                        '统计项': f'{feature}平均值',
                        '数值': f'{valid_values.mean():.3f} ± {valid_values.std():.3f}'
                    })
        
        return pd.DataFrame(stats) if stats else None

    def _extract_middle_slices(self, array: np.ndarray, num_slices: int, axis: int) -> List[np.ndarray]:
        size = array.shape[axis]
        middle = size // 2
        
        if num_slices % 2 == 0:
            start_idx = middle - num_slices // 2
            end_idx = middle + num_slices // 2
        else:
            start_idx = middle - num_slices // 2
            end_idx = middle + num_slices // 2 + 1
        
        start_idx = max(0, start_idx)
        end_idx = min(size, end_idx)
        
        slices = []
        for i in range(start_idx, end_idx):
            if axis == 0:
                slices.append(array[i, :, :])
            elif axis == 1:
                slices.append(array[:, i, :])
            else:
                slices.append(array[:, :, i])
        
        return slices

    def _calculate_asi_without_csf(self, image_slices: List[np.ndarray], 
                                roi_masks: List[np.ndarray]) -> Dict:

        all_pixels = []
        roi_pixels = []
        
        for img, mask in zip(image_slices, roi_masks):
            all_pixels.extend(img.flatten())
            roi_pixels.extend(img[mask > 0])
        
        global_mean = np.mean(all_pixels)
        global_std = np.std(all_pixels)

        result = {
            'asi': np.mean(roi_pixels) / global_mean if global_mean > 0 else 0,
            'peak_diff': np.std(roi_pixels),
            'csf_intensity': global_mean,  
            'note': 'No CSF region found, using global normalization'
        }
        
        return result

    def _calculate_t2si_global(self, image_slices: List[np.ndarray], 
                            roi_masks: List[np.ndarray]) -> Dict:

        roi_signals = []
        for img, mask in zip(image_slices, roi_masks):
            roi_pixels = img[mask > 0]
            if len(roi_pixels) > 0:
                roi_signals.append(np.mean(roi_pixels))
        
        roi_mean = np.mean(roi_signals) if roi_signals else 0

        all_pixels = []
        for img in image_slices:
            all_pixels.extend(img.flatten())
        
        ref_intensity = np.percentile(all_pixels, 95)
        
        result = {
            'roi_si': roi_mean,
            'csf_si': ref_intensity, 
            'si_ratio': roi_mean / ref_intensity if ref_intensity > 0 else 0,
            'roi_method': 'GLOBAL',
            'note': 'No CSF region found, using 95th percentile as reference'
        }
        
        return result
    
    def extract_pyradiomics_features(self, matched_pairs=None):
        if not PYRADIOMICS_AVAILABLE:
            self.log_message("❌ PyRadiomics不可用")
            return {'error': 'PyRadiomics not available'}
        
        results = {}
        
        try:
            if self.input_type.get() == "single":
                image_path = self.input_path.get()
                mask_path = self.mask_path.get()
                
                if not image_path or not mask_path:
                    raise ValueError("请选择图像和掩码文件")
                
                self.log_message("初始化PyRadiomics...")

                params = self._create_pyradiomics_params()

                extractor = featureextractor.RadiomicsFeatureExtractor(**params)

                if self.enable_shape.get():
                    extractor.enableFeatureClassByName('shape')
                if self.enable_shape2d.get():
                    extractor.enableFeatureClassByName('shape2D')
                if self.enable_firstorder.get():
                    extractor.enableFeatureClassByName('firstorder')
                if self.enable_glcm.get():
                    extractor.enableFeatureClassByName('glcm')
                if self.enable_glrlm.get():
                    extractor.enableFeatureClassByName('glrlm')
                if self.enable_glszm.get():
                    extractor.enableFeatureClassByName('glszm')
                if self.enable_gldm.get():
                    extractor.enableFeatureClassByName('gldm')
                if self.enable_ngtdm.get():
                    extractor.enableFeatureClassByName('ngtdm')

                if self.enable_log.get():
                    extractor.enableImageTypeByName('LoG')

                if self.enable_wavelet.get():
                    extractor.enableImageTypeByName('Wavelet')
                
                if self.enable_square.get():
                    extractor.enableImageTypeByName('Square')
                if self.enable_square_root.get():
                    extractor.enableImageTypeByName('SquareRoot')
                if self.enable_logarithm.get():
                    extractor.enableImageTypeByName('Logarithm')
                if self.enable_exponential.get():
                    extractor.enableImageTypeByName('Exponential')
                if self.enable_gradient.get():
                    extractor.enableImageTypeByName('Gradient')
                
                if self.enable_lbp2d.get():
                    extractor.enableImageTypeByName('LBP2D')
                
                if self.enable_lbp3d.get():
                    extractor.enableImageTypeByName('LBP3D')
                
                self.log_message("提取PyRadiomics特征...")

                feature_vector = extractor.execute(image_path, mask_path, label=self.label.get())

                pyrad_features = {}
                diagnostics = {}
                
                for key, value in feature_vector.items():
                    if key.startswith('diagnostics_'):
                        if self.additional_info.get():
                            diagnostics[key] = value
                    elif not key.startswith('general_info_'):
                        if isinstance(value, (np.ndarray, np.generic)):
                            value = value.tolist() if isinstance(value, np.ndarray) else value.item()
                        pyrad_features[key] = value

                p = Path(image_path)
                base_name = p.name
                while Path(base_name).suffix:
                    base_name = Path(base_name).stem
                case_id = base_name
                
                result = {
                    'case_id': case_id,
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'num_features': len(pyrad_features),
                    'status': 'success'
                }
                result.update(pyrad_features)
                
                if self.additional_info.get():
                    result['diagnostics'] = diagnostics
                
                self.log_message(f"✅ 成功提取 {len(pyrad_features)} 个PyRadiomics特征")
                
                results = {'results': [result]}
                
            else: 
                input_dir = self.input_path.get()
                mask_dir = self.mask_path.get()
                
                if not input_dir or not mask_dir:
                    raise ValueError("请选择输入文件夹和掩码文件夹")
                
                self.log_message("开始PyRadiomics批量处理...")

                params = self._create_pyradiomics_params()
                extractor = featureextractor.RadiomicsFeatureExtractor(**params)
                self._configure_extractor(extractor)
                
                batch_results = []
                
                for idx, (case_id, image_path, mask_path, rel_path) in enumerate(matched_pairs):
                    self.log_message(f"\n处理病例 {idx+1}/{len(matched_pairs)}: {case_id}")
                    
                    try:
                        feature_vector = extractor.execute(image_path, mask_path, label=self.label.get())

                        pyrad_features = {}
                        for key, value in feature_vector.items():
                            if not key.startswith('diagnostics_') and not key.startswith('general_info_'):
                                if isinstance(value, (np.ndarray, np.generic)):
                                    value = value.tolist() if isinstance(value, np.ndarray) else value.item()
                                pyrad_features[key] = value
                        
                        result = {
                            'case_id': case_id,
                            'status': 'success',
                            'num_features': len(pyrad_features),
                            'relative_path': rel_path
                        }
                        result.update(pyrad_features)
                        
                        batch_results.append(result)
                        self.log_message(f"✅ 成功提取 {len(pyrad_features)} 个PyRadiomics特征")
                        
                    except Exception as e:
                        self.log_message(f"❌ 处理失败: {str(e)}")
                        batch_results.append({
                            'case_id': case_id,
                            'status': 'failed',
                            'error': str(e),
                            'relative_path': rel_path
                        })
                
                results = {
                    'batch_mode': True,
                    'total_cases': len(matched_pairs),
                    'results': batch_results
                }
            
            return results
            
        except Exception as e:
            self.log_message(f"❌ PyRadiomics特征提取失败: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
            return {'error': str(e)}

    def _create_pyradiomics_params(self):
            params = {
                'binWidth': self.bin_width.get() if self.bin_width.get() > 0 else None,
                'binCount': self.bin_count.get() if self.bin_count.get() > 0 else None,
                'normalize': self.normalize.get(),
                'normalizeScale': self.normalize_scale.get(),
                'removeOutliers': self.remove_outliers.get() if self.remove_outliers.get() > 0 else None,
                'correctMask': self.correct_mask.get(),
                'interpolator': self.interpolator.get(),
                'padDistance': self.pad_distance.get(),
                'geometryTolerance': float(self.geometry_tolerance.get()),
                'additionalInfo': self.additional_info.get(),
                'enableCExtensions': self.enable_c_extensions.get(),
                'minimumROIDimensions': self.minimum_roi_dimensions.get(),
                'minimumROISize': self.minimum_roi_size.get(),
                'preCrop': self.preCrop.get(),
                'voxelArrayShift': self.voxel_array_shift.get()
            }

            if self.resample_spacing.get():
                spacing = [float(s.strip()) for s in self.resample_spacing.get().split(',')]
                params['resampledPixelSpacing'] = spacing

            if self.force2D.get():
                params['force2D'] = True
                params['force2Ddimension'] = self.force2D_dimension.get()
                params['force2DExtraction'] = self.force2D_aggregator.get()

            if self.distances.get():
                distances = [int(d.strip()) for d in self.distances.get().split(',')]
                params['distances'] = distances
            
            params['weightingNorm'] = self.weighting_norm.get() if self.weighting_norm.get() != 'no_weighting' else None
            params['symmetricalGLCM'] = self.symmetrical_glcm.get()
            
            if self.gldm_a.get() > 0:
                params['gldm_a'] = self.gldm_a.get()

            if self.resegment_range.get():
                try:
                    range_values = [float(v.strip()) for v in self.resegment_range.get().split(',')]
                    params['resegmentRange'] = range_values
                    params['resegmentMode'] = self.resegment_mode.get()
                    params['resegmentShape'] = self.resegment_shape.get()
                except:
                    self.log_message("⚠️ 重分割范围格式错误，跳过重分割设置")

            if hasattr(self, 'kernel_radius'):
                params['kernelRadius'] = self.kernel_radius.get()
                params['maskedKernel'] = self.masked_kernel.get()
                params['initValue'] = float(self.init_value.get()) if self.init_value.get() else 0
                params['voxelBatch'] = self.voxel_batch.get() if self.voxel_batch.get() > 0 else -1

            params['imageType'] = {}
            if self.enable_log.get():
                sigma_values = [float(s.strip()) for s in self.log_sigma.get().split(',')]
                params['imageType']['LoG'] = {'sigma': sigma_values}
            
            if self.enable_wavelet.get():
                params['imageType']['Wavelet'] = {
                    'level': self.wavelet_level.get(),
                    'start_level': self.wavelet_start_level.get(),
                    'wavelet': self.wavelet_type.get()
                }

            if self.enable_square.get():
                params['imageType']['Square'] = {}
            if self.enable_square_root.get():
                params['imageType']['SquareRoot'] = {}
            if self.enable_logarithm.get():
                params['imageType']['Logarithm'] = {}
            if self.enable_exponential.get():
                params['imageType']['Exponential'] = {}
            
            if self.enable_gradient.get():
                params['imageType']['Gradient'] = {'gradientUsingSpacing': self.gradient_sigma.get()}
            
            if self.enable_lbp2d.get():
                params['imageType']['LBP2D'] = {
                    'radius': self.lbp2d_radius.get(),
                    'samples': self.lbp2d_samples.get(),
                    'method': self.lbp2d_method.get()
                }
            
            if self.enable_lbp3d.get():
                params['imageType']['LBP3D'] = {
                    'levels': self.lbp3d_levels.get(),
                    'icosphereRadius': self.lbp3d_icosphere_radius.get(),
                    'icosphereSubdivision': self.lbp3d_icosphere_subdivision.get()
                }
            
            params = {k: v for k, v in params.items() if v is not None}
            
            return params

    def _configure_extractor(self, extractor):
        if self.enable_shape.get():
            extractor.enableFeatureClassByName('shape')
        if self.enable_shape2d.get():
            extractor.enableFeatureClassByName('shape2D')
        if self.enable_firstorder.get():
            extractor.enableFeatureClassByName('firstorder')
        if self.enable_glcm.get():
            extractor.enableFeatureClassByName('glcm')
        if self.enable_glrlm.get():
            extractor.enableFeatureClassByName('glrlm')
        if self.enable_glszm.get():
            extractor.enableFeatureClassByName('glszm')
        if self.enable_gldm.get():
            extractor.enableFeatureClassByName('gldm')
        if self.enable_ngtdm.get():
            extractor.enableFeatureClassByName('ngtdm')

        if self.enable_log.get():
            extractor.enableImageTypeByName('LoG')

        if self.enable_wavelet.get():
            extractor.enableImageTypeByName('Wavelet')
        
        if self.enable_square.get():
            extractor.enableImageTypeByName('Square')
        if self.enable_square_root.get():
            extractor.enableImageTypeByName('SquareRoot')
        if self.enable_logarithm.get():
            extractor.enableImageTypeByName('Logarithm')
        if self.enable_exponential.get():
            extractor.enableImageTypeByName('Exponential')
        if self.enable_gradient.get():
            extractor.enableImageTypeByName('Gradient')
        
        if self.enable_lbp2d.get():
            extractor.enableImageTypeByName('LBP2D')
        
        if self.enable_lbp3d.get():
            extractor.enableImageTypeByName('LBP3D')
    

    def select_input(self):
        if self.input_type.get() == "single":
            path = filedialog.askopenfilename(
                title=self.get_text('select_input_file'),
                filetypes=[
                    ("NIfTI文件", "*.nii *.nii.gz"),
                    ("DICOM文件", "*.dcm"),
                    ("NRRD文件", "*.nrrd"),
                    ("MHA/MHD文件", "*.mha *.mhd"),
                    ("所有支持格式", "*.dcm *.nii *.nii.gz *.nrrd *.mha *.mhd"),
                    ("所有文件", "*.*")
                ]
            )
        else:

            path = filedialog.askdirectory(
                title="选择图像文件夹"
            )
        
        if path:
            self.input_path.set(path)

    def select_mask(self):
        if self.input_type.get() == "single":
            path = filedialog.askopenfilename(
                title=self.get_text('select_mask_file'),
                filetypes=[
                    ("NIfTI文件", "*.nii *.nii.gz"),
                    ("DICOM文件", "*.dcm"),
                    ("NRRD文件", "*.nrrd"),
                    ("MHA/MHD文件", "*.mha *.mhd"),
                    ("所有支持格式", "*.dcm *.nii *.nii.gz *.nrrd *.mha *.mhd"),
                    ("所有文件", "*.*")
                ]
            )
        else:
            path = filedialog.askdirectory(
                title="选择掩码文件夹"
            )
        if path:
            self.mask_path.set(path)

    def select_output(self):
        path = filedialog.askdirectory(title=self.get_text('save_results'))
        
        if path:
            self.output_path.set(path)

    def _scan_image_files(self, root_dir: str) -> List[Tuple[str, str]]:

        image_files = []
        root_path = Path(root_dir)

        supported_patterns = ['*.dcm', '*.nii', '*.nii.gz', '*.nrrd', '*.mha', '*.mhd']
        
        for pattern in supported_patterns:
            for file_path in root_path.rglob(pattern):
                rel_path = file_path.relative_to(root_path)
                image_files.append((str(file_path), str(rel_path.parent)))
        
        return sorted(image_files)

    def _scan_mask_files(self, root_dir: str) -> List[Tuple[str, str]]:

        mask_files = []
        root_path = Path(root_dir)

        supported_patterns = ['*.dcm', '*.nii', '*.nii.gz', '*.nrrd', '*.mha', '*.mhd']
        
        for pattern in supported_patterns:
            for file_path in root_path.rglob(pattern):
                rel_path = file_path.relative_to(root_path)
                mask_files.append((str(file_path), str(rel_path.parent)))
        
        return sorted(mask_files)

    def _match_files(self, image_files: List[Tuple[str, str]], 
                    mask_files: List[Tuple[str, str]],
                    input_dir: str, mask_dir: str) -> List[Tuple[str, str, str, str]]:

        matched_pairs = []
        
        mask_dict = {}
        for mask_path, mask_rel_path in mask_files:
            p = Path(mask_path)
            base_name = p.name
            while Path(base_name).suffix:
                base_name = Path(base_name).stem

            clean_base_name = base_name.replace('_mask', '').replace('_seg', '').replace('-mask', '').replace('-seg', '')
            
            key = f"{mask_rel_path}/{clean_base_name}"
            mask_dict[key] = mask_path

        for image_path, image_rel_path in image_files:
            p = Path(image_path)
            base_name = p.name
            while Path(base_name).suffix:
                base_name = Path(base_name).stem

            key = f"{image_rel_path}/{base_name}"
            
            if key in mask_dict:
                case_id = base_name
                matched_pairs.append((case_id, image_path, mask_dict[key], image_rel_path))

        return matched_pairs
    

if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedFeatureExtractorGUI(root)
    root.mainloop()
