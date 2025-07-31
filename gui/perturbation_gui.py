import sys
import os
import numpy as np
import cv2
from tkinter import ttk, filedialog, messagebox, scrolledtext
import tkinter as tk
from PyQt5.QtCore import QThread, pyqtSignal
import SimpleITK as sitk
import pydicom
from pathlib import Path
import json
from datetime import datetime

PERTURB_TEXT_DICT = {
    'cn': {
        'file_selection': '📁 文件选择',
        'process_mode': '处理模式:',
        'batch_mode': '📊 批量处理',
        'single_mode': '🔍 单个案例',
        'input_path': '输入路径:',
        'mask_path': '掩码路径:',
        'output_path': '输出路径:',
        'select': '选择',
        'perturbation_types': '🔧 扰动类型',
        'original': '原始',
        'dilation': '膨胀',
        'erosion': '腐蚀',
        'contour_random': '轮廓随机化',
        'translation': '平移',
        'rotation': '旋转',
        'gaussian_noise': '高斯噪声',
        'translation_rotation': '平移+旋转',
        'dilation_trans_rot': '膨胀+平移+旋转',
        'erosion_trans_rot': '腐蚀+平移+旋转',
        'contour_trans_rot': '轮廓随机化+平移+旋转',
        'contour_trans_rot_noise': '轮廓随机化+平移+旋转+噪声',
        'param_settings': '🔧 参数设置',
        'translation_range': '平移范围(像素):',
        'rotation_range': '旋转范围(度):',
        'noise_std': '噪声标准差:',
        'execution_control': '执行控制',
        'select_all': '全选',
        'clear_all': '清除',
        'start_processing': '🚀 开始处理',
        'run_log': '📝 运行日志',
        'welcome_msg': '🎯 椎间盘图像扰动系统已就绪！\n💡 提示：选择文件和扰动类型后点击开始处理'
    },
    'en': {
        'file_selection': '📁 File Selection',
        'process_mode': 'Process Mode:',
        'batch_mode': '📊 Batch Process',
        'single_mode': '🔍 Single Case',
        'input_path': 'Input Path:',
        'mask_path': 'Mask Path:',
        'output_path': 'Output Path:',
        'select': 'Select',
        'perturbation_types': '🔧 Perturbation Types',
        'original': 'Original',
        'dilation': 'Dilation',
        'erosion': 'Erosion',
        'contour_random': 'Contour Randomization',
        'translation': 'Translation',
        'rotation': 'Rotation',
        'gaussian_noise': 'Gaussian Noise',
        'translation_rotation': 'Translation+Rotation',
        'dilation_trans_rot': 'Dilation+Translation+Rotation',
        'erosion_trans_rot': 'Erosion+Translation+Rotation',
        'contour_trans_rot': 'Contour Random+Translation+Rotation',
        'contour_trans_rot_noise': 'Contour Random+Translation+Rotation+Noise',
        'param_settings': '🔧 Parameter Settings',
        'translation_range': 'Translation Range (pixels):',
        'rotation_range': 'Rotation Range (degrees):',
        'noise_std': 'Noise Std Dev:',
        'execution_control': 'Execution Control',
        'select_all': 'Select All',
        'clear_all': 'Clear All',
        'start_processing': '🚀 Start Processing',
        'run_log': '📝 Run Log',
        'welcome_msg': '🎯 IVD Image Perturbation System Ready!\n💡 Tip: Select files and perturbation types then click start'
    }
}

class PerturbationWorker(QThread):
    """后台处理线程"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, image_path, mask_path, output_path, perturbations, params):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.output_path = output_path
        self.perturbations = perturbations
        self.params = params
        
    def run(self):
        try:
            self.process_files()
        except Exception as e:
            self.error.emit(str(e))
            
    def process_files(self):
        if os.path.isfile(self.image_path) and os.path.isfile(self.mask_path):
            self.process_single_file(self.image_path, self.mask_path, self.output_path)
        elif os.path.isdir(self.image_path) and os.path.isdir(self.mask_path):
            self.process_directory(self.image_path, self.mask_path, self.output_path)
        else:
            self.error.emit("输入路径必须都是文件或都是文件夹")
            
    def process_directory(self, img_dir, mask_dir, out_dir):

        img_files = {}
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith(('.dcm', '.nii', '.nii.gz')):
                    rel_path = os.path.relpath(root, img_dir)
                    base_name = Path(file).stem
                    if base_name.endswith('.nii'):
                        base_name = Path(base_name).stem
                    key = os.path.join(rel_path, base_name).replace('\\', '/')
                    img_files[key] = (os.path.join(root, file), rel_path)
        
        mask_files = {}
        for root, dirs, files in os.walk(mask_dir):
            for file in files:
                if file.lower().endswith(('.dcm', '.nii', '.nii.gz')):
                    rel_path = os.path.relpath(root, mask_dir)
                    base_name = Path(file).stem
                    if base_name.endswith('.nii'):
                        base_name = Path(base_name).stem
                    key = os.path.join(rel_path, base_name).replace('\\', '/')
                    mask_files[key] = (os.path.join(root, file), rel_path)
        
        matched_keys = sorted(set(img_files.keys()) & set(mask_files.keys()))
        total_files = len(matched_keys)
        
        if total_files == 0:
            self.error.emit("没有找到匹配的图像和掩膜文件对")
            return
        
        self.status.emit(f"找到 {total_files} 对匹配的文件")
        
        for i, key in enumerate(matched_keys):
            img_path, img_rel = img_files[key]
            mask_path, mask_rel = mask_files[key]
            
            self.progress.emit(int((i / total_files) * 100))
            self.status.emit(f"处理文件 {i+1}/{total_files}: {os.path.basename(img_path)}")
            
            out_subdir = os.path.join(out_dir, img_rel)
            os.makedirs(out_subdir, exist_ok=True)
            
            self.process_single_file(img_path, mask_path, out_subdir)
            
    def process_single_file(self, img_path, mask_path, out_dir):

        image = self.read_medical_image(img_path)
        mask = self.read_medical_image(mask_path)
        
        if image is None or mask is None:
            self.error.emit(f"无法读取文件: {img_path} 或 {mask_path}")
            return
            
        base_name = Path(img_path).stem
        if base_name.endswith('.nii'):
            base_name = base_name[:-4]
            
        for perturb_name in self.perturbations:
            self.status.emit(f"应用扰动: {perturb_name}")
            
            if perturb_name == "原始":
                perturbed_img, perturbed_mask = image.copy(), mask.copy()
            elif perturb_name == "膨胀":
                perturbed_img, perturbed_mask = self.apply_dilation(image, mask)
            elif perturb_name == "腐蚀":
                perturbed_img, perturbed_mask = self.apply_erosion(image, mask)
            elif perturb_name == "轮廓随机化":
                perturbed_img, perturbed_mask = self.apply_contour_randomization(image, mask)
            elif perturb_name == "平移":
                perturbed_img, perturbed_mask = self.apply_translation(image, mask)
            elif perturb_name == "旋转":
                perturbed_img, perturbed_mask = self.apply_rotation(image, mask)
            elif perturb_name == "高斯噪声":
                perturbed_img, perturbed_mask = self.apply_gaussian_noise(image, mask)
            elif perturb_name == "平移+旋转":
                perturbed_img, perturbed_mask = self.apply_translation_rotation(image, mask)
            elif perturb_name == "膨胀+平移+旋转":
                img_temp, mask_temp = self.apply_dilation(image, mask)
                perturbed_img, perturbed_mask = self.apply_translation_rotation(img_temp, mask_temp)
            elif perturb_name == "腐蚀+平移+旋转":
                img_temp, mask_temp = self.apply_erosion(image, mask)
                perturbed_img, perturbed_mask = self.apply_translation_rotation(img_temp, mask_temp)
            elif perturb_name == "轮廓随机化+平移+旋转":
                img_temp, mask_temp = self.apply_contour_randomization(image, mask)
                perturbed_img, perturbed_mask = self.apply_translation_rotation(img_temp, mask_temp)
            elif perturb_name == "轮廓随机化+平移+旋转+噪声":
                img_temp, mask_temp = self.apply_contour_randomization(image, mask)
                img_temp2, mask_temp2 = self.apply_translation_rotation(img_temp, mask_temp)
                perturbed_img, perturbed_mask = self.apply_gaussian_noise(img_temp2, mask_temp2)
                
            safe_name = perturb_name.replace("+", "_")
            img_out_path = os.path.join(out_dir, f"{base_name}_{safe_name}_image.nii.gz")
            mask_out_path = os.path.join(out_dir, f"{base_name}_{safe_name}_mask.nii.gz")
            
            self.save_medical_image(perturbed_img, img_out_path)
            self.save_medical_image(perturbed_mask, mask_out_path)
            
    def read_medical_image(self, path):

        try:
            if path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(path)
                return ds.pixel_array.astype(np.float32)
            else:
                img = sitk.ReadImage(path)
                return sitk.GetArrayFromImage(img).astype(np.float32)
        except Exception as e:
            self.error.emit(f"读取文件错误 {path}: {str(e)}")
            return None
            
    def save_medical_image(self, array, path):

        img = sitk.GetImageFromArray(array)
        sitk.WriteImage(img, path)
        
    def apply_dilation(self, image, mask):

        kernel = np.ones((2, 2), np.uint8)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return image.copy(), dilated_mask.astype(np.float32)
        
    def apply_erosion(self, image, mask):

        kernel = np.ones((2, 2), np.uint8)
        eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        return image.copy(), eroded_mask.astype(np.float32)
        
    def apply_contour_randomization(self, image, mask):

        kernel = np.ones((2, 2), np.uint8)
        
        if np.random.random() > 0.5:
            randomized_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        else:
            randomized_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
            
        return image.copy(), randomized_mask.astype(np.float32)
        
    def apply_translation(self, image, mask):

        tx = np.random.randint(-self.params['translation_range'], self.params['translation_range'] + 1)
        ty = np.random.randint(-self.params['translation_range'], self.params['translation_range'] + 1)
        
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        
        h, w = image.shape[:2]
        translated_img = cv2.warpAffine(image, M, (w, h))
        translated_mask = cv2.warpAffine(mask, M, (w, h))
        
        return translated_img, translated_mask
        
    def apply_rotation(self, image, mask):

        angle = np.random.randint(0, self.params['rotation_range'])
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated_img = cv2.warpAffine(image, M, (w, h))
        rotated_mask = cv2.warpAffine(mask, M, (w, h))
        
        return rotated_img, rotated_mask
        
    def apply_gaussian_noise(self, image, mask):

        noise = np.random.normal(0, self.params['noise_std'], image.shape)
        noisy_image = image + noise
        
        noisy_image = np.clip(noisy_image, 0, 255)
        
        return noisy_image, mask.copy()
    
    def apply_translation_rotation(self, image, mask):

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        angle = np.random.uniform(0, self.params['rotation_range'])
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        tx = np.random.randint(-self.params['translation_range'], self.params['translation_range'] + 1)
        ty = np.random.randint(-self.params['translation_range'], self.params['translation_range'] + 1)
        
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty
        
        transformed_img = cv2.warpAffine(image, M_rot, (w, h))
        transformed_mask = cv2.warpAffine(mask, M_rot, (w, h))
        
        return transformed_img, transformed_mask

class PerturbationGUI:
    def __init__(self, parent):
        self.parent = parent
        self.current_lang = tk.StringVar(value="cn")
        self.image_path = None
        self.mask_path = None
        self.output_path = None
        self.input_type = tk.StringVar(value="batch")
        self.widgets = {}
        self.create_checkbox_icon()
        self.setup_gui()

    def get_text(self, key):

        lang = self.current_lang.get()
        return PERTURB_TEXT_DICT[lang].get(key, key)

    def update_language(self):

        if 'file_frame' in self.widgets:
            self.widgets['file_frame'].config(text=self.get_text('file_selection'))
        if 'perturb_frame' in self.widgets:
            self.widgets['perturb_frame'].config(text=self.get_text('perturbation_types'))
        if 'param_frame' in self.widgets:
            self.widgets['param_frame'].config(text=self.get_text('param_settings'))
        if 'control_frame' in self.widgets:
            self.widgets['control_frame'].config(text=self.get_text('execution_control'))
        if 'log_frame' in self.widgets:
            self.widgets['log_frame'].config(text=self.get_text('run_log'))
        
        if 'mode_label' in self.widgets:
            self.widgets['mode_label'].config(text=self.get_text('process_mode'))
        if 'input_label' in self.widgets:
            self.widgets['input_label'].config(text=self.get_text('input_path'))
        if 'mask_label' in self.widgets:
            self.widgets['mask_label'].config(text=self.get_text('mask_path'))
        if 'output_label' in self.widgets:
            self.widgets['output_label'].config(text=self.get_text('output_path'))
        if 'trans_label' in self.widgets:
            self.widgets['trans_label'].config(text=self.get_text('translation_range'))
        if 'rot_label' in self.widgets:
            self.widgets['rot_label'].config(text=self.get_text('rotation_range'))
        if 'noise_label' in self.widgets:
            self.widgets['noise_label'].config(text=self.get_text('noise_std'))
        
        if 'input_btn' in self.widgets:
            self.widgets['input_btn'].config(text="📂 " + self.get_text('select'))
        if 'mask_btn' in self.widgets:
            self.widgets['mask_btn'].config(text="🎯 " + self.get_text('select'))
        if 'output_btn' in self.widgets:
            self.widgets['output_btn'].config(text="💾 " + self.get_text('select'))
        if 'select_all_btn' in self.widgets:
            self.widgets['select_all_btn'].config(text=self.get_text('select_all'))
        if 'clear_all_btn' in self.widgets:
            self.widgets['clear_all_btn'].config(text=self.get_text('clear_all'))
        if 'start_btn' in self.widgets:
            self.widgets['start_btn'].config(text=self.get_text('start_processing'))
        
        if 'batch_radio' in self.widgets:
            self.widgets['batch_radio'].config(text=self.get_text('batch_mode'))
        if 'single_radio' in self.widgets:
            self.widgets['single_radio'].config(text=self.get_text('single_mode'))
        
        if hasattr(self, 'perturb_checks'):
            perturbation_map = {
                "原始": 'original',
                "膨胀": 'dilation',
                "腐蚀": 'erosion',
                "轮廓随机化": 'contour_random',
                "平移": 'translation',
                "旋转": 'rotation',
                "高斯噪声": 'gaussian_noise',
                "平移+旋转": 'translation_rotation',
                "膨胀+平移+旋转": 'dilation_trans_rot',
                "腐蚀+平移+旋转": 'erosion_trans_rot',
                "轮廓随机化+平移+旋转": 'contour_trans_rot',
                "轮廓随机化+平移+旋转+噪声": 'contour_trans_rot_noise'
            }
            
            for name, check in self.perturb_checks.items():
                if name in perturbation_map:
                    check.config(text=self.get_text(perturbation_map[name]))
        
        if hasattr(self, 'log_text'):
            self.log_text.delete(1.0, tk.END)
            self.log_message(self.get_text('welcome_msg'))
    
    def create_checkbox_icon(self):

        import tkinter as tk
        from PIL import Image, ImageDraw
        
        img = Image.new('RGBA', (13, 13), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        draw.line([(3, 7), (6, 10)], fill='white', width=2)
        draw.line([(6, 10), (10, 3)], fill='white', width=2)
        
        self.checkbox_icon_path = os.path.join(os.path.dirname(__file__), "checkbox_check.png")
        img.save(self.checkbox_icon_path)
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.parent, padding="15")
        main_frame.pack(fill="both", expand=True)
        
        file_frame = ttk.LabelFrame(main_frame, text=self.get_text('file_selection'), padding="10")
        file_frame.pack(fill="x", pady=5)
        self.widgets['file_frame'] = file_frame
        
        mode_frame = ttk.Frame(file_frame)
        mode_frame.pack(fill="x", pady=(0, 10))

        mode_label = ttk.Label(mode_frame, text=self.get_text('process_mode'))
        mode_label.pack(side="left", padx=(0, 10))
        self.widgets['mode_label'] = mode_label

        self.batch_radio = ttk.Radiobutton(mode_frame, text=self.get_text('batch_mode'), 
                                        value="batch", variable=self.input_type)
        self.batch_radio.pack(side="left", padx=(0, 20))
        self.widgets['batch_radio'] = self.batch_radio

        self.single_radio = ttk.Radiobutton(mode_frame, text=self.get_text('single_mode'),
                                        value="single", variable=self.input_type)
        self.single_radio.pack(side="left")
        self.widgets['single_radio'] = self.single_radio

        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill="x", pady=2)
        input_label = ttk.Label(input_frame, text=self.get_text('input_path'), width=10)
        input_label.pack(side="left")
        self.widgets['input_label'] = input_label
        
        self.input_entry = ttk.Entry(input_frame)
        self.input_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        input_btn = ttk.Button(input_frame, text="📂 " + self.get_text('select'), 
                            command=self.select_input)
        input_btn.pack(side="left")
        self.widgets['input_btn'] = input_btn

        mask_frame = ttk.Frame(file_frame)
        mask_frame.pack(fill="x", pady=2)
        mask_label = ttk.Label(mask_frame, text=self.get_text('mask_path'), width=10)
        mask_label.pack(side="left")
        self.widgets['mask_label'] = mask_label
        
        self.mask_entry = ttk.Entry(mask_frame)
        self.mask_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        mask_btn = ttk.Button(mask_frame, text="🎯 " + self.get_text('select'),
                            command=self.select_mask)
        mask_btn.pack(side="left")
        self.widgets['mask_btn'] = mask_btn

        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill="x", pady=2)
        output_label = ttk.Label(output_frame, text=self.get_text('output_path'), width=10)
        output_label.pack(side="left")
        self.widgets['output_label'] = output_label
        
        self.output_entry = ttk.Entry(output_frame)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        output_btn = ttk.Button(output_frame, text="💾 " + self.get_text('select'),
                            command=self.select_output)
        output_btn.pack(side="left")
        self.widgets['output_btn'] = output_btn

        perturb_frame = ttk.LabelFrame(main_frame, text=self.get_text('perturbation_types'), padding="10")
        perturb_frame.pack(fill="x", pady=5)
        self.widgets['perturb_frame'] = perturb_frame
        
        self.perturb_checks = {}
        perturbations = [
            "原始", "膨胀", "腐蚀", "轮廓随机化", 
            "平移", "旋转", "高斯噪声", "平移+旋转",
            "膨胀+平移+旋转", "腐蚀+平移+旋转", 
            "轮廓随机化+平移+旋转", "轮廓随机化+平移+旋转+噪声"
        ]
        
        perturbation_keys = {
            "原始": 'original',
            "膨胀": 'dilation',
            "腐蚀": 'erosion',
            "轮廓随机化": 'contour_random',
            "平移": 'translation',
            "旋转": 'rotation',
            "高斯噪声": 'gaussian_noise',
            "平移+旋转": 'translation_rotation',
            "膨胀+平移+旋转": 'dilation_trans_rot',
            "腐蚀+平移+旋转": 'erosion_trans_rot',
            "轮廓随机化+平移+旋转": 'contour_trans_rot',
            "轮廓随机化+平移+旋转+噪声": 'contour_trans_rot_noise'
        }
        
        for i, name in enumerate(perturbations):
            key = perturbation_keys.get(name, name)
            check = ttk.Checkbutton(perturb_frame, text=self.get_text(key))
            self.perturb_checks[name] = check
            check.grid(row=i // 3, column=i % 3, sticky="w", padx=5, pady=2)
        
        param_frame = ttk.LabelFrame(main_frame, text=self.get_text('param_settings'), padding="10")
        param_frame.pack(fill="x", pady=5)
        self.widgets['param_frame'] = param_frame
        
        param_grid = ttk.Frame(param_frame)
        param_grid.pack(fill="x")
        
        trans_label = ttk.Label(param_grid, text=self.get_text('translation_range'))
        trans_label.grid(row=0, column=0, sticky="w", pady=2)
        self.widgets['trans_label'] = trans_label
        
        self.translation_var = tk.IntVar(value=10)
        self.translation_spin = ttk.Spinbox(param_grid, from_=1, to=20, 
                                        textvariable=self.translation_var, width=10)
        self.translation_spin.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        rot_label = ttk.Label(param_grid, text=self.get_text('rotation_range'))
        rot_label.grid(row=0, column=2, sticky="w", padx=(20,0), pady=2)
        self.widgets['rot_label'] = rot_label
        
        self.rotation_var = tk.IntVar(value=360)
        self.rotation_spin = ttk.Spinbox(param_grid, from_=1, to=360,
                                    textvariable=self.rotation_var, width=10)
        self.rotation_spin.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        noise_label = ttk.Label(param_grid, text=self.get_text('noise_std'))
        noise_label.grid(row=1, column=0, sticky="w", pady=2)
        self.widgets['noise_label'] = noise_label
        
        self.noise_var = tk.DoubleVar(value=8.0)
        self.noise_spin = ttk.Spinbox(param_grid, from_=0.1, to=20.0, increment=0.5,
                                    textvariable=self.noise_var, width=10)
        self.noise_spin.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        control_frame = ttk.LabelFrame(main_frame, text=self.get_text('execution_control'), padding="10")
        control_frame.pack(fill="x", pady=5)
        self.widgets['control_frame'] = control_frame
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x")
        
        self.select_all_btn = ttk.Button(button_frame, text=self.get_text('select_all'),
                                    command=self.select_all_perturbations)
        self.select_all_btn.pack(side="left", padx=2)
        self.widgets['select_all_btn'] = self.select_all_btn
        
        self.clear_all_btn = ttk.Button(button_frame, text=self.get_text('clear_all'),
                                    command=self.clear_all_perturbations)
        self.clear_all_btn.pack(side="left", padx=2)
        self.widgets['clear_all_btn'] = self.clear_all_btn
        
        self.start_btn = ttk.Button(button_frame, text=self.get_text('start_processing'),
                                command=self.start_processing)
        self.start_btn.pack(side="right", padx=2)
        self.widgets['start_btn'] = self.start_btn
        
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var)
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.pack_forget()
        
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.pack()
        self.status_label.pack_forget()
        
        log_frame = ttk.LabelFrame(main_frame, text=self.get_text('run_log'), padding="10")
        log_frame.pack(fill="both", expand=True, pady=5)
        self.widgets['log_frame'] = log_frame
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD,
                                                font=('Consolas', 9))
        self.log_text.pack(fill="both", expand=True)
        
        self.log_message(self.get_text('welcome_msg'))
    
    def log_message(self, message):

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.parent.update()
    
    def select_input(self):

        if self.input_type.get() == "single":
            path = filedialog.askopenfilename(
                title="选择图像文件",
                filetypes=[("医学图像文件", "*.dcm *.nii *.nii.gz"), ("所有文件", "*.*")]
            )
        else:
            path = filedialog.askdirectory(title="选择图像文件夹")
        
        if path:
            self.image_path = path
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, path)
    
    def select_mask(self):

        if self.input_type.get() == "single":
            path = filedialog.askopenfilename(
                title="选择掩膜文件",
                filetypes=[("医学图像文件", "*.dcm *.nii *.nii.gz"), ("所有文件", "*.*")]
            )
        else:
            path = filedialog.askdirectory(title="选择掩膜文件夹")
        
        if path:
            self.mask_path = path
            self.mask_entry.delete(0, tk.END)
            self.mask_entry.insert(0, path)
    
    def select_output(self):

        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_path = path
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)
    
    def select_all_perturbations(self):

        for check in self.perturb_checks.values():
            check.state(['selected'])
    
    def clear_all_perturbations(self):

        for check in self.perturb_checks.values():
            check.state(['!selected'])
    
    def start_processing(self):

        if not self.image_path or not self.mask_path or not self.output_path:
            messagebox.showwarning("警告", "请选择所有必需的文件路径")
            return
        
        selected_perturbations = [name for name, check in self.perturb_checks.items() 
                                if check.instate(['selected'])]
        if not selected_perturbations:
            messagebox.showwarning("警告", "请至少选择一种扰动类型")
            return
        
        params = {
            'translation_range': self.translation_var.get(),
            'rotation_range': self.rotation_var.get(),
            'noise_std': self.noise_var.get()
        }
        
        self.start_btn.config(state='disabled')
        self.progress_bar.pack(fill="x", pady=5)
        self.status_label.pack()
        
        self.log_text.delete(1.0, tk.END)
        self.log_message("🚀 开始处理...")
        
        self.worker = PerturbationWorker(
            self.image_path, self.mask_path, self.output_path,
            selected_perturbations, params
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.processing_finished)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def update_progress(self, value):

        self.progress_var.set(value)
    
    def update_status(self, text):

        self.status_label.config(text=text)
        self.log_message(text)
    
    def processing_finished(self):

        self.start_btn.config(state='normal')
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.log_message("✅ 处理完成！")
        messagebox.showinfo("完成", "图像扰动处理完成！")
    
    def show_error(self, error_msg):

        self.start_btn.config(state='normal')
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.log_message(f"❌ 错误: {error_msg}")
        messagebox.showerror("错误", f"处理过程中出现错误：\n{error_msg}")

    def __del__(self):

        try:
            if hasattr(self, 'checkbox_icon_path') and os.path.exists(self.checkbox_icon_path):
                os.remove(self.checkbox_icon_path)
        except:
            pass

__all__ = ['PerturbationGUI', 'PerturbationWorker']