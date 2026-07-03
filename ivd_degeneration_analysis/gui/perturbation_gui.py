import sys
import os
import numpy as np
import cv2
from tkinter import ttk, filedialog, messagebox, scrolledtext
import tkinter as tk
import SimpleITK as sitk
import pydicom
from pathlib import Path
import json
from datetime import datetime
import threading
from queue import Queue

SUPPORTED_MEDICAL_EXTENSIONS = ('.dcm', '.nii', '.nii.gz', '.nrrd', '.mha', '.mhd')
MEDICAL_FILETYPES = [("医学图像文件", "*.dcm *.nii *.nii.gz *.nrrd *.mha *.mhd"), ("所有文件", "*.*")]

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
        'translation': '平移',
        'rotation': '旋转',
        'gaussian_noise': '高斯噪声',
        'translation_rotation': '平移+旋转',
        'dilation_trans_rot': '膨胀+平移+旋转',
        'erosion_trans_rot': '腐蚀+平移+旋转',
        'param_settings': '🔧 参数设置',
        'translation_range': '平移范围(像素):',
        'rotation_range': '旋转范围(度):',
        'noise_std': '噪声标准差:',
        'morph_kernel_size': '形态学核大小:',
        'morph_iterations': '迭代次数:',
        'execution_control': '执行控制',
        'select_all': '全选',
        'clear_all': '清除',
        'start_processing': '🚀 开始处理',
        'run_log': '📝 运行日志',
        'welcome_msg': '🎯 椎间盘图像扰动系统已就绪！'
    }
}

class PerturbationWorker(threading.Thread):
    
    def __init__(self, image_path, mask_path, output_path, perturbations, params):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.output_path = output_path
        self.perturbations = perturbations
        self.params = params

    MIN_PIXEL_THRESHOLD = 20

    PERTURBATION_MAPPING = {
        "膨胀": "dilation",
        "腐蚀": "erosion",
        "平移": "translation",
        "旋转": "rotation",
        "高斯噪声": "gaussian_noise",
        "平移+旋转": "translation_rotation",
        "膨胀+平移+旋转": "dilation_trans_rot",
        "腐蚀+平移+旋转": "erosion_trans_rot"
    }
        
    def __init__(self, image_path, mask_path, output_path, perturbations, params, callback_queue):
        super().__init__(daemon=True)
        self.image_path = image_path
        self.mask_path = mask_path
        self.output_path = output_path
        self.perturbations = perturbations
        self.params = params
        self.callback_queue = callback_queue
        self._stop_event = threading.Event()
        
    def stop(self):
        self._stop_event.set()
        
    def emit_progress(self, value):
        self.callback_queue.put(('progress', value))
        
    def emit_status(self, text):
        self.callback_queue.put(('status', text))
        
    def emit_error(self, text):
        self.callback_queue.put(('error', text))
        
    def emit_finished(self):
        self.callback_queue.put(('finished', None))
    
    def run(self):
        try:
            self.process_files()
            self.emit_finished()
        except Exception as e:
            self.emit_error(str(e))
            
    def process_files(self):
        self.image_output_dir = os.path.join(self.output_path, 'image')
        self.mask_output_dir = os.path.join(self.output_path, 'mask')
        os.makedirs(self.image_output_dir, exist_ok=True)
        os.makedirs(self.mask_output_dir, exist_ok=True)
        
        if os.path.isfile(self.image_path) and os.path.isfile(self.mask_path):
            self.process_single_file(self.image_path, self.mask_path)
        elif os.path.isdir(self.image_path) and os.path.isdir(self.mask_path):
            self.process_directory(self.image_path, self.mask_path)
        else:
            self.emit_error("输入路径必须都是文件或都是文件夹")
            
    def process_directory(self, img_dir, mask_dir):
        img_files = {}
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if self._is_supported_medical_file(file):
                    rel_path = os.path.relpath(root, img_dir)
                    base_name = self._strip_all_suffixes(file)
                    
                    key = os.path.join(rel_path, base_name).replace('\\', '/')
                    img_files[key] = (os.path.join(root, file), rel_path)
        
        mask_dict = {}
        for root, dirs, files in os.walk(mask_dir):
            for file in files:
                if self._is_supported_medical_file(file):
                    rel_path = os.path.relpath(root, mask_dir)
                    base_name = self._strip_all_suffixes(file)
                    
                    clean_base_name = base_name.replace('_mask', '').replace('_seg', '').replace('-mask', '').replace('-seg', '')
                    
                    key = os.path.join(rel_path, clean_base_name).replace('\\', '/')
                    mask_dict[key] = os.path.join(root, file)

        matched_pairs = []
        for key, (img_path, img_rel_path) in img_files.items():
            if key in mask_dict:
                mask_path = mask_dict[key]
                case_id = self._strip_all_suffixes(Path(img_path).name)
                matched_pairs.append((case_id, img_path, mask_path, img_rel_path))
        
        matched_pairs = sorted(matched_pairs)
        total_files = len(matched_pairs)
        
        if total_files == 0:
            self.emit_error("没有找到匹配的图像和掩膜文件对")
            return
        
        self.emit_status(f"找到 {total_files} 对匹配的文件")
        
        for i, (case_id, img_path, mask_path, rel_path) in enumerate(matched_pairs):
            if self._stop_event.is_set():
                self.emit_status("处理被用户中止。")
                break

            self.emit_progress(int((i / total_files) * 100))
            self.emit_status(f"处理文件 {i+1}/{total_files}: {case_id}")
            
            self.process_single_file(img_path, mask_path)
            
    def process_single_file(self, img_path, mask_path):
        image = self.read_medical_image(img_path)
        mask = self.read_medical_image(mask_path)
        
        if image is None or mask is None:
            self.emit_error(f"无法读取文件: {img_path} 或 {mask_path}")
            return
        
        base_name = Path(img_path).stem
        if base_name.endswith('.nii'):
            base_name = Path(base_name).stem
        
        total_perturbations = len(self.perturbations)
        
        for p_idx, perturb_name in enumerate(self.perturbations):
            if self._stop_event.is_set():
                break
                
            self.emit_status(f"应用扰动 {p_idx+1}/{total_perturbations}: {perturb_name}")
            
            safe_name = self.PERTURBATION_MAPPING.get(perturb_name, perturb_name)
            safe_name = safe_name.replace("+", "_").replace(" ", "_")
            
            random_params = self._generate_random_params(perturb_name)
            
            perturbed_img, perturbed_mask = self._apply_perturbation_with_params(
                image.astype(np.float32), mask.astype(np.float32), perturb_name, random_params, img_path
            )

            progress = int(((p_idx + 1) / total_perturbations) * 100)
            self.emit_progress(progress)
            
            img_out_path = os.path.join(self.image_output_dir, f"{base_name}_{safe_name}.nii.gz")
            mask_out_path = os.path.join(self.mask_output_dir, f"{base_name}_{safe_name}_mask.nii.gz")
            
            self.save_medical_image(perturbed_img, img_out_path, img_path)
            self.save_medical_image(perturbed_mask, mask_out_path, img_path)
            
            self.emit_status(f"已保存: {base_name}_{safe_name}")

    @staticmethod
    def _is_supported_medical_file(filename):
        return filename.lower().endswith(SUPPORTED_MEDICAL_EXTENSIONS)

    @staticmethod
    def _strip_all_suffixes(filename):
        base_name = Path(filename).name
        while Path(base_name).suffix:
            base_name = Path(base_name).stem
        return base_name

    def _apply_perturbation(self, img, mask, perturb_name):
        img = np.squeeze(img) if img.ndim > 2 else img
        mask = np.squeeze(mask) if mask.ndim > 2 else mask
        
        if perturb_name == "原始":
            return img.copy(), mask.copy()
        elif perturb_name == "膨胀":
            return self.apply_dilation(img, mask)
        elif perturb_name == "腐蚀":
            return self.apply_erosion(img, mask)
        elif perturb_name == "平移":
            return self.apply_translation(img, mask)
        elif perturb_name == "旋转":
            return self.apply_rotation(img, mask)
        elif perturb_name == "高斯噪声":
            return self.apply_gaussian_noise(img, mask)
        elif perturb_name == "平移+旋转":
            return self.apply_translation_rotation(img, mask)
        elif perturb_name == "膨胀+平移+旋转":
            img_temp, mask_temp = self.apply_translation_rotation(img, mask)
            return self.apply_dilation(img_temp, mask_temp)
        elif perturb_name == "腐蚀+平移+旋转":
            img_temp, mask_temp = self.apply_translation_rotation(img, mask)
            return self.apply_erosion(img_temp, mask_temp)
        else:
            return img.copy(), mask.copy()
            
    def _apply_perturbation_to_slice(self, img_slice, mask_slice, perturb_name):
        if len(img_slice.shape) > 2:
            img_slice = img_slice.squeeze()
        if len(mask_slice.shape) > 2:
            mask_slice = mask_slice.squeeze()
        
        if perturb_name == "原始":
            return img_slice.copy(), mask_slice.copy()
        elif perturb_name == "膨胀":
            return self.apply_dilation(img_slice, mask_slice)
        elif perturb_name == "腐蚀":
            return self.apply_erosion(img_slice, mask_slice)
        elif perturb_name == "平移":
            return self.apply_translation(img_slice, mask_slice)
        elif perturb_name == "旋转":
            return self.apply_rotation(img_slice, mask_slice)
        elif perturb_name == "高斯噪声":
            return self.apply_gaussian_noise(img_slice, mask_slice)
        elif perturb_name == "平移+旋转":
            return self.apply_translation_rotation(img_slice, mask_slice)
        elif perturb_name == "膨胀+平移+旋转":
            img_temp, mask_temp = self.apply_dilation(img_slice, mask_slice)
            return self.apply_translation_rotation(img_temp, mask_temp)
        elif perturb_name == "腐蚀+平移+旋转":
            img_temp, mask_temp = self.apply_erosion(img_slice, mask_slice)
            return self.apply_translation_rotation(img_temp, mask_temp)
        else:
            return img_slice.copy(), mask_slice.copy()

    def read_medical_image(self, path):
        try:
            if path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(path)
                return ds.pixel_array.astype(np.float32)
            else:
                img = sitk.ReadImage(path)
                return sitk.GetArrayFromImage(img).astype(np.float32)
        except Exception as e:
            self.emit_error(f"读取文件错误 {path}: {str(e)}")
            return None
            
    def save_medical_image(self, array, path, reference_path=None):
        img = sitk.GetImageFromArray(array)
        if reference_path and os.path.exists(reference_path):
            try:
                reference_img = sitk.ReadImage(reference_path)
                img.CopyInformation(reference_img)
            except Exception as e:
                self.emit_status(f"警告: 无法从 {reference_path} 复制元数据: {e}")
        sitk.WriteImage(img, path)
        
    def apply_dilation(self, image, mask):
        kernel_size = self.params.get('morph_kernel_size', 1)
        iterations = self.params.get('morph_iterations', 1)
        disc_labels = [3, 5, 7, 9, 11]
        
        if image.ndim == 3:
            final_mask = mask.copy()
            final_mask[np.isin(mask, disc_labels)] = 0

            kernel_2d_size = max(1, int(kernel_size))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_2d_size, kernel_2d_size))

            for label in disc_labels:
                binary_mask = (mask == label).astype(np.uint8)
                if np.sum(binary_mask) == 0:
                    continue

                dilated = np.zeros_like(binary_mask, dtype=np.uint8)
                for z in range(binary_mask.shape[0]):
                    if np.sum(binary_mask[z]) == 0:
                        continue
                    dilated[z] = cv2.dilate(binary_mask[z], kernel, iterations=int(iterations))

                final_mask[dilated > 0] = label

            return image.copy(), final_mask

        else:
            result_mask = mask.copy()
            kernel_2d_size = max(1, int(kernel_size))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_2d_size, kernel_2d_size))
            for label in disc_labels:
                binary_mask = (mask == label).astype(np.uint8)
                if np.sum(binary_mask) == 0:
                    continue
                dilated = cv2.dilate(binary_mask, kernel, iterations=int(iterations))
                result_mask[dilated > 0] = label
            return image.copy(), result_mask

    def apply_erosion(self, image, mask):
        kernel_size = self.params.get('morph_kernel_size', 1)
        iterations = self.params.get('morph_iterations', 1)
        disc_labels = [3, 5, 7, 9, 11]

        if image.ndim == 3:
            final_mask = mask.copy()
            final_mask[np.isin(mask, disc_labels)] = 0

            kernel_2d_size = max(1, int(kernel_size))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_2d_size, kernel_2d_size))

            for label in disc_labels:
                binary_mask = (mask == label).astype(np.uint8)
                if np.sum(binary_mask) == 0:
                    continue

                eroded = np.zeros_like(binary_mask, dtype=np.uint8)
                for z in range(binary_mask.shape[0]):
                    if np.sum(binary_mask[z]) == 0:
                        continue
                    eroded[z] = cv2.erode(binary_mask[z], kernel, iterations=int(iterations))

                if np.sum(eroded) > self.MIN_PIXEL_THRESHOLD:
                    final_mask[eroded > 0] = label
                else:
                    final_mask[binary_mask > 0] = label

            return image.copy(), final_mask

        else:
            result_mask = mask.copy()
            kernel_2d_size = max(1, int(kernel_size))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_2d_size, kernel_2d_size))
            for label in disc_labels:
                binary_mask = (mask == label).astype(np.uint8)
                if np.sum(binary_mask) == 0:
                    continue
                eroded = cv2.erode(binary_mask, kernel, iterations=int(iterations))
                if np.sum(eroded) > self.MIN_PIXEL_THRESHOLD:
                    result_mask[mask == label] = 0
                    result_mask[eroded > 0] = label
            return image.copy(), result_mask

    def apply_translation(self, image, mask):
        image = np.squeeze(image) if image.ndim > 2 else image
        mask = np.squeeze(mask) if mask.ndim > 2 else mask
        
        range_val = self.params['translation_range']
        tx = np.random.randint(-range_val, range_val + 1)
        ty = np.random.randint(-range_val, range_val + 1)
        
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        h, w = image.shape[:2]
        
        translated_img = cv2.warpAffine(
            image.astype(np.float32),
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        translated_mask = cv2.warpAffine(
            mask.astype(np.float32),
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        
        return translated_img, translated_mask

    def apply_rotation(self, image, mask):
        image = np.squeeze(image) if image.ndim > 2 else image
        mask = np.squeeze(mask) if mask.ndim > 2 else mask
        
        max_angle = self.params['rotation_range']
        angle = np.random.uniform(-max_angle, max_angle)
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated_img = cv2.warpAffine(
            image.astype(np.float32),
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rotated_mask = cv2.warpAffine(
            mask.astype(np.float32),
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        
        return rotated_img, rotated_mask

    def apply_gaussian_noise(self, image, mask):
        image = np.squeeze(image) if image.ndim > 2 else image
        mask = np.squeeze(mask) if mask.ndim > 2 else mask
        
        noise_std = self.params['noise_std']
        noise = np.random.normal(0, noise_std, image.shape)
        noisy_image = image + noise
        
        return noisy_image.astype(np.float32), mask.copy()

    def apply_translation_rotation(self, image, mask):
        image = np.squeeze(image) if image.ndim > 2 else image
        mask = np.squeeze(mask) if mask.ndim > 2 else mask
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        range_val = self.params['translation_range']
        max_angle = self.params['rotation_range']
        
        angle = np.random.uniform(-max_angle, max_angle)
        tx = np.random.randint(-range_val, range_val + 1)
        ty = np.random.randint(-range_val, range_val + 1)
        
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty
        
        transformed_img = cv2.warpAffine(
            image.astype(np.float32),
            M_rot,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        transformed_mask = cv2.warpAffine(
            mask.astype(np.float32),
            M_rot,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        
        return transformed_img, transformed_mask
    
    def _get_mask_center_of_mass(self, sitk_mask):
        binary_mask = sitk_mask > 0
        
        label_stats_filter = sitk.LabelShapeStatisticsImageFilter()
        label_stats_filter.Execute(binary_mask)
        
        center_of_mass_phys = label_stats_filter.GetCentroid(1)
        
        return center_of_mass_phys
    
    def _generate_random_params(self, perturb_name):
        params = {}
        
        if "平移" in perturb_name:
            range_val = self.params['translation_range']
            params['tx'] = np.random.randint(-range_val, range_val + 1)
            params['ty'] = np.random.randint(-range_val, range_val + 1)
        
        if "旋转" in perturb_name:
            max_angle = self.params['rotation_range']
            params['angle'] = np.random.uniform(-max_angle, max_angle)
        
        return params

    def _apply_perturbation_with_params(self, img, mask, perturb_name, random_params, img_path=None):
        img = np.squeeze(img) if img.ndim > 2 else img
        mask = np.squeeze(mask) if mask.ndim > 2 else mask
        
        if perturb_name == "原始":
            return img.copy(), mask.copy()
        elif perturb_name == "膨胀":
            return self.apply_dilation(img, mask)
        elif perturb_name == "腐蚀":
            return self.apply_erosion(img, mask)
        elif perturb_name == "平移":
            return self.apply_translation_with_params(img, mask, random_params, img_path)
        elif perturb_name == "旋转":
            return self.apply_rotation_with_params(img, mask, random_params, img_path)
        elif perturb_name == "高斯噪声":
            return self.apply_gaussian_noise(img, mask)
        elif perturb_name == "平移+旋转":
            return self.apply_translation_rotation_with_params(img, mask, random_params, img_path)
        elif perturb_name == "膨胀+平移+旋转":
            img_temp, mask_temp = self.apply_translation_rotation_with_params(img, mask, random_params, img_path)
            return self.apply_dilation(img_temp, mask_temp)
        elif perturb_name == "腐蚀+平移+旋转":
            img_temp, mask_temp = self.apply_translation_rotation_with_params(img, mask, random_params, img_path)
            return self.apply_erosion(img_temp, mask_temp)
        else:
            return img.copy(), mask.copy()

    def apply_translation_with_params(self, image, mask, params, img_path):
        tx = params.get('tx', 0)
        ty = params.get('ty', 0)

        if image.ndim == 3:
            reference_sitk_image = sitk.ReadImage(img_path)

            sitk_image = sitk.GetImageFromArray(image)
            sitk_image.CopyInformation(reference_sitk_image)
            
            sitk_mask = sitk.GetImageFromArray(mask)
            sitk_mask.CopyInformation(reference_sitk_image)

            translation_vector = (0.0, float(ty), float(tx))
            transform = sitk.TranslationTransform(3, translation_vector)
            
            resampled_sitk_image = sitk.Resample(sitk_image, reference_sitk_image, transform, sitk.sitkLinear, 0.0, sitk_image.GetPixelID())
            resampled_sitk_mask = sitk.Resample(sitk_mask, reference_sitk_image, transform, sitk.sitkNearestNeighbor, 0.0, sitk_mask.GetPixelID())

            return sitk.GetArrayFromImage(resampled_sitk_image), sitk.GetArrayFromImage(resampled_sitk_mask)
        
        else:
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            h, w = image.shape[:2]
            translated_img = cv2.warpAffine(
                image.astype(np.float32),
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            translated_mask = cv2.warpAffine(
                mask.astype(np.float32),
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return translated_img, translated_mask

    def apply_rotation_with_params(self, image, mask, params, img_path):
        angle = params.get('angle', 0)
        
        if image.ndim == 3:
            reference_sitk_image = sitk.ReadImage(img_path)

            sitk_image = sitk.GetImageFromArray(image)
            sitk_image.CopyInformation(reference_sitk_image)
            
            sitk_mask = sitk.GetImageFromArray(mask)
            sitk_mask.CopyInformation(reference_sitk_image)
            
            angle_rad = np.deg2rad(angle)
            transform = sitk.Euler3DTransform()
            
            center_phys = self._get_mask_center_of_mass(sitk_mask)
            transform.SetCenter(center_phys)

            transform.SetRotation(angle_rad, 0, 0)

            resampled_sitk_image = sitk.Resample(sitk_image, reference_sitk_image, transform, sitk.sitkLinear, 0.0, sitk_image.GetPixelID())
            resampled_sitk_mask = sitk.Resample(sitk_mask, reference_sitk_image, transform, sitk.sitkNearestNeighbor, 0.0, sitk_mask.GetPixelID())

            return sitk.GetArrayFromImage(resampled_sitk_image), sitk.GetArrayFromImage(resampled_sitk_mask)

        else:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img = cv2.warpAffine(
                image.astype(np.float32),
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            rotated_mask = cv2.warpAffine(
                mask.astype(np.float32),
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return rotated_img, rotated_mask

    def apply_translation_rotation_with_params(self, image, mask, params, img_path):
        angle = params.get('angle', 0)
        tx = params.get('tx', 0)
        ty = params.get('ty', 0)
        
        if image.ndim == 3:
            reference_sitk_image = sitk.ReadImage(img_path)

            sitk_image = sitk.GetImageFromArray(image)
            sitk_image.CopyInformation(reference_sitk_image)
            
            sitk_mask = sitk.GetImageFromArray(mask)
            sitk_mask.CopyInformation(reference_sitk_image)

            angle_rad = np.deg2rad(angle)
            translation_vector = (0.0, float(ty), float(tx))

            transform = sitk.Euler3DTransform()
            center_phys = self._get_mask_center_of_mass(sitk_mask)
            transform.SetCenter(center_phys)

            transform.SetRotation(angle_rad, 0, 0)
            transform.SetTranslation(translation_vector)

            resampled_sitk_image = sitk.Resample(sitk_image, reference_sitk_image, transform, sitk.sitkLinear, 0.0, sitk_image.GetPixelID())
            resampled_sitk_mask = sitk.Resample(sitk_mask, reference_sitk_image, transform, sitk.sitkNearestNeighbor, 0.0, sitk_mask.GetPixelID())
            
            return sitk.GetArrayFromImage(resampled_sitk_image), sitk.GetArrayFromImage(resampled_sitk_mask)

        else:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            M_rot[0, 2] += tx
            M_rot[1, 2] += ty
            transformed_img = cv2.warpAffine(
                image.astype(np.float32),
                M_rot,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            transformed_mask = cv2.warpAffine(
                mask.astype(np.float32),
                M_rot,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
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
        if 'morph_label' in self.widgets:
            self.widgets['morph_label'].config(text=self.get_text('morph_kernel_size'))
        if 'morph_iter_label' in self.widgets:
            self.widgets['morph_iter_label'].config(text=self.get_text('morph_iterations'))
        
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
                "平移": 'translation',
                "旋转": 'rotation',
                "高斯噪声": 'gaussian_noise',
                "平移+旋转": 'translation_rotation',
                "膨胀+平移+旋转": 'dilation_trans_rot',
                "腐蚀+平移+旋转": 'erosion_trans_rot',
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
        self.canvas = tk.Canvas(self.parent, bg='#f0f0f0', highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollable_frame = ttk.Frame(self.canvas)
        canvas_window = self.canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def configure_scroll_region(event=None):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            canvas_width = self.canvas.winfo_width()
            if canvas_width > 0:
                self.canvas.itemconfig(canvas_window, width=canvas_width)
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(canvas_window, width=e.width))

        main_frame = ttk.Frame(scrollable_frame, padding="15")
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
        
        self.perturb_check_vars = {} 
        perturbations = [
            "膨胀", "腐蚀",
            "平移", "旋转", "高斯噪声", "平移+旋转",
            "膨胀+平移+旋转", "腐蚀+平移+旋转"
        ]
        
        perturbation_keys = {
            "膨胀": 'dilation', "腐蚀": 'erosion',
            "平移": 'translation', "旋转": 'rotation', "高斯噪声": 'gaussian_noise',
            "平移+旋转": 'translation_rotation', "膨胀+平移+旋转": 'dilation_trans_rot',
            "腐蚀+平移+旋转": 'erosion_trans_rot'
        }
        
        for i, name in enumerate(perturbations):
            key = perturbation_keys.get(name, name)
            var = tk.BooleanVar(value=True) 
            self.perturb_check_vars[name] = var
            check = ttk.Checkbutton(perturb_frame, text=self.get_text(key), variable=var)
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
        
        self.rotation_var = tk.IntVar(value=20)
        self.rotation_spin = ttk.Spinbox(param_grid, from_=1, to=30,
                                    textvariable=self.rotation_var, width=10)
        self.rotation_spin.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        noise_label = ttk.Label(param_grid, text=self.get_text('noise_std'))
        noise_label.grid(row=1, column=0, sticky="w", pady=2)
        self.widgets['noise_label'] = noise_label
        
        self.noise_var = tk.DoubleVar(value=8.0)
        self.noise_spin = ttk.Spinbox(param_grid, from_=0.1, to=20.0, increment=0.5,
                                    textvariable=self.noise_var, width=10)
        self.noise_spin.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        row = 1
        row += 1
        morph_label = ttk.Label(param_grid, text=self.get_text('morph_kernel_size'))
        morph_label.grid(row=row, column=0, sticky="w", pady=2)
        self.widgets['morph_label'] = morph_label

        self.morph_kernel_size = tk.IntVar(value=2)
        morph_spin = ttk.Spinbox(param_grid, from_=1, to=5, 
                                textvariable=self.morph_kernel_size, width=10)
        morph_spin.grid(row=row, column=1, sticky="w", padx=5, pady=2)

        morph_iter_label = ttk.Label(param_grid, text=self.get_text('morph_iterations'))
        morph_iter_label.grid(row=row, column=2, sticky="w", padx=(20,0), pady=2)
        self.widgets['morph_iter_label'] = morph_iter_label

        self.morph_iterations = tk.IntVar(value=1)
        morph_iter_spin = ttk.Spinbox(param_grid, from_=1, to=10,
                                    textvariable=self.morph_iterations, width=10)
        morph_iter_spin.grid(row=row, column=3, sticky="w", padx=5, pady=2)
        
        control_frame = ttk.LabelFrame(main_frame, text=self.get_text('execution_control'), padding="10")
        control_frame.pack(fill="x", pady=5)
        self.widgets['control_frame'] = control_frame
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x")

        left_button_frame = ttk.Frame(button_frame)
        left_button_frame.pack(side="left")

        self.select_all_btn = ttk.Button(left_button_frame, text=self.get_text('select_all'),
                                    command=self.select_all_perturbations)
        self.select_all_btn.pack(side="left", padx=2)
        self.widgets['select_all_btn'] = self.select_all_btn
        
        self.clear_all_btn = ttk.Button(left_button_frame, text=self.get_text('clear_all'),
                                    command=self.clear_all_perturbations)
        self.clear_all_btn.pack(side="left", padx=2)
        self.widgets['clear_all_btn'] = self.clear_all_btn

        center_button_frame = ttk.Frame(button_frame)
        center_button_frame.pack(fill="x", expand=True)
        
        self.start_btn = ttk.Button(center_button_frame, text=self.get_text('start_processing'),
                                command=self.start_processing)
        self.start_btn.pack(anchor="center")
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
                filetypes=MEDICAL_FILETYPES
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
                filetypes=MEDICAL_FILETYPES
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
        for var in self.perturb_check_vars.values():
            var.set(True)
    
    def clear_all_perturbations(self):
        for var in self.perturb_check_vars.values():
            var.set(False)
    
    def start_processing(self):
        if not self.image_path or not self.mask_path or not self.output_path:
            messagebox.showwarning("警告", "请选择所有必需的文件路径")
            return
        
        selected_perturbations = [name for name, var in self.perturb_check_vars.items() 
                                if var.get()]
        if not selected_perturbations:
            messagebox.showwarning("警告", "请至少选择一种扰动类型")
            return
        
        params = {
            'translation_range': self.translation_var.get(),
            'rotation_range': self.rotation_var.get(),
            'noise_std': self.noise_var.get(),
            'morph_kernel_size': self.morph_kernel_size.get(),
            'morph_iterations': self.morph_iterations.get()
        }
        
        self.start_btn.config(state='disabled')
        self.progress_bar.pack(fill="x", pady=5)
        self.status_label.pack()
        
        self.log_text.delete(1.0, tk.END)
        self.log_message("🚀 开始处理...")
        
        self.callback_queue = Queue()
        
        self.worker = PerturbationWorker(
            self.image_path, self.mask_path, self.output_path,
            selected_perturbations, params, self.callback_queue
        )
        self.worker.start()
        
        self.check_callbacks()

    def check_callbacks(self):
        try:
            while not self.callback_queue.empty():
                msg_type, msg_data = self.callback_queue.get_nowait()
                if msg_type == 'progress':
                    self.update_progress(msg_data)
                elif msg_type == 'status':
                    self.update_status(msg_data)
                elif msg_type == 'error':
                    self.show_error(msg_data)
                elif msg_type == 'finished':
                    self.processing_finished()
                    return
        except:
            pass
        
        if hasattr(self, 'worker') and self.worker.is_alive():
            self.parent.after(100, self.check_callbacks)
    
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
