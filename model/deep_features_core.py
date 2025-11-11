import os
import warnings
import numpy as np
import torch
import timm
import nibabel as nib
import cv2
import pandas as pd
from scipy.ndimage import center_of_mass
from torchvision import transforms
import urllib.request
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_URLS = {
    "base": "https://huggingface.co/Snarcy/RadioDino-b16/resolve/main/pytorch_model.bin?download=true",
    "small": "https://huggingface.co/Snarcy/RadioDino-s16/resolve/main/pytorch_model.bin?download=true"
}

class TqdmUpTo(tqdm):

    def update_to(self, b=1, bsize=1, tsize=None):

        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def preprocess_slice(image_slice_2d, mask_slice_2d, view_name, case_id, padding_ratio):
    if image_slice_2d.max() > 0:
        img_normalized_8bit = cv2.normalize(image_slice_2d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        img_normalized_8bit = image_slice_2d.astype(np.uint8)

    rows = np.any(mask_slice_2d, axis=1)
    cols = np.any(mask_slice_2d, axis=0)
    if not np.any(rows) or not np.any(cols):
        print(f"    - 警告: 在视图 {view_name} 中未找到掩码区域，跳过此视图。")
        return None, None
        
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    padding_y = int((rmax - rmin) * padding_ratio)
    padding_x = int((cmax - cmin) * padding_ratio)

    rmin = max(0, rmin - padding_y)
    rmax = min(image_slice_2d.shape[0], rmax + padding_y)
    cmin = max(0, cmin - padding_x)
    cmax = min(image_slice_2d.shape[1], cmax + padding_x)

    cropped_image = img_normalized_8bit[rmin:rmax, cmin:cmax]
    cropped_mask = mask_slice_2d[rmin:rmax, cmin:cmax]

    resized_image = cv2.resize(cropped_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
    
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform_pipeline(resized_image).unsqueeze(0)
    
    return image_tensor, resized_mask

def get_feature_vector_from_slice(model, image_tensor, mask_224, device, aggregation_strategy):
    model.eval()
    with torch.no_grad():
        intermediate_output = model.get_intermediate_layers(image_tensor.to(device), n=1)[0]

        patch_tokens = intermediate_output

        patch_grid_size = 14
        
        patch_mask = cv2.resize(mask_224.astype(np.uint8), 
                                (patch_grid_size, patch_grid_size), 
                                interpolation=cv2.INTER_NEAREST)
        
        patch_mask_binary = (patch_mask > 0).astype(bool)

        flat_patch_mask = patch_mask_binary.flatten()
        
        if not np.any(flat_patch_mask):
            print("    - 警告: 在patch网格中未找到对应的掩码区域。返回零向量。")
            embed_dim = patch_tokens.shape[-1]
            return torch.zeros(embed_dim, device=device)
            
        patch_tokens_flat = patch_tokens.squeeze(0)
        
        selected_tokens = patch_tokens_flat[flat_patch_mask, :]
        
        if aggregation_strategy == 'mean':
            feature_vector = torch.mean(selected_tokens, dim=0)
        elif aggregation_strategy == 'max':
            feature_vector = torch.max(selected_tokens, dim=0)[0]
        elif aggregation_strategy == 'both':
            mean_vec = torch.mean(selected_tokens, dim=0)
            max_vec = torch.max(selected_tokens, dim=0)[0]
            feature_vector = torch.cat((mean_vec, max_vec))
        else:
            raise ValueError(f"不支持的聚合策略: {aggregation_strategy}")

    return feature_vector

def extract_disc_features(image_3d, mask_3d, disc_label, disc_name, model, device, 
                          case_id, padding_ratio, aggregation_strategy, model_name_str, base_embed_dim):
    
    
    binary_mask_3d = (mask_3d == disc_label)
    if not np.any(binary_mask_3d):
        print(f"    - 错误: 在掩码文件中未找到标签为 {disc_label} 的区域。")
        return None, None
        
    z_center, y_center, x_center = map(int, center_of_mass(binary_mask_3d))
    
    x_dim_size = image_3d.shape[2]
    middle_x_index = x_dim_size // 2
    
    img_sagittal = image_3d[:, :, middle_x_index]
    mask_sagittal = binary_mask_3d[:, :, middle_x_index]
    
    img_coronal = image_3d[:, y_center, :]
    mask_coronal = binary_mask_3d[:, y_center, :]

    img_axial = image_3d[z_center, :, :]
    mask_axial = binary_mask_3d[z_center, :, :]
    
    views = {
        'sagittal': (img_sagittal, mask_sagittal),
        'coronal': (img_coronal, mask_coronal),
        'axial': (img_axial, mask_axial)
    }

    all_view_features = []
    feature_names = []
    

    view_order = ['axial', 'sagittal', 'coronal']

    single_view_embed_dim = base_embed_dim * 2 if aggregation_strategy == 'both' else base_embed_dim

    for view_name in view_order:
        img_slice, mask_slice = views[view_name]
        
        image_tensor, resized_mask = preprocess_slice(img_slice, mask_slice, f"{disc_name}_{view_name}", case_id, padding_ratio)
        
        if image_tensor is None:
            feature_vector = torch.zeros(single_view_embed_dim, device=device)
        else:
            feature_vector = get_feature_vector_from_slice(model, image_tensor, resized_mask, device, aggregation_strategy)

        all_view_features.append(feature_vector)
        
        if aggregation_strategy == 'both':
            for i in range(base_embed_dim):
                feature_names.append(f"Deep_{disc_name}_{model_name_str}_{view_name}_mean_dim_{i}")
            for i in range(base_embed_dim):
                feature_names.append(f"Deep_{disc_name}_{model_name_str}_{view_name}_max_dim_{i}")
        else:
             for i in range(base_embed_dim):
                feature_names.append(f"Deep_{disc_name}_{model_name_str}_{view_name}_{aggregation_strategy}_dim_{i}")

    final_disc_vector = torch.cat(all_view_features).cpu().numpy()
    
    return final_disc_vector, feature_names

def download_model_if_needed(model_size, logger_callback=print):

    url = MODEL_URLS.get(model_size)
    if not url:
        logger_callback(f"!!! 错误: 未找到模型大小 '{model_size}' 的下载链接。")
        return False

    model_dir = os.path.join("./model", model_size)
    weights_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(weights_path):
        return True

    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError as e:
        logger_callback(f"!!! 错误: 创建文件夹 {model_dir} 失败: {e}")
        return False

    try:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                      desc=f"下载 {model_size} 模型") as t:
            urllib.request.urlretrieve(url, filename=weights_path, reporthook=t.update_to)
        logger_callback(f"✓ 模型 {model_size} 下载完成")
        return True
    except Exception as e:
        logger_callback(f"!!! 错误: 下载模型 {model_size} 失败: {e}")
        if os.path.exists(weights_path):
            os.remove(weights_path)
        return False

def load_deep_model(model_size, device, logger_callback=print):
    success = download_model_if_needed(model_size, logger_callback)
    if not success:
        raise FileNotFoundError(f"无法下载或找到模型 '{model_size}' 的权重文件。请检查网络连接或手动下载。")

    if model_size == "small":
        model_architecture = 'vit_small_patch16_224'
        local_weights_path = "./model/small/pytorch_model.bin"
    elif model_size == "base":
        model_architecture = 'vit_base_patch16_224'
        local_weights_path = "./model/base/pytorch_model.bin"
    else:
        raise ValueError(f"不支持的模型大小: {model_size}。请选择 'small' 或 'base'。")

    if not os.path.exists(local_weights_path):
        raise FileNotFoundError(f"模型权重文件未找到: {local_weights_path}")

    model = timm.create_model(model_architecture, pretrained=False, num_classes=0)
    state_dict = torch.load(local_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def extract_deep_features_for_case(image_path, mask_path, config, model_size, agg_strategy, padding_ratio, logger_callback=print):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger_callback(f"使用设备: {device} 加载 {model_size} 模型...")
    
    try:
        model = load_deep_model(model_size, device, logger_callback)
    except Exception as e:
        logger_callback(f"错误: 无法加载深度学习模型: {e}")
        return None

    if model_size == "small":
        model_name_str = "small"
        base_embed_dim = 384
    else:
        model_name_str = "base"
        base_embed_dim = 768
    
    final_embed_dim = base_embed_dim * 2 if agg_strategy == 'both' else base_embed_dim

    try:
        img_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)
        image_3d_data = img_nii.get_fdata()
        mask_3d_data = mask_nii.get_fdata()
    except Exception as e:
        logger_callback(f"错误: 无法加载NIfTI文件: {e}")
        return None

    case_id = os.path.basename(image_path).replace('.nii.gz', '').replace('.nii', '')
    case_features = {'case_id': case_id}

    for disc_name, labels in config.DISC_LABELS.items():
        disc_label = labels['disc']
        
        logger_callback(f"  -> 正在处理椎间盘: {disc_name}")
        
        disc_vector, feature_names = extract_disc_features(
            image_3d_data, mask_3d_data, disc_label, disc_name, model, device, 
            case_id, padding_ratio, agg_strategy, model_name_str, base_embed_dim
        )
        
        if disc_vector is not None and feature_names:
            num_features = len(disc_vector)
            logger_callback(f"    - 成功提取 {num_features} 个 {disc_name} 的深度学习特征")
            for i, fname in enumerate(feature_names):
                case_features[fname] = disc_vector[i]
        else:
            logger_callback(f"    - 警告: 椎间盘 {disc_name} 未能提取特征，将填充NaN。")
            view_order = ['axial', 'sagittal', 'coronal']
            for view_name in view_order:
                if agg_strategy == 'both':
                    for i in range(base_embed_dim):
                        case_features[f"Deep_{disc_name}_{model_name_str}_{view_name}_mean_dim_{i}"] = np.nan
                    for i in range(base_embed_dim):
                        case_features[f"Deep_{disc_name}_{model_name_str}_{view_name}_max_dim_{i}"] = np.nan
                else:
                    for i in range(base_embed_dim):
                        case_features[f"Deep_{disc_name}_{model_name_str}_{view_name}_{agg_strategy}_dim_{i}"] = np.nan

    return case_features
