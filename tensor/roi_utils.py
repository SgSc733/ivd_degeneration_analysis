import numpy as np
from typing import Tuple, Optional, List

from utils import Preprocessor


def extract_disc_roi_3d(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    spacing_zyx: List[float],
    disc_label: int,
    roi_size: Tuple[int, int, int] = (64, 64, 32),
    target_spacing_mm: float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[float]]]:

    if image_array.shape != mask_array.shape:
        raise ValueError(f"image_array.shape {image_array.shape} 与 mask_array.shape {mask_array.shape} 不一致")

    preprocessor = Preprocessor()
    target_spacing = [target_spacing_mm, target_spacing_mm, target_spacing_mm]

    resampled_image, actual_spacing = preprocessor.resample_image(
        image_array,
        original_spacing=spacing_zyx,
        target_spacing=target_spacing,
        interpolation="linear",
        is_label=False,
    )
    resampled_mask, _ = preprocessor.resample_image(
        mask_array,
        original_spacing=spacing_zyx,
        target_spacing=target_spacing,
        interpolation="nearest",
        is_label=True,
    )

    disc_mask = (resampled_mask == int(disc_label)).astype(np.uint8)
    if not np.any(disc_mask):
        return None, None, None

    coords = np.argwhere(disc_mask > 0)
    center_z, center_y, center_x = coords.mean(axis=0)
    center_z = int(round(center_z))
    center_y = int(round(center_y))
    center_x = int(round(center_x))

    dz, dy, dx = roi_size
    z_dim, y_dim, x_dim = resampled_image.shape

    pad_z = max(0, dz - z_dim)
    pad_y = max(0, dy - y_dim)
    pad_x = max(0, dx - x_dim)

    if pad_z or pad_y or pad_x:
        pad_width = (
            (pad_z // 2, pad_z - pad_z // 2),
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_x // 2, pad_x - pad_x // 2),
        )
        resampled_image = np.pad(resampled_image, pad_width, mode="constant", constant_values=0)
        resampled_mask = np.pad(resampled_mask, pad_width, mode="constant", constant_values=0)
        disc_mask = np.pad(disc_mask, pad_width, mode="constant", constant_values=0)

        z_dim, y_dim, x_dim = resampled_image.shape
        coords = np.argwhere(disc_mask > 0)
        center_z, center_y, center_x = coords.mean(axis=0)
        center_z = int(round(center_z))
        center_y = int(round(center_y))
        center_x = int(round(center_x))

    start_z = max(0, min(center_z - dz // 2, z_dim - dz))
    start_y = max(0, min(center_y - dy // 2, y_dim - dy))
    start_x = max(0, min(center_x - dx // 2, x_dim - dx))
    end_z = start_z + dz
    end_y = start_y + dy
    end_x = start_x + dx

    roi_image = resampled_image[start_z:end_z, start_y:end_y, start_x:end_x].astype(np.float32)
    roi_mask = (resampled_mask[start_z:end_z, start_y:end_y, start_x:end_x] == int(disc_label)).astype(np.uint8)

    return roi_image, roi_mask, actual_spacing


def normalize_roi_intensity(
    roi_image: np.ndarray,
    roi_mask: np.ndarray,
    q_low: float = 1,
    q_high: float = 99,
) -> np.ndarray:

    if roi_image.shape != roi_mask.shape:
        raise ValueError("ROI 图像与掩码形状不一致")

    roi_values = roi_image[roi_mask > 0].astype(np.float32)
    if roi_values.size < 10:
        norm = roi_image.astype(np.float32).copy()
        norm[roi_mask == 0] = 0.0
        return norm

    low_val, high_val = np.percentile(roi_values, [q_low, q_high])
    if high_val <= low_val:
        clipped = roi_values
    else:
        clipped = np.clip(roi_values, low_val, high_val)

    mean_val = float(np.mean(clipped))
    std_val = float(np.std(clipped))
    if std_val < 1e-6:
        std_val = 1e-6

    norm = roi_image.astype(np.float32).copy()
    norm[roi_mask > 0] = (norm[roi_mask > 0] - mean_val) / std_val
    norm[roi_mask == 0] = 0.0
    return norm


def mode_n_unfold(tensor: np.ndarray, mode: int) -> np.ndarray:

    if tensor.ndim < 2:
        raise ValueError("mode_n_unfold 只支持阶数 >= 2 的张量")
    if mode < 0 or mode >= tensor.ndim:
        raise ValueError(f"mode 必须在 [0, {tensor.ndim - 1}] 范围内，当前为 {mode}")

    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def mode_n_fold(matrix: np.ndarray, mode: int, shape: Tuple[int, ...]) -> np.ndarray:

    if mode < 0 or mode >= len(shape):
        raise ValueError(f"mode 必须在 [0, {len(shape) - 1}] 范围内，当前为 {mode}")

    target_shape = [shape[mode]] + [s for i, s in enumerate(shape) if i != mode]
    tensor = matrix.reshape(target_shape)
    tensor = np.moveaxis(tensor, 0, mode)
    return tensor
