import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

from config import Config


def estimate_tensor_roi_size(
    image_mask_pairs: Iterable[Tuple[Path, Path]],
    margin_mm: float = 5.0,
    round_base: int = 8,
) -> Tuple[int, int, int, Dict[str, float]]:

    extents_z_mm: List[float] = []
    extents_y_mm: List[float] = []
    extents_x_mm: List[float] = []

    pairs = list(image_mask_pairs)
    if not pairs:
        raise ValueError("estimate_tensor_roi_size: 提供的图像/掩码对为空。")

    for img_path, mask_path in pairs:
        img_path = Path(img_path)
        mask_path = Path(mask_path)
        if not img_path.exists() or not mask_path.exists():
            continue

        img = sitk.ReadImage(str(img_path))
        mask = sitk.ReadImage(str(mask_path))

        sx, sy, sz = img.GetSpacing()
        mask_arr = sitk.GetArrayFromImage(mask)

        for level_name, labels in Config.DISC_LABELS.items():
            disc_label = labels["disc"]
            disc_mask = (mask_arr == disc_label)
            coords = np.argwhere(disc_mask)
            if coords.size == 0:
                continue

            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0)

            extent_z_mm = (z_max - z_min + 1) * sz
            extent_y_mm = (y_max - y_min + 1) * sy
            extent_x_mm = (x_max - x_min + 1) * sx

            extents_z_mm.append(extent_z_mm)
            extents_y_mm.append(extent_y_mm)
            extents_x_mm.append(extent_x_mm)

    if not extents_z_mm:
        raise ValueError("estimate_tensor_roi_size: 所有图像中都未找到任何椎间盘 ROI。")

    pz = float(np.percentile(extents_z_mm, 98))
    py = float(np.percentile(extents_y_mm, 98))
    px = float(np.percentile(extents_x_mm, 98))

    min_z, max_z = float(min(extents_z_mm)), float(max(extents_z_mm))
    min_y, max_y = float(min(extents_y_mm)), float(max(extents_y_mm))
    min_x, max_x = float(min(extents_x_mm)), float(max(extents_x_mm))

    roi_z_mm = pz + 2 * margin_mm
    roi_y_mm = py + 2 * margin_mm
    roi_x_mm = px + 2 * margin_mm

    target_spacing_mm = float(Config.TENSOR_ROI_PARAMS.get("target_spacing_mm", 1.0))
    roi_z_vox = int(np.ceil(roi_z_mm / target_spacing_mm))
    roi_y_vox = int(np.ceil(roi_y_mm / target_spacing_mm))
    roi_x_vox = int(np.ceil(roi_x_mm / target_spacing_mm))

    def round_to_multiple(x: int, base: int = 8) -> int:
        return int(np.ceil(x / base) * base)

    roi_z_vox_r = round_to_multiple(roi_z_vox, round_base)
    roi_y_vox_r = round_to_multiple(roi_y_vox, round_base)
    roi_x_vox_r = round_to_multiple(roi_x_vox, round_base)

    stats = {
        "min_z_mm": min_z,
        "max_z_mm": max_z,
        "p98_z_mm": pz,
        "min_y_mm": min_y,
        "max_y_mm": max_y,
        "p98_y_mm": py,
        "min_x_mm": min_x,
        "max_x_mm": max_x,
        "p98_x_mm": px,
        "roi_z_mm": roi_z_mm,
        "roi_y_mm": roi_y_mm,
        "roi_x_mm": roi_x_mm,
        "target_spacing_mm": target_spacing_mm,
        "margin_mm": margin_mm,
        "num_pairs": len(pairs),
    }

    return roi_z_vox_r, roi_y_vox_r, roi_x_vox_r, stats


def main():
    root = Path("test")
    image_files = sorted(
        [p for p in root.glob("*.nii.gz") if not p.name.endswith("_mask.nii.gz")]
    )

    if not image_files:
        return

    pairs = []
    for img_path in image_files:
        mask_path = img_path.with_name(img_path.stem + "_mask.nii.gz")
        if not mask_path.exists():
            print(f"[跳过] 未找到掩码: {mask_path.name}")
            continue
        pairs.append((img_path, mask_path))

    if not pairs:
        print("未找到任何有效的 图像/掩码 对。")
        return

    roi_z, roi_y, roi_x, stats = estimate_tensor_roi_size(pairs)

    print("\n==== 建议的张量 ROI 尺寸 ====")
    print(f"  Z ≈ {stats['roi_z_mm']:.1f}mm -> 建议取 {roi_z} 体素")
    print(f"  Y ≈ {stats['roi_y_mm']:.1f}mm -> 建议取 {roi_y} 体素")
    print(f"  X ≈ {stats['roi_x_mm']:.1f}mm -> 建议取 {roi_x} 体素")


if __name__ == "__main__":
    main()
