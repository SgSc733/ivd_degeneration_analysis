from .patch_tensor_features import PatchTensorFeatures

# tensorly 在某些最小化测试环境中可能未安装；此时仍允许导入 ROI/patch 相关工具。
try:
    from .global_tucker_features import GlobalTuckerTensorFeatures
except ModuleNotFoundError as e:
    if getattr(e, "name", None) == "tensorly":
        GlobalTuckerTensorFeatures = None  # type: ignore[assignment]
    else:
        raise

try:
    from .cp_tensor_features import CPTensorFeatures
except ModuleNotFoundError as e:
    if getattr(e, "name", None) == "tensorly":
        CPTensorFeatures = None  # type: ignore[assignment]
    else:
        raise
from .roi_utils import (
    extract_disc_roi_3d,
    normalize_roi_intensity,
    mode_n_unfold,
    mode_n_fold,
)

__all__ = [
    "GlobalTuckerTensorFeatures",
    "PatchTensorFeatures",
    "CPTensorFeatures",
    "extract_disc_roi_3d",
    "normalize_roi_intensity",
    "mode_n_unfold",
    "mode_n_fold",
]
