import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, List
import logging


class ImageIO:
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_image(self, image_path: Union[str, Path]) -> sitk.Image:

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        self.logger.info(f"加载图像: {image_path}")
        image = sitk.ReadImage(str(image_path))
        return image
    
    def load_image_and_mask(self, image_path: Union[str, Path], 
                           mask_path: Union[str, Path]) -> Tuple[sitk.Image, sitk.Image]:

        image = self.load_image(image_path)
        mask = self.load_image(mask_path)

        if image.GetSize() != mask.GetSize():
            raise ValueError(f"图像尺寸 {image.GetSize()} 与掩模尺寸 {mask.GetSize()} 不匹配")
            
        return image, mask
    
    def sitk_to_numpy(self, image: sitk.Image) -> np.ndarray:
        return sitk.GetArrayFromImage(image)
    
    def numpy_to_sitk(self, array: np.ndarray, reference_image: Optional[sitk.Image] = None) -> sitk.Image:

        image = sitk.GetImageFromArray(array)
        
        if reference_image is not None:
            image.CopyInformation(reference_image)
            
        return image
    
    def save_image(self, image: sitk.Image, output_path: Union[str, Path]) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"保存图像: {output_path}")
        sitk.WriteImage(image, str(output_path))
    
    def extract_slice(self, image: sitk.Image, slice_idx: int, axis: int = 2) -> np.ndarray:

        array = self.sitk_to_numpy(image)
        
        if axis == 0:
            return array[slice_idx, :, :]
        elif axis == 1:
            return array[:, slice_idx, :]
        else:
            return array[:, :, slice_idx]
    
    def extract_middle_slices(self, image: sitk.Image, num_slices: int = 3, 
                            axis: int = 0) -> List[np.ndarray]:

        array = self.sitk_to_numpy(image)
        size = array.shape[axis]

        middle = size // 2
        half_num = num_slices // 2
        
        start_idx = max(0, middle - half_num)
        end_idx = min(size, start_idx + num_slices)
        
        slices = []
        for i in range(start_idx, end_idx):
            slices.append(self.extract_slice(image, i, axis))
            
        return slices
    
    def find_files(self, directory: Union[str, Path], 
                  pattern: str = "*.nii*",
                  recursive: bool = True) -> List[Path]:

        directory = Path(directory)
        
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
            
        return sorted(files)