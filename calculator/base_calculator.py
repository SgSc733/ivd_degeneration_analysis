from abc import ABC, abstractmethod
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional
import multiprocessing as mp
import numpy as np
import logging


class BaseCalculator(ABC):
    
    def __init__(self, name: str, enable_parallel: bool = True):
        self.name = name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.enable_parallel = enable_parallel
        self.max_workers = mp.cpu_count() // 2
    
    @abstractmethod
    def calculate(self, *args, **kwargs) -> Dict[str, Any]:
        pass
    
    def calculate_parallel(self, *args, **kwargs) -> Dict[str, Any]:
        return self.calculate(*args, **kwargs)
    
    def set_parallel_workers(self, max_workers: int):
        self.max_workers = max_workers
    
    def validate_input(self, image: np.ndarray, mask: np.ndarray) -> None:
        if image is None or mask is None:
            raise ValueError("图像和掩模不能为空")
            
        if image.shape != mask.shape:
            raise ValueError(f"图像形状 {image.shape} 与掩模形状 {mask.shape} 不匹配")
            
        if not np.any(mask):
            raise ValueError("掩模中没有有效像素")
