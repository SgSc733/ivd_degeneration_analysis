from abc import ABC, abstractmethod
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional
import multiprocessing as mp
import numpy as np
import logging


class BaseCalculator(ABC):
    
    def __init__(self, name: str, enable_parallel: bool = True, logger_callback=None, debug_mode: bool = False):
        self.name = name
        self.debug_mode = debug_mode
        if logger_callback:
            self.logger = GuiLoggerProxy(logger_callback, debug_mode=self.debug_mode)
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
        self.enable_parallel = enable_parallel
        self.max_workers = mp.cpu_count() // 2

    def set_debug_mode(self, enabled: bool):

        self.debug_mode = enabled
        if isinstance(self.logger, GuiLoggerProxy):
            self.logger.set_debug_mode(enabled)
    
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


class GuiLoggerProxy:

    def __init__(self, callback, debug_mode: bool = False):
        self.callback = callback
        self.debug_enabled = debug_mode

    def set_debug_mode(self, enabled: bool):

        self.debug_enabled = enabled
        status = "开启" if enabled else "关闭"
        self.info(f"调试信息显示已{status}。")

    def info(self, msg):
        if self.debug_enabled:
            self.callback(f"[INFO] {msg}")

    def warning(self, msg):
        self.callback(f"[WARNING] {msg}")

    def error(self, msg):
        self.callback(f"[ERROR] {msg}")

    def debug(self, msg):
        if self.debug_enabled:
            self.callback(f"[DEBUG] {msg}")
