import functools
import psutil
import gc
import logging
from typing import Callable, Any

def monitor_memory(threshold_percent: float = 80):

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            memory_before = psutil.virtual_memory()
            
            if memory_before.percent > threshold_percent:
                logging.warning(f"{func.__name__} - 内存使用率高: {memory_before.percent}%")
                gc.collect()
            
            try:
                result = func(*args, **kwargs)

                memory_after = psutil.virtual_memory()
                memory_increase = memory_after.percent - memory_before.percent
                
                if memory_increase > 10:
                    logging.info(f"{func.__name__} - 内存增加: {memory_increase:.1f}%")
                
                return result
                
            except MemoryError:
                logging.error(f"{func.__name__} - 内存不足")
                gc.collect()
                raise
                
        return wrapper
    return decorator