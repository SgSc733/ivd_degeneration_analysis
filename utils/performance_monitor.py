import time
import psutil
import logging
from functools import wraps

def monitor_performance(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  

        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} - 执行时间: {end_time - start_time:.2f}秒, "
                   f"内存使用: {end_memory - start_memory:.2f}MB")
        
        return result
    return wrapper