import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Any, Optional
import numpy as np
import logging
from tqdm import tqdm
import gc
import psutil
import os

class ParallelProcessor:
    
    def __init__(self, max_workers: Optional[int] = None, 
                 backend: str = 'multiprocessing',
                 show_progress: bool = True):

        self.backend = backend
        self.show_progress = show_progress

        if max_workers is None:
            self.max_workers = mp.cpu_count()
        else:
            self.max_workers = min(max_workers, mp.cpu_count())
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def process_batch(self, func: Callable, items: List[Any], 
                     func_args: Optional[Dict] = None,
                     desc: str = "Processing") -> List[Any]:

        if func_args is None:
            func_args = {}
        
        results = [None] * len(items)

        Executor = ProcessPoolExecutor if self.backend == 'multiprocessing' else ThreadPoolExecutor
        
        with Executor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(func, item, **func_args): idx 
                for idx, item in enumerate(items)
            }

            if self.show_progress:
                futures = tqdm(as_completed(future_to_idx), 
                             total=len(items), 
                             desc=desc)
            else:
                futures = as_completed(future_to_idx)
            
            for future in futures:
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    self.logger.error(f"处理项目 {idx} 时出错: {str(e)}")
                    results[idx] = {'error': str(e)}
                    
        return results
    
    def process_batch_with_memory_limit(self, func: Callable, items: List[Any],
                                      max_memory_gb: float = 8,
                                      **kwargs) -> List[Any]:

        results = []
        current_batch = []
        
        for item in items:
            current_batch.append(item)

            memory_usage_gb = psutil.virtual_memory().used / (1024**3)
            
            if memory_usage_gb > max_memory_gb * 0.8: 
                batch_results = self.process_batch(func, current_batch, **kwargs)
                results.extend(batch_results)
                current_batch = []

                gc.collect()

        if current_batch:
            batch_results = self.process_batch(func, current_batch, **kwargs)
            results.extend(batch_results)
            
        return results

def parallel_feature_extraction(args):

    case_info, feature_set, config = args

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from main import IVDAnalysisSystem
    
    try:
        system = IVDAnalysisSystem(config)

        df = system.analyze_single_case(
            case_info['image_path'],
            case_info['mask_path'],
            case_info.get('case_id'),
            feature_set,
            case_info.get('spacing')
        )

        results = {
            'status': 'success',
            'case_id': case_info['case_id'],
            'results': df.to_dict('records')
        }
        
    except Exception as e:
        results = {
            'status': 'failed',
            'error': str(e),
            'case_id': case_info['case_id']
        }
    
    return results