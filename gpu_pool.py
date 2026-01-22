import multiprocessing as mp

# ✅ 强制使用spawn模式 - 避免fork导致的CUDA context冲突
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，忽略
    pass

import threading
from typing import List, Dict
import time
from batch_processor import BatchGaussianProcessor
from config import settings

class GPUWorkerProcess:
    @staticmethod
    def worker_loop(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue):
        """
        Worker进程主循环
        
        ✅ 关键：在spawn模式下，这个函数在全新的进程中运行
        不会继承主进程的任何CUDA状态
        """
        try:
            # 在新进程中初始化GPU资源
            processor = BatchGaussianProcessor(
                gpu_id=gpu_id,
                sigma=settings.gaussian_sigma,
                ksize=settings.gaussian_ksize,
                num_streams=settings.num_streams_per_gpu
            )
            print(f"[GPU {gpu_id}] Worker started (PID: {mp.current_process().pid})")
        except Exception as e:
            print(f"[GPU {gpu_id}] Init failed: {e}")
            result_queue.put((-1, None, f"GPU {gpu_id} init failed: {e}"))
            return
        
        while True:
            try:
                task = task_queue.get(timeout=1)
            except:
                continue
            
            if task is None:
                print(f"[GPU {gpu_id}] Shutting down")
                break
            
            request_id, images_bytes, quality = task
            
            try:
                start_time = time.perf_counter()
                results = processor.process_batch(images_bytes, quality)
                elapsed = (time.perf_counter() - start_time) * 1000
                
                print(f"[GPU {gpu_id}] Batch {request_id}: {len(images_bytes)} imgs in {elapsed:.2f}ms")
                result_queue.put((request_id, results, None))
            except Exception as e:
                print(f"[GPU {gpu_id}] Error: {e}")
                result_queue.put((request_id, None, str(e)))


class BatchGPUPool:
    def __init__(self):
        self.num_gpus = settings.num_gpus
        self.gpu_ids = settings.gpu_ids[:self.num_gpus]
        
        queue_size = self.num_gpus * settings.queue_size_multiplier
        
        # ✅ 使用multiprocessing.Manager创建Queue，兼容spawn模式
        manager = mp.Manager()
        self.task_queue = manager.Queue(maxsize=queue_size)
        self.result_queue = manager.Queue()
        
        self.pending_requests: Dict[int, 'asyncio.Future'] = {}
        self.request_counter = 0
        self.lock = threading.Lock()
        
        # 启动worker进程
        self.processes = []
        for gpu_id in self.gpu_ids:
            p = mp.Process(
                target=GPUWorkerProcess.worker_loop,
                args=(gpu_id, self.task_queue, self.result_queue),
                daemon=False
            )
            p.start()
            self.processes.append(p)
        
        # 启动结果收集线程
        self.collector_thread = threading.Thread(target=self._collect_results, daemon=True)
        self.collector_thread.start()
        
        print(f"[GPUPool] Initialized with {self.num_gpus} GPUs (spawn mode)")
    
    def _collect_results(self):
        while True:
            request_id, results, error = self.result_queue.get()
            
            if request_id == -1:
                continue
            
            with self.lock:
                if request_id in self.pending_requests:
                    future = self.pending_requests.pop(request_id)
                    if error:
                        future.set_exception(Exception(error))
                    else:
                        future.set_result(results)
    
    async def submit_batch(self, images_bytes: List[bytes], quality: int = 75) -> List[bytes]:
        import asyncio
        
        batch_size = len(images_bytes)
        if batch_size == 0:
            raise ValueError("Empty batch")
        if batch_size > settings.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds limit")
        
        with self.lock:
            request_id = self.request_counter
            self.request_counter += 1
            future = asyncio.Future()
            self.pending_requests[request_id] = future
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.task_queue.put, (request_id, images_bytes, quality))
        
        return await future
    
    def get_stats(self) -> dict:
        alive = sum(1 for p in self.processes if p.is_alive())
        with self.lock:
            return {
                "num_gpus": self.num_gpus,
                "alive_workers": alive,
                "pending_requests": len(self.pending_requests),
                "queue_size": self.task_queue.qsize(),
            }
    
    def shutdown(self):
        print("[GPUPool] Shutting down...")
        for _ in range(self.num_gpus):
            try:
                self.task_queue.put(None, timeout=1)
            except:
                pass
        
        for p in self.processes:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        
        print("[GPUPool] Shutdown complete")


_gpu_pool = None

def get_gpu_pool() -> BatchGPUPool:
    global _gpu_pool
    if _gpu_pool is None:
        _gpu_pool = BatchGPUPool()
    return _gpu_pool
