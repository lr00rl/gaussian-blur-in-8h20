import multiprocessing as mp
import threading
from typing import List, Dict, Tuple
from queue import Queue
import time
from batch_processor import BatchGaussianProcessor
from config import settings

class GPUWorkerProcess:
    """GPU工作进程包装"""

    @staticmethod
    def worker_loop(
        gpu_id: int,
        task_queue: mp.Queue,
        result_queue: mp.Queue
    ):
        """工作进程主循环"""
        # 初始化processor
        processor = BatchGaussianProcessor(
            gpu_id=gpu_id,
            sigma=settings.gaussian_sigma,
            num_streams=settings.num_streams_per_gpu
        )

        print(f"[GPU {gpu_id}] Worker started")

        while True:
            task = task_queue.get()
            if task is None:  # Poison pill
                print(f"[GPU {gpu_id}] Shutting down")
                break

            request_id, images_bytes, quality = task

            try:
                start_time = time.perf_counter()

                # 处理batch
                results = processor.process_batch(images_bytes, quality)

                elapsed = (time.perf_counter() - start_time) * 1000
                print(f"[GPU {gpu_id}] Processed batch {request_id}: "
                      f"{len(images_bytes)} images in {elapsed:.2f}ms")

                result_queue.put((request_id, results, None))

            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing batch {request_id}: {e}")
                result_queue.put((request_id, None, str(e)))


class BatchGPUPool:
    """多GPU池管理器"""

    def __init__(self):
        self.num_gpus = settings.num_gpus
        self.gpu_ids = settings.gpu_ids[:self.num_gpus]

        # 队列
        queue_size = self.num_gpus * settings.queue_size_multiplier
        self.task_queue = mp.Queue(maxsize=queue_size)
        self.result_queue = mp.Queue()

        # 请求跟踪
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
        self.collector_thread = threading.Thread(
            target=self._collect_results,
            daemon=True
        )
        self.collector_thread.start()

        print(f"[GPUPool] Initialized with {self.num_gpus} GPUs")

    def _collect_results(self):
        """后台线程：收集处理结果"""
        while True:
            request_id, results, error = self.result_queue.get()

            with self.lock:
                if request_id in self.pending_requests:
                    future = self.pending_requests.pop(request_id)

                    if error:
                        future.set_exception(Exception(error))
                    else:
                        future.set_result(results)

    async def submit_batch(
        self,
        images_bytes: List[bytes],
        quality: int = 75
    ) -> List[bytes]:
        """提交batch任务"""
        import asyncio

        # 验证batch大小
        batch_size = len(images_bytes)
        if batch_size == 0:
            raise ValueError("Empty batch")
        if batch_size > settings.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds limit {settings.max_batch_size}"
            )

        # 创建Future
        with self.lock:
            request_id = self.request_counter
            self.request_counter += 1
            future = asyncio.Future()
            self.pending_requests[request_id] = future

        # 提交到队列
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.task_queue.put,
            (request_id, images_bytes, quality)
        )

        # 等待结果
        return await future

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self.lock:
            return {
                "num_gpus": self.num_gpus,
                "pending_requests": len(self.pending_requests),
                "queue_size": self.task_queue.qsize(),
            }

    def shutdown(self):
        """优雅关闭"""
        print("[GPUPool] Shutting down...")

        # 发送poison pill
        for _ in range(self.num_gpus):
            self.task_queue.put(None)

        # 等待进程结束
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        print("[GPUPool] Shutdown complete")


# 全局单例
_gpu_pool = None

def get_gpu_pool() -> BatchGPUPool:
    global _gpu_pool
    if _gpu_pool is None:
        _gpu_pool = BatchGPUPool()
    return _gpu_pool
