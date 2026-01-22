import cv2
import numpy as np
import cupy as cp
from typing import List
from cupyx.scipy.ndimage import gaussian_filter
import time

class BatchGaussianProcessor:
    """使用CuPy实现GPU高斯模糊 - 稳定可靠"""

    def __init__(self, gpu_id: int, sigma: float = 4.0, ksize: int = 21, num_streams: int = 4):
        self.gpu_id = gpu_id
        self.sigma = sigma
        self.ksize = ksize  # 保留但不使用（CuPy的gaussian_filter只需要sigma）
        
        # 设置CuPy使用的GPU
        with cp.cuda.Device(gpu_id):
            # 预热GPU
            dummy = cp.zeros((100, 100, 3), dtype=cp.uint8)
            _ = gaussian_filter(dummy[:, :, 0].astype(cp.float32), sigma=sigma)
        
        print(f"[GPU {gpu_id}] Worker initialized (CuPy): sigma={sigma:.2f}, PID={cp.cuda.runtime.getDevice()}")

    def process_batch(self, images_bytes: List[bytes], quality: int = 75) -> List[bytes]:
        """批量处理"""
        batch_size = len(images_bytes)
        start_time = time.perf_counter()
        
        results = []
        for idx, img_bytes in enumerate(images_bytes):
            try:
                result = self._process_single(img_bytes, quality)
                results.append(result)
            except Exception as e:
                print(f"[GPU {self.gpu_id}][Img {idx}] Error: {e}")
                raise
        
        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"[GPU {self.gpu_id}] ✅ Processed {batch_size} images in {elapsed:.2f}ms "
              f"({batch_size/(elapsed/1000):.1f} img/s)")
        
        return results

    def _process_single(self, img_bytes: bytes, quality: int) -> bytes:
        """
        单张处理流程：
        CPU解码 → GPU传输 → GPU高斯模糊 → CPU传输 → CPU编码
        """
        # 设置当前GPU
        with cp.cuda.Device(self.gpu_id):
            # 1. CPU解码
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            
            # 2. 转移到GPU（异步传输）
            gpu_img = cp.asarray(img)
            
            # 3. GPU高斯模糊（对每个通道并行处理）
            # 转换为float32以获得更好的精度
            gpu_img_float = gpu_img.astype(cp.float32)
            
            # 对BGR三个通道分别模糊
            blurred_b = gaussian_filter(gpu_img_float[:, :, 0], sigma=self.sigma, mode='reflect')
            blurred_g = gaussian_filter(gpu_img_float[:, :, 1], sigma=self.sigma, mode='reflect')
            blurred_r = gaussian_filter(gpu_img_float[:, :, 2], sigma=self.sigma, mode='reflect')
            
            # 合并通道并转回uint8
            gpu_blurred = cp.stack([blurred_b, blurred_g, blurred_r], axis=2)
            gpu_blurred = cp.clip(gpu_blurred, 0, 255).astype(cp.uint8)
            
            # 4. 转回CPU
            cpu_blurred = cp.asnumpy(gpu_blurred)
            
            # 5. CPU编码
            success, jpeg = cv2.imencode('.jpg', cpu_blurred, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not success:
                raise RuntimeError("Failed to encode JPEG")
            
            return jpeg.tobytes()
