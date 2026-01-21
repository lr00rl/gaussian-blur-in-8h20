import cv2
import numpy as np
from typing import List

class BatchGaussianProcessor:
    """单GPU的Batch处理器，支持CUDA Stream并行"""
    
    def __init__(self, gpu_id: int, sigma: float = 4.0, ksize: int = 21, num_streams: int = 4):
        self.gpu_id = gpu_id
        self.sigma = sigma
        
        # 确保ksize是奇数且 <= 31
        if ksize <= 0:
            # 自动计算但限制在31以内
            ksize = min(int(round(sigma * 3) * 2 + 1), 31)
        
        self.ksize = ksize if ksize % 2 == 1 else ksize - 1
        
        if self.ksize > 31:
            self.ksize = 31
            print(f"[GPU {gpu_id}] Warning: ksize clamped to 31")
        
        # 设置GPU
        cv2.cuda.setDevice(gpu_id)
        
        # 创建CUDA Stream
        self.streams = [cv2.cuda_Stream() for _ in range(num_streams)]
        
        # 预创建滤波器
        try:
            self.filter = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC3, 
                cv2.CV_8UC3,
                ksize=(self.ksize, self.ksize),
                sigma1=self.sigma,
                sigma2=self.sigma
            )
            print(f"[GPU {gpu_id}] Filter created: ksize={self.ksize}, sigma={self.sigma:.2f}")
        except cv2.error as e:
            raise RuntimeError(f"GPU {gpu_id} filter creation failed: {e}")
    
    def process_batch(self, images_bytes: List[bytes], quality: int = 75) -> List[bytes]:
        """处理一个batch的图片"""
        batch_size = len(images_bytes)
        
        if batch_size <= 3:
            return [self._process_single(img_bytes, quality) for img_bytes in images_bytes]
        else:
            return self._process_batch_parallel(images_bytes, quality)
    
    def _process_single(self, img_bytes: bytes, quality: int) -> bytes:
        """单张图片处理"""
        # CPU解码
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        
        # GPU处理
        gpu_in = cv2.cuda_GpuMat()
        gpu_in.upload(img)
        gpu_out = self.filter.apply(gpu_in)
        result = gpu_out.download()
        
        # CPU编码
        success, jpeg = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError("Failed to encode JPEG")
        
        return jpeg.tobytes()
    
    def _process_batch_parallel(self, images_bytes: List[bytes], quality: int) -> List[bytes]:
        """多张图片并行处理"""
        num_streams = len(self.streams)
        
        # 阶段1: CPU批量解码
        cpu_images = []
        for img_bytes in images_bytes:
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            cpu_images.append(img)
        
        # 阶段2: GPU并行处理
        gpu_inputs = []
        gpu_outputs = []
        
        for i, img in enumerate(cpu_images):
            stream_idx = i % num_streams
            stream = self.streams[stream_idx]
            
            # 异步上传
            gpu_in = cv2.cuda_GpuMat()
            gpu_in.upload(img, stream=stream)
            gpu_inputs.append(gpu_in)
            
            # 异步模糊
            gpu_out = cv2.cuda_GpuMat()
            self.filter.apply(gpu_in, gpu_out, stream=stream)
            gpu_outputs.append(gpu_out)
        
        # 同步所有stream
        for stream in self.streams:
            stream.waitForCompletion()
        
        # 阶段3: CPU批量编码
        results = []
        for gpu_out in gpu_outputs:
            result = gpu_out.download()
            success, jpeg = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not success:
                raise RuntimeError("Failed to encode JPEG")
            results.append(jpeg.tobytes())
        
        return results
