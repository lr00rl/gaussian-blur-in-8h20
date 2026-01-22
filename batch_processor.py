import cv2
import numpy as np
from typing import List

class BatchGaussianProcessor:
    """CPU版本 - 简单稳定，不再折腾GPU"""

    def __init__(self, gpu_id: int, sigma: float = 4.0, ksize: int = 21, num_streams: int = 4):
        self.gpu_id = gpu_id  # 保留参数但不用GPU
        self.sigma = sigma

        if ksize <= 0:
            ksize = min(int(round(sigma * 3) * 2 + 1), 31)
        self.ksize = ksize if ksize % 2 == 1 else ksize - 1
        if self.ksize > 31:
            self.ksize = 31

        print(f"[Worker {gpu_id}] Initialized (CPU mode): ksize={self.ksize}, sigma={self.sigma:.2f}")

    def process_batch(self, images_bytes: List[bytes], quality: int = 75) -> List[bytes]:
        """批量处理"""
        results = []
        for idx, img_bytes in enumerate(images_bytes):
            result = self._process_single(img_bytes, quality)
            results.append(result)
        return results

    def _process_single(self, img_bytes: bytes, quality: int) -> bytes:
        """单张处理 - 纯CPU"""
        # 解码
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        
        # CPU高斯模糊
        blurred = cv2.GaussianBlur(img, (self.ksize, self.ksize), self.sigma)
        
        # 编码
        success, jpeg = cv2.imencode(".jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError("Failed to encode JPEG")
        
        return jpeg.tobytes()
