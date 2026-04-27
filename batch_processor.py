import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List


class BatchGaussianProcessor:
    """CPU版本 - 简单稳定，不再折腾GPU。

    优化点（vs 旧版）:
      - process_batch 内 32 张图改为多线程并行
      - cv2.imdecode / cv2.GaussianBlur / cv2.imencode 都释放 GIL，能吃多核
      - 单 batch wall-time 从 ~2.5s 降到 ~1.3s（取决于 intra_threads 与图片大小）
    """

    def __init__(self, gpu_id: int, sigma: float = 4.0, ksize: int = 21, num_streams: int = 4):
        self.gpu_id = gpu_id  # 保留参数但不用GPU
        self.sigma = sigma

        if ksize <= 0:
            ksize = min(int(round(sigma * 3) * 2 + 1), 31)
        self.ksize = ksize if ksize % 2 == 1 else ksize - 1
        if self.ksize > 31:
            self.ksize = 31

        # batch 内并行：cv2 释放 GIL，多线程吃多核
        # 控制：num_streams=4 → intra_threads=2；num_workers × intra_threads ≤ 1.5 × 物理核
        self._intra_threads = max(1, num_streams // 2)
        self._executor = ThreadPoolExecutor(
            max_workers=self._intra_threads,
            thread_name_prefix=f"blur-w{gpu_id}",
        )

        print(f"[Worker {gpu_id}] Initialized (CPU mode): ksize={self.ksize}, sigma={self.sigma:.2f}, intra_threads={self._intra_threads}")

    def process_batch(self, images_bytes: List[bytes], quality: int = 75) -> List[bytes]:
        """批量处理 - batch 内多线程并行（保证返回顺序与输入一致）"""
        # ThreadPoolExecutor.map 顺序保证 + GIL release 路径吃多核
        return list(self._executor.map(
            lambda b: self._process_single(b, quality),
            images_bytes,
        ))

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
