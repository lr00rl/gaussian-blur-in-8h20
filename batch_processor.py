import cv2
import numpy as np
from typing import List, Tuple
import cupy as cp

class BatchGaussianProcessor:
    """单GPU的Batch处理器，支持CUDA Stream并行"""

    def __init__(self, gpu_id: int, sigma: float = 6.0, num_streams: int = 4):
        self.gpu_id = gpu_id
        self.sigma = sigma
        cv2.cuda.setDevice(gpu_id)

        # 预创建多个CUDA Stream实现GPU内并行
        self.streams = [cv2.cuda_Stream() for _ in range(num_streams)]

        # 预创建滤波器（所有stream共享）
        self.filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3, cv2.CV_8UC3,
            ksize=(0, 0),
            sigma1=sigma,
            sigma2=sigma
        )

    def process_batch(
        self,
        images_bytes: List[bytes],
        quality: int = 75
    ) -> List[bytes]:
        """
        处理一个batch的图片

        策略：
        1. 小batch (<4张): 串行处理
        2. 大batch (>=4张): 使用多stream并行
        """
        batch_size = len(images_bytes)

        if batch_size <= 3:
            # 小batch直接串行处理（避免stream切换开销）
            return [self._process_single(img_bytes, quality)
                    for img_bytes in images_bytes]
        else:
            # 大batch使用stream并行
            return self._process_batch_parallel(images_bytes, quality)

    def _process_single(self, img_bytes: bytes, quality: int) -> bytes:
        """单张图片处理（无stream）"""
        # CPU解码
        img = cv2.imdecode(
            np.frombuffer(img_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        # GPU处理
        gpu_in = cv2.cuda_GpuMat()
        gpu_in.upload(img)
        gpu_out = self.filter.apply(gpu_in)
        result = gpu_out.download()

        # CPU编码
        _, jpeg = cv2.imencode(
            ".jpg", result,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return jpeg.tobytes()

    def _process_batch_parallel(
        self,
        images_bytes: List[bytes],
        quality: int
    ) -> List[bytes]:
        """多张图片并行处理（使用CUDA Stream）"""
        batch_size = len(images_bytes)
        num_streams = len(self.streams)

        # 预分配GPU内存（复用，减少malloc开销）
        gpu_inputs = []
        gpu_outputs = []
        cpu_images = []

        # 阶段1: CPU解码（多线程）
        for img_bytes in images_bytes:
            img = cv2.imdecode(
                np.frombuffer(img_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )
            cpu_images.append(img)

        # 阶段2: 批量上传+处理（Stream并行）
        for i, img in enumerate(cpu_images):
            stream_idx = i % num_streams
            stream = self.streams[stream_idx]

            # 使用stream异步上传
            gpu_in = cv2.cuda_GpuMat()
            gpu_in.upload(img, stream=stream)
            gpu_inputs.append(gpu_in)

            # 使用stream异步处理
            gpu_out = cv2.cuda_GpuMat()
            self.filter.apply(gpu_in, gpu_out, stream=stream)
            gpu_outputs.append(gpu_out)

        # 同步所有stream（等待GPU完成）
        for stream in self.streams:
            stream.waitForCompletion()

        # 阶段3: 批量下载+编码
        results = []
        for gpu_out in gpu_outputs:
            result = gpu_out.download()
            _, jpeg = cv2.imencode(
                ".jpg", result,
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            results.append(jpeg.tobytes())

        return results


class OptimizedBatchProcessor(BatchGaussianProcessor):
    """优化版：使用NVJPEG和GPU内存池"""

    def __init__(self, gpu_id: int, sigma: float = 6.0, num_streams: int = 4):
        super().__init__(gpu_id, sigma, num_streams)

        # GPU内存池（避免重复分配）
        self.memory_pool = cv2.cuda_BufferPool(cv2.cuda.Stream_Null())

    def _process_batch_parallel(
        self,
        images_bytes: List[bytes],
        quality: int
    ) -> List[bytes]:
        """零拷贝优化版（需要NVJPEG支持）"""
        try:
            # 尝试使用NVJPEG在GPU上解码
            return self._process_with_nvjpeg(images_bytes, quality)
        except Exception:
            # Fallback到标准实现
            return super()._process_batch_parallel(images_bytes, quality)

    def _process_with_nvjpeg(
        self,
        images_bytes: List[bytes],
        quality: int
    ) -> List[bytes]:
        """使用NVJPEG的零拷贝实现"""
        # 注意：需要安装 pip install nvidia-nvjpeg-cu12
        try:
            from cuda import cudart
            import nvjpeg
        except ImportError:
            raise RuntimeError("NVJPEG not available")

        results = []
        nvjpeg_handle = nvjpeg.NvJpeg()

        for i, img_bytes in enumerate(images_bytes):
            stream_idx = i % len(self.streams)
            stream = self.streams[stream_idx]

            # GPU上解码
            gpu_img = nvjpeg_handle.decode(img_bytes, stream=stream.cudaPtr())

            # GPU模糊
            gpu_blurred = self.filter.apply(gpu_img, stream=stream)

            # GPU上编码
            encoded = nvjpeg_handle.encode(
                gpu_blurred,
                quality=quality,
                stream=stream.cudaPtr()
            )
            results.append(encoded)

        # 同步
        for stream in self.streams:
            stream.waitForCompletion()

        return results
