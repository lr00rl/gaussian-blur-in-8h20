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
            # 小batch用简单方式
            return [self._process_single(img_bytes, quality) for img_bytes in images_bytes]
        else:
            # 大batch用分组并行
            return self._process_batch_chunked(images_bytes, quality)

    def _process_single(self, img_bytes: bytes, quality: int) -> bytes:
        """单张图片处理（同步方式）"""
        try:
            # CPU解码
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")

            # GPU处理
            gpu_in = cv2.cuda_GpuMat()
            gpu_in.upload(img)

            gpu_out = cv2.cuda_GpuMat()
            self.filter.apply(gpu_in, gpu_out)

            result = gpu_out.download()

            # 验证result不为空
            if result is None or result.size == 0:
                raise ValueError("GPU processing returned empty result")

            # CPU编码
            success, jpeg = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not success:
                raise RuntimeError("Failed to encode JPEG")

            return jpeg.tobytes()

        except Exception as e:
            print(f"[GPU {self.gpu_id}] Error processing single image: {e}")
            raise

    def _process_batch_chunked(self, images_bytes: List[bytes], quality: int) -> List[bytes]:
        """
        分组并行处理 - 关键修复

        将batch按stream数量分组，每组内完全同步后再处理下一组
        避免异步操作之间的竞态条件
        """
        num_streams = len(self.streams)
        results = []

        # 按stream数量分chunk
        for chunk_start in range(0, len(images_bytes), num_streams):
            chunk_end = min(chunk_start + num_streams, len(images_bytes))
            chunk = images_bytes[chunk_start:chunk_end]

            # 处理这个chunk
            chunk_results = self._process_chunk_parallel(chunk, quality)
            results.extend(chunk_results)

        return results

    def _process_chunk_parallel(self, chunk: List[bytes], quality: int) -> List[bytes]:
        """
        并行处理一个chunk（大小 <= num_streams）

        关键改进：
        1. 每个stream处理一张图，确保一对一映射
        2. 显式同步每个stream后再download
        3. 添加详细的错误检查
        """
        chunk_size = len(chunk)

        # 阶段1: 批量解码
        cpu_images = []
        for idx, img_bytes in enumerate(chunk):
            try:
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to decode image {idx}")
                cpu_images.append(img)
            except Exception as e:
                print(f"[GPU {self.gpu_id}] Decode error at {idx}: {e}")
                raise

        # 阶段2: GPU异步上传和处理
        # 为每张图创建独立的GpuMat，并与stream一一对应
        operations = []  # 存储 (stream, gpu_in, gpu_out)

        for idx, img in enumerate(cpu_images):
            stream = self.streams[idx]  # 一对一映射

            try:
                # 创建GpuMat
                gpu_in = cv2.cuda_GpuMat()
                gpu_out = cv2.cuda_GpuMat()

                # 异步上传
                gpu_in.upload(img, stream=stream)

                # 异步模糊
                self.filter.apply(gpu_in, gpu_out, stream=stream)

                operations.append((stream, gpu_in, gpu_out))

            except Exception as e:
                print(f"[GPU {self.gpu_id}] GPU operation error at {idx}: {e}")
                raise

        # 阶段3: 同步每个stream并下载
        results = []
        for idx, (stream, gpu_in, gpu_out) in enumerate(operations):
            try:
                # 显式等待该stream完成
                stream.waitForCompletion()

                # 下载结果
                result = gpu_out.download()

                # 验证结果有效性
                if result is None or result.size == 0:
                    raise ValueError(f"Empty result at index {idx}")

                # 编码
                success, jpeg = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not success:
                    raise RuntimeError(f"Encode failed at index {idx}")

                results.append(jpeg.tobytes())

            except Exception as e:
                print(f"[GPU {self.gpu_id}] Download/encode error at {idx}: {e}")
                raise

        return results
