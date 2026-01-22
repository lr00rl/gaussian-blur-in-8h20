import cv2
import numpy as np
from typing import List

class BatchGaussianProcessor:
    """单GPU的Batch处理器，支持CUDA Stream并行 - 使用预分配buffer"""

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
        self.num_streams = num_streams
        self.streams = [cv2.cuda_Stream() for _ in range(num_streams)]
        print(f"[GPU {gpu_id}] Created {num_streams} CUDA streams")

        # ✅ 新增：为每个stream预分配GpuMat buffer
        self.gpu_buffers = []
        for i in range(num_streams):
            gpu_in = cv2.cuda_GpuMat()
            gpu_out = cv2.cuda_GpuMat()
            self.gpu_buffers.append((gpu_in, gpu_out))
        print(f"[GPU {gpu_id}] Pre-allocated {num_streams} GPU buffer pairs")

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
        print(f"[GPU {self.gpu_id}] Processing batch: {batch_size} images, quality={quality}")

        if batch_size <= 3:
            # 小batch用简单方式
            print(f"[GPU {self.gpu_id}] Using simple mode (batch_size <= 3)")
            return [self._process_single(img_bytes, quality, idx) for idx, img_bytes in enumerate(images_bytes)]
        else:
            # 大batch用分组并行
            print(f"[GPU {self.gpu_id}] Using chunked parallel mode (batch_size > 3)")
            return self._process_batch_chunked(images_bytes, quality)

    def _process_single(self, img_bytes: bytes, quality: int, idx: int = 0) -> bytes:
        """单张图片处理（同步方式）"""
        try:
            print(f"[GPU {self.gpu_id}][Img {idx}] Decoding image, size={len(img_bytes)} bytes")

            # CPU解码
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image {idx}")

            print(f"[GPU {self.gpu_id}][Img {idx}] Decoded shape: {img.shape}, dtype: {img.dtype}")

            # GPU处理
            gpu_in = cv2.cuda_GpuMat()
            gpu_in.upload(img)
            print(f"[GPU {self.gpu_id}][Img {idx}] Uploaded to GPU")

            gpu_out = cv2.cuda_GpuMat()
            self.filter.apply(gpu_in, gpu_out)
            print(f"[GPU {self.gpu_id}][Img {idx}] Gaussian filter applied")

            result = gpu_out.download()
            print(f"[GPU {self.gpu_id}][Img {idx}] Downloaded from GPU, shape: {result.shape if result is not None else 'None'}")

            # 验证result不为空
            if result is None or result.size == 0:
                raise ValueError(f"GPU processing returned empty result for image {idx}")

            # CPU编码
            success, jpeg = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not success:
                raise RuntimeError(f"Failed to encode JPEG for image {idx}")

            jpeg_bytes = jpeg.tobytes()
            print(f"[GPU {self.gpu_id}][Img {idx}] Encoded JPEG, size={len(jpeg_bytes)} bytes")

            return jpeg_bytes

        except Exception as e:
            print(f"[GPU {self.gpu_id}][Img {idx}] ❌ Error processing single image: {e}")
            raise

    def _process_batch_chunked(self, images_bytes: List[bytes], quality: int) -> List[bytes]:
        """
        分组并行处理

        将batch按stream数量分组，每组内完全同步后再处理下一组
        """
        results = []
        total_images = len(images_bytes)

        print(f"[GPU {self.gpu_id}] Chunking {total_images} images by {self.num_streams} streams")

        # 按stream数量分chunk
        for chunk_start in range(0, total_images, self.num_streams):
            chunk_end = min(chunk_start + self.num_streams, total_images)
            chunk = images_bytes[chunk_start:chunk_end]

            print(f"[GPU {self.gpu_id}] Processing chunk [{chunk_start}:{chunk_end}], size={len(chunk)}")

            # 处理这个chunk
            chunk_results = self._process_chunk_parallel(chunk, quality, chunk_start)
            results.extend(chunk_results)

            print(f"[GPU {self.gpu_id}] Chunk [{chunk_start}:{chunk_end}] completed, got {len(chunk_results)} results")

        print(f"[GPU {self.gpu_id}] ✅ All chunks completed, total results: {len(results)}")
        return results

    def _process_chunk_parallel(self, chunk: List[bytes], quality: int, chunk_offset: int = 0) -> List[bytes]:
        """
        并行处理一个chunk（大小 <= num_streams）

        ✅ 关键改进：使用预分配的GPU buffer，避免GpuMat生命周期问题
        """
        chunk_size = len(chunk)
        print(f"[GPU {self.gpu_id}] === Starting parallel chunk processing, size={chunk_size} ===")

        # ========================================
        # 阶段1: 批量解码（CPU）
        # ========================================
        print(f"[GPU {self.gpu_id}] Stage 1: Batch decoding on CPU")
        cpu_images = []
        for idx, img_bytes in enumerate(chunk):
            global_idx = chunk_offset + idx
            try:
                print(f"[GPU {self.gpu_id}][Img {global_idx}] Decoding, input size={len(img_bytes)} bytes")

                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to decode image {global_idx}")

                print(f"[GPU {self.gpu_id}][Img {global_idx}] Decoded OK, shape={img.shape}, dtype={img.dtype}")
                cpu_images.append(img)

            except Exception as e:
                print(f"[GPU {self.gpu_id}][Img {global_idx}] ❌ Decode error: {e}")
                raise

        print(f"[GPU {self.gpu_id}] Stage 1 complete: {len(cpu_images)} images decoded")

        # ========================================
        # 阶段2: GPU异步上传和处理
        # ========================================
        print(f"[GPU {self.gpu_id}] Stage 2: Async GPU upload & processing")
        operations = []  # 存储 (stream_idx, global_img_idx)

        for idx, img in enumerate(cpu_images):
            global_idx = chunk_offset + idx
            stream_idx = idx  # stream索引与chunk内索引一致
            stream = self.streams[stream_idx]

            # ✅ 使用预分配的buffer
            gpu_in, gpu_out = self.gpu_buffers[stream_idx]

            try:
                print(f"[GPU {self.gpu_id}][Img {global_idx}] Using stream {stream_idx}, buffer pair {stream_idx}")
                print(f"[GPU {self.gpu_id}][Img {global_idx}] Before upload: img.shape={img.shape}")

                # 异步上传
                gpu_in.upload(img, stream=stream)
                print(f"[GPU {self.gpu_id}][Img {global_idx}] Upload queued to stream {stream_idx}")

                # 异步模糊
                self.filter.apply(gpu_in, gpu_out, stream=stream)
                print(f"[GPU {self.gpu_id}][Img {global_idx}] Filter apply queued to stream {stream_idx}")

                # ✅ 只存储索引，不存储GpuMat引用
                operations.append((stream_idx, global_idx))

            except Exception as e:
                print(f"[GPU {self.gpu_id}][Img {global_idx}] ❌ GPU operation error: {e}")
                raise

        print(f"[GPU {self.gpu_id}] Stage 2 complete: {len(operations)} async operations queued")

        # ========================================
        # 阶段3: 同步每个stream并下载
        # ========================================
        print(f"[GPU {self.gpu_id}] Stage 3: Sync streams & download results")
        results = []

        for stream_idx, global_idx in operations:
            try:
                stream = self.streams[stream_idx]
                gpu_in, gpu_out = self.gpu_buffers[stream_idx]

                print(f"[GPU {self.gpu_id}][Img {global_idx}] Waiting for stream {stream_idx} completion...")

                # 显式等待该stream完成
                stream.waitForCompletion()
                print(f"[GPU {self.gpu_id}][Img {global_idx}] Stream {stream_idx} completed")

                # 下载结果
                print(f"[GPU {self.gpu_id}][Img {global_idx}] Downloading from GPU buffer {stream_idx}...")
                result = gpu_out.download()

                # 详细验证
                if result is None:
                    raise ValueError(f"Download returned None at index {global_idx}")
                if result.size == 0:
                    raise ValueError(f"Download returned empty array (size=0) at index {global_idx}")

                print(f"[GPU {self.gpu_id}][Img {global_idx}] Downloaded OK, shape={result.shape}, dtype={result.dtype}, size={result.size}")

                # 编码
                print(f"[GPU {self.gpu_id}][Img {global_idx}] Encoding to JPEG...")
                success, jpeg = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not success:
                    raise RuntimeError(f"Encode failed at index {global_idx}")

                jpeg_bytes = jpeg.tobytes()
                print(f"[GPU {self.gpu_id}][Img {global_idx}] Encoded OK, JPEG size={len(jpeg_bytes)} bytes")

                results.append(jpeg_bytes)

            except Exception as e:
                print(f"[GPU {self.gpu_id}][Img {global_idx}] ❌ Download/encode error: {e}")
                raise

        print(f"[GPU {self.gpu_id}] Stage 3 complete: {len(results)} results ready")
        print(f"[GPU {self.gpu_id}] === Parallel chunk processing finished ===\n")

        return results
