import cv2
import numpy as np
import cupy as cp
from typing import List
from cupyx.scipy.ndimage import gaussian_filter
import time

class BatchGaussianProcessor:
    def __init__(self, gpu_id: int, sigma: float = 4.0):
        self.gpu_id = gpu_id
        self.sigma = sigma

        with cp.cuda.Device(gpu_id):
            # 预编译kernel,避免首次调用开销
            dummy = cp.zeros((2, 100, 100, 3), dtype=cp.float32)
            _ = self._batch_gaussian_filter(dummy)

    def _batch_gaussian_filter(self, batch_tensor: cp.ndarray) -> cp.ndarray:
        """
        批量高斯模糊 - 关键优化点

        输入: (N, H, W, 3) - N张图片
        输出: (N, H, W, 3)
        """
        N = batch_tensor.shape[0]

        # 将通道移到第一维: (N, H, W, 3) -> (N, 3, H, W)
        # 这样可以对所有图片的同一通道批量处理
        batch_tensor = cp.transpose(batch_tensor, (0, 3, 1, 2))

        # 批量处理: 对每个通道并行应用filter
        results = []
        for c in range(3):
            channel_batch = batch_tensor[:, c, :, :]  # (N, H, W)

            # 对N张图片的同一通道同时模糊
            blurred_batch = cp.stack([
                gaussian_filter(channel_batch[i], sigma=self.sigma, mode='reflect')
                for i in range(N)
            ])
            results.append(blurred_batch)

        # 合并通道: (3, N, H, W) -> (N, 3, H, W) -> (N, H, W, 3)
        output = cp.stack(results, axis=1)  # (N, 3, H, W)
        output = cp.transpose(output, (0, 2, 3, 1))  # (N, H, W, 3)

        return output

    def process_batch(self, images_bytes: List[bytes], quality: int = 75) -> List[bytes]:
        """优化后的批处理"""
        with cp.cuda.Device(self.gpu_id):
            start = time.perf_counter()

            # === 阶段1: CPU并行解码 ===
            # 使用ThreadPoolExecutor并行解码JPEG
            from concurrent.futures import ThreadPoolExecutor

            def decode_jpeg(img_bytes):
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Decode failed")
                return img

            with ThreadPoolExecutor(max_workers=4) as executor:
                cpu_images = list(executor.map(decode_jpeg, images_bytes))

            decode_time = (time.perf_counter() - start) * 1000

            # === 阶段2: 统一尺寸 (padding到最大尺寸) ===
            # 找到最大尺寸
            max_h = max(img.shape[0] for img in cpu_images)
            max_w = max(img.shape[1] for img in cpu_images)

            # Padding到统一尺寸 (batch要求shape一致)
            padded_images = []
            padding_info = []

            for img in cpu_images:
                h, w = img.shape[:2]
                padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                padded[:h, :w] = img
                padded_images.append(padded)
                padding_info.append((h, w))

            # === 阶段3: 批量GPU传输 ===
            gpu_batch = cp.asarray(np.stack(padded_images))  # (N, H, W, 3)
            upload_time = (time.perf_counter() - start - decode_time/1000) * 1000

            # === 阶段4: 批量GPU计算 ===
            gpu_batch_float = gpu_batch.astype(cp.float32)
            gpu_blurred = self._batch_gaussian_filter(gpu_batch_float)
            gpu_blurred = cp.clip(gpu_blurred, 0, 255).astype(cp.uint8)

            compute_time = (time.perf_counter() - start - decode_time/1000 - upload_time/1000) * 1000

            # === 阶段5: 批量GPU传输回CPU ===
            cpu_batch = cp.asnumpy(gpu_blurred)
            download_time = (time.perf_counter() - start - decode_time/1000 - upload_time/1000 - compute_time/1000) * 1000

            # === 阶段6: CPU并行编码 ===
            def encode_jpeg(args):
                idx, img = args
                h, w = padding_info[idx]
                cropped = img[:h, :w]  # 去除padding
                success, jpeg = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not success:
                    raise RuntimeError("Encode failed")
                return jpeg.tobytes()

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(encode_jpeg, enumerate(cpu_batch)))

            total_time = (time.perf_counter() - start) * 1000
            encode_time = total_time - decode_time - upload_time - compute_time - download_time

            print(f"[GPU {self.gpu_id}] Batch {len(images_bytes)} imgs: "
                  f"decode={decode_time:.1f}ms, upload={upload_time:.1f}ms, "
                  f"compute={compute_time:.1f}ms, download={download_time:.1f}ms, "
                  f"encode={encode_time:.1f}ms, total={total_time:.1f}ms "
                  f"({len(images_bytes)/(total_time/1000):.1f} img/s)")

            return results
