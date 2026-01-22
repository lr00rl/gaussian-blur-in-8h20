import cv2
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter

print("=" * 60)
print("CuPy GPU高斯模糊测试")
print("=" * 60)

# 1. 检查GPU
print(f"可用GPU数量: {cp.cuda.runtime.getDeviceCount()}")

# 2. 设置GPU 0
cp.cuda.Device(0).use()
print(f"当前GPU: {cp.cuda.Device()}")

# 3. 测试高斯模糊
print("\n测试高斯模糊...")

# 创建测试图像
cpu_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
print(f"CPU图像: {cpu_img.shape}, {cpu_img.dtype}")

# 转到GPU
gpu_img = cp.asarray(cpu_img)
print(f"GPU图像: {gpu_img.shape}, {gpu_img.dtype}")

# 对每个通道应用高斯模糊
blurred_channels = []
for c in range(3):
    channel = gpu_img[:, :, c].astype(cp.float32)
    blurred = gaussian_filter(channel, sigma=4.0, mode='reflect')
    blurred_channels.append(blurred)

gpu_blurred = cp.stack(blurred_channels, axis=2)
gpu_blurred = cp.clip(gpu_blurred, 0, 255).astype(cp.uint8)

print(f"模糊后GPU: {gpu_blurred.shape}, {gpu_blurred.dtype}")

# 转回CPU
cpu_blurred = cp.asnumpy(gpu_blurred)
print(f"模糊后CPU: {cpu_blurred.shape}, {cpu_blurred.dtype}")

# 验证结果不同
if not np.array_equal(cpu_img, cpu_blurred):
    print("✅ 模糊确实修改了图像")
else:
    print("❌ 模糊没有生效")

# 4. 完整流程测试
print("\n完整流程测试...")

# 编码
success, jpeg_bytes = cv2.imencode('.jpg', cpu_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
print(f"原始JPEG: {len(jpeg_bytes)} bytes")

# 解码
decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)

# GPU模糊
gpu_decoded = cp.asarray(decoded)
blurred_channels = []
for c in range(3):
    channel = gpu_decoded[:, :, c].astype(cp.float32)
    blurred = gaussian_filter(channel, sigma=4.0, mode='reflect')
    blurred_channels.append(blurred)

gpu_result = cp.stack(blurred_channels, axis=2)
gpu_result = cp.clip(gpu_result, 0, 255).astype(cp.uint8)
cpu_result = cp.asnumpy(gpu_result)

# 编码
success, final_jpeg = cv2.imencode('.jpg', cpu_result, [cv2.IMWRITE_JPEG_QUALITY, 75])
print(f"最终JPEG: {len(final_jpeg.tobytes())} bytes")

if success:
    print("\n✅ 所有测试通过！CuPy方案可用")
else:
    print("\n❌ 编码失败")
