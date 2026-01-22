import cv2
import numpy as np

print("=" * 60)
print("OpenCV CUDA 环境诊断")
print("=" * 60)

# 1. 检查CUDA设备
print(f"\n1. CUDA设备数量: {cv2.cuda.getCudaEnabledDeviceCount()}")

if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    print("❌ 没有检测到CUDA设备！")
    exit(1)

# 2. 设置GPU 0并测试
cv2.cuda.setDevice(0)
print(f"2. 当前GPU设备: {cv2.cuda.getDevice()}")

# 3. 测试基础GpuMat操作
print("\n3. 测试GpuMat基础操作...")
try:
    cpu_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    print(f"   创建CPU图像: {cpu_img.shape}")
    
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(cpu_img)
    print(f"   上传到GPU: {gpu_mat.size()}, empty={gpu_mat.empty()}")
    
    cpu_result = gpu_mat.download()
    print(f"   下载到CPU: {cpu_result.shape if cpu_result is not None else 'None'}")
    
    if cpu_result is None:
        print("   ❌ GpuMat.download() 返回 None")
    else:
        print("   ✅ GpuMat基础操作正常")
except Exception as e:
    print(f"   ❌ GpuMat操作失败: {e}")
    exit(1)

# 4. 测试Gaussian Filter
print("\n4. 测试Gaussian Filter...")
try:
    cpu_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    # 创建filter
    gaussian_filter = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC3,
        cv2.CV_8UC3,
        ksize=(21, 21),
        sigma1=4.0,
        sigma2=4.0
    )
    print(f"   Filter创建成功")
    
    # 上传
    gpu_in = cv2.cuda_GpuMat()
    gpu_in.upload(cpu_img)
    print(f"   输入GpuMat: size={gpu_in.size()}, empty={gpu_in.empty()}")
    
    # 应用filter
    gpu_out = cv2.cuda_GpuMat()
    gaussian_filter.apply(gpu_in, gpu_out)
    print(f"   输出GpuMat: size={gpu_out.size()}, empty={gpu_out.empty()}")
    
    # 下载
    result = gpu_out.download()
    print(f"   下载结果: {result.shape if result is not None else 'None'}")
    
    if result is None or result.size == 0:
        print("   ❌ Filter输出为空！")
        print("   可能原因：OpenCV CUDA编译有问题")
    else:
        print(f"   ✅ Gaussian Filter正常工作！")
        
        # 验证结果
        if np.array_equal(result, cpu_img):
            print("   ⚠️  警告：输出和输入完全相同，filter可能没生效")
        else:
            print("   ✅ Filter确实修改了图像")
            
except Exception as e:
    print(f"   ❌ Filter测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. 测试完整流程
print("\n5. 测试完整的编码-解码-模糊-编码流程...")
try:
    # 创建测试图像并编码
    test_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    _, jpeg_bytes = cv2.imencode('.jpg', test_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    print(f"   原始JPEG: {len(jpeg_bytes)} bytes")
    
    # 解码
    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    print(f"   解码后: {decoded.shape}")
    
    # GPU模糊
    gpu_in = cv2.cuda_GpuMat()
    gpu_in.upload(decoded)
    
    gpu_out = cv2.cuda_GpuMat()
    gaussian_filter.apply(gpu_in, gpu_out)
    
    blurred = gpu_out.download()
    
    if blurred is None or blurred.size == 0:
        print("   ❌ 模糊结果为空")
        exit(1)
    
    print(f"   模糊后: {blurred.shape}")
    
    # 重新编码
    success, final_jpeg = cv2.imencode('.jpg', blurred, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not success:
        print("   ❌ 编码失败")
        exit(1)
    
    print(f"   最终JPEG: {len(final_jpeg.tobytes())} bytes")
    print("   ✅ 完整流程测试成功！")
    
except Exception as e:
    print(f"   ❌ 完整流程失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！GPU环境正常")
print("=" * 60)
