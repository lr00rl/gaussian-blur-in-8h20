import asyncio
import aiohttp
import time
from pathlib import Path
import zipfile
import io

async def test_batch_blur():
    """测试批量模糊"""
    # 准备32张测试图片
    test_dir = Path("test_avatars")
    image_files = list(test_dir.glob("*.jpg"))[:32]

    if len(image_files) < 32:
        print(f"Warning: Only found {len(image_files)} images")

    # 构建multipart请求
    data = aiohttp.FormData()
    for img_file in image_files:
        data.add_field(
            'files',
            open(img_file, 'rb'),
            filename=img_file.name,
            content_type='image/jpeg'
        )
    data.add_field('quality', '75')

    # 发送请求
    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8000/blur/batch',
            data=data
        ) as response:
            if response.status == 200:
                zip_content = await response.read()
                elapsed = (time.perf_counter() - start_time) * 1000

                # 解压查看结果
                with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                    print(f"✓ Processed {len(zf.namelist())} images in {elapsed:.2f}ms")
                    print(f"  Throughput: {len(image_files) / (elapsed/1000):.1f} images/sec")

                    # 保存结果
                    zf.extractall("output_batch")
            else:
                print(f"✗ Error: {response.status}")
                print(await response.text())


async def benchmark_concurrent_batches():
    """压力测试：并发多个batch"""
    async def send_batch(session, batch_id):
        # 每个batch 16张图
        data = aiohttp.FormData()
        for i in range(16):
            img_path = Path(f"test_avatars/avatar_{i%10}.jpg")
            data.add_field(
                'files',
                open(img_path, 'rb'),
                filename=f"img_{i}.jpg",
                content_type='image/jpeg'
            )
        data.add_field('quality', '75')

        start = time.perf_counter()
        async with session.post('http://localhost:8000/blur/batch', data=data) as resp:
            await resp.read()
            elapsed = time.perf_counter() - start
            return batch_id, elapsed

    # 并发发送10个batch
    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        tasks = [send_batch(session, i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        print("\n=== Concurrent Batch Test ===")
        for batch_id, elapsed in results:
            print(f"Batch {batch_id}: {elapsed*1000:.2f}ms")
        print(f"Total time: {total_time:.2f}s")
        print(f"Total images: {10 * 16} = 160")
        print(f"Throughput: {160/total_time:.1f} images/sec")


if __name__ == "__main__":
    # 单batch测试
    asyncio.run(test_batch_blur())

    # 并发压测
    asyncio.run(benchmark_concurrent_batches())
