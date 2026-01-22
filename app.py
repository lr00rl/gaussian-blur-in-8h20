from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List
import io
import zipfile
from contextlib import asynccontextmanager
from gpu_pool import get_gpu_pool
from config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理"""
    pool = get_gpu_pool()
    yield
    pool.shutdown()

app = FastAPI(
    title="Batch Gaussian Blur Service",
    description="Multi-GPU batch processing for avatar blur",
    lifespan=lifespan
)


@app.post("/blur/batch")
async def blur_batch(
    files: List[UploadFile] = File(...),
    quality: int = Form(75)
):
    """
    批量处理接口

    请求示例:
    curl -X POST http://localhost:9000/blur/batch \
      -F "files=@avatar1.jpg" \
      -F "files=@avatar2.jpg" \
      -F "quality=75" \
      -o result.zip
    """
    # 验证
    if not files:
        raise HTTPException(400, "No files provided")

    if len(files) > settings.max_batch_size:
        raise HTTPException(
            400,
            f"Batch size {len(files)} exceeds limit {settings.max_batch_size}"
        )

    # 读取所有文件（关键修复：确保完整读取）
    images_bytes = []
    filenames = []

    for idx, file in enumerate(files):
        # 验证content type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                400,
                f"File {idx} ({file.filename}): Invalid content type '{file.content_type}'"
            )

        # 完整读取文件内容
        try:
            content = await file.read()

            # 验证数据非空
            if not content or len(content) == 0:
                raise HTTPException(
                    400,
                    f"File {idx} ({file.filename}): Empty file"
                )

            print(f"[API] File {idx}: name={file.filename}, size={len(content)}, "
                  f"header={content[:4].hex() if len(content) >= 4 else 'TOO_SHORT'}")

            # 验证是有效的图片格式（简单检查magic number）
            if not _is_valid_image(content):
                raise HTTPException(
                    400,
                    f"File {idx} ({file.filename}): Invalid image format"
                )

            images_bytes.append(content)
            filenames.append(file.filename or f"image_{idx}.jpg")

        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(500, f"Failed to read file {idx}: {str(e)}")
        finally:
            await file.close()

    print(f"[API] Received {len(images_bytes)} images, sizes: {[len(b) for b in images_bytes]}")

    # 提交到GPU池处理
    try:
        pool = get_gpu_pool()
        results = await pool.submit_batch(images_bytes, quality)
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

    # 打包成ZIP
    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, (result, filename) in enumerate(zip(results, filenames)):
                # 生成输出文件名
                name_parts = filename.rsplit('.', 1)
                if len(name_parts) == 2:
                    output_name = f"blurred_{i:03d}_{name_parts[0]}.{name_parts[1]}"
                else:
                    output_name = f"blurred_{i:03d}_{filename}.jpg"

                zf.writestr(output_name, result)

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=blurred_batch_{len(results)}_images.zip"
            }
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to create ZIP: {str(e)}")


def _is_valid_image(data: bytes) -> bool:
    """简单验证图片格式（通过magic number）"""
    if len(data) < 12:
        return False

    # JPEG: FF D8 FF
    if data[:3] == b'\xff\xd8\xff':
        return True

    # PNG: 89 50 4E 47
    if data[:4] == b'\x89PNG':
        return True

    # WebP: RIFF ... WEBP
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return True

    return False


@app.post("/blur/batch/json")
async def blur_batch_json(
    files: List[UploadFile] = File(...),
    quality: int = Form(75)
):
    """
    批量处理接口（JSON返回base64）
    """
    import base64

    if not files or len(files) > settings.max_batch_size:
        raise HTTPException(400, "Invalid batch size")

    # 读取文件
    images_bytes = []
    filenames = []

    for idx, file in enumerate(files):
        try:
            content = await file.read()
            if not content:
                raise HTTPException(400, f"File {idx}: Empty")

            images_bytes.append(content)
            filenames.append(file.filename or f"image_{idx}.jpg")
        finally:
            await file.close()

    # 处理
    try:
        pool = get_gpu_pool()
        results = await pool.submit_batch(images_bytes, quality)
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

    # 返回base64
    return JSONResponse({
        "count": len(results),
        "images": [
            {
                "index": i,
                "filename": filenames[i],
                "size": len(result),
                "data": base64.b64encode(result).decode()
            }
            for i, result in enumerate(results)
        ]
    })


@app.get("/stats")
async def get_stats():
    """获取服务统计"""
    pool = get_gpu_pool()
    return pool.get_stats()


@app.get("/health")
async def health():
    """健康检查"""
    pool = get_gpu_pool()
    stats = pool.get_stats()

    return {
        "status": "healthy" if stats["alive_workers"] > 0 else "degraded",
        "num_gpus": settings.num_gpus,
        "alive_workers": stats["alive_workers"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        loop="uvloop",
        log_level="info"
    )
