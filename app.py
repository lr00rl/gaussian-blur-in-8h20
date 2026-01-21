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
    # Startup
    pool = get_gpu_pool()
    yield
    # Shutdown
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
    curl -X POST http://localhost:8000/blur/batch \
      -F "files=@avatar1.jpg" \
      -F "files=@avatar2.jpg" \
      -F "quality=75"

    返回: ZIP文件包含所有处理后的图片
    """
    # 验证
    if not files:
        raise HTTPException(400, "No files provided")

    if len(files) > settings.max_batch_size:
        raise HTTPException(
            400,
            f"Batch size {len(files)} exceeds limit {settings.max_batch_size}"
        )

    # 读取所有文件
    images_bytes = []
    filenames = []

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(400, f"Invalid file type: {file.filename}")

        content = await file.read()
        images_bytes.append(content)
        filenames.append(file.filename)

    # 提交到GPU池
    pool = get_gpu_pool()
    results = await pool.submit_batch(images_bytes, quality)

    # 打包成ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, (result, filename) in enumerate(zip.enumerate(results, filenames)):
            output_name = f"blurred_{i:03d}_{filename}"
            zf.writestr(output_name, result)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=blurred_batch.zip"}
    )


@app.post("/blur/batch/json")
async def blur_batch_json(
    files: List[UploadFile] = File(...),
    quality: int = Form(75)
):
    """
    批量处理接口（JSON返回base64）

    适用场景：前端需要立即预览所有图片
    """
    import base64

    if not files or len(files) > settings.max_batch_size:
        raise HTTPException(400, "Invalid batch size")

    # 读取文件
    images_bytes = [await f.read() for f in files]

    # 处理
    pool = get_gpu_pool()
    results = await pool.submit_batch(images_bytes, quality)

    # 返回base64编码
    return JSONResponse({
        "count": len(results),
        "images": [
            {
                "index": i,
                "filename": files[i].filename,
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
    return {"status": "healthy", "num_gpus": settings.num_gpus}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        loop="uvloop",
        log_level="info"
    )
