from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # GPU配置（保持字段名兼容旧代码；实际是 worker 进程数）
    num_gpus: int = 24          # ← 8 → 16，匹配 jumper 16 物理核
    gpu_ids: list[int] = list(range(24))   # ← 同上

    # 处理参数
    max_batch_size: int = 32
    gaussian_sigma: float = 18.0
    gaussian_ksize: int = 21  # 必须是奇数，且 <= 31
    jpeg_quality: int = 75

    # 性能调优
    queue_size_multiplier: int = 16   # ← 4 → 16，task_queue 容量从 32 提到 256，吸收 client 突发
    enable_cuda_stream: bool = True
    num_streams_per_gpu: int = 8      # batch 内并行线程数 = num_streams // 2 = 2

    # 服务配置
    host: str = "0.0.0.0"
    port: int = 9000

    class Config:
        env_file = ".env"

settings = Settings()
