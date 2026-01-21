from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # GPU配置
    num_gpus: int = 8
    gpu_ids: list[int] = list(range(8))
    
    # 处理参数
    max_batch_size: int = 32
    gaussian_sigma: float = 4.0
    gaussian_ksize: int = 21  # 必须是奇数，且 <= 31
    jpeg_quality: int = 75
    
    # 性能调优
    queue_size_multiplier: int = 4
    enable_cuda_stream: bool = True
    num_streams_per_gpu: int = 4
    
    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()
