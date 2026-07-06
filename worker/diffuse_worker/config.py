import os
from dataclasses import dataclass


@dataclass
class WorkerConfig:
    host: str = "127.0.0.1"
    port: int = 50051
    hf_token: str | None = None
    device: str = "cpu"
    cache_dir: str | None = None

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        return cls(
            host=os.environ.get("DIFFUSE_WORKER_HOST", "127.0.0.1"),
            port=int(os.environ.get("DIFFUSE_WORKER_PORT", "50051")),
            hf_token=os.environ.get("HF_TOKEN") or None,
            device=os.environ.get("DIFFUSE_WORKER_DEVICE", "cpu"),
            cache_dir=os.environ.get("DIFFUSE_WORKER_CACHE") or None,
        )