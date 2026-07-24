import os
from dataclasses import dataclass


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class WorkerConfig:
    host: str = "127.0.0.1"
    port: int = 50051
    hf_token: str | None = None
    device: str = "cpu"
    cache_dir: str | None = None
    max_concurrency: int = 16
    torch_threads: int = 0
    max_batch: int = 8

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        return cls(
            host=os.environ.get("DIFFUSE_WORKER_HOST", "127.0.0.1"),
            port=int(os.environ.get("DIFFUSE_WORKER_PORT", "50051")),
            hf_token=os.environ.get("HF_TOKEN") or None,
            device=os.environ.get("DIFFUSE_WORKER_DEVICE") or _default_device(),
            cache_dir=os.environ.get("DIFFUSE_WORKER_CACHE") or None,
            max_concurrency=int(os.environ.get("DIFFUSE_WORKER_CONCURRENCY", "16")),
            torch_threads=int(os.environ.get("DIFFUSE_WORKER_THREADS", "0")),
            max_batch=int(os.environ.get("DIFFUSE_WORKER_MAX_BATCH", "8")),
        )