import re

import psutil
from huggingface_hub import get_safetensors_metadata

from diffuse_worker.slicing import _is_tower_tensor

_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}

_BLOCK_RE = re.compile(r"(?:^|\.)(?:blocks|transformer_blocks|single_transformer_blocks)\.(\d+)\.")
_LAYER_RE = re.compile(r"\.(?:layers|h|blocks|block|decoder\.layers)\.(\d+)\.")


def _normalize_dtype(d: str) -> str:
    return {
        "bfloat16": "BF16", "float16": "F16", "float32": "F32",
        "bf16": "BF16", "f16": "F16", "f32": "F32",
    }.get(d.lower(), "BF16")


def _profile_diffusion(model_id, config, load_dtype, per):
    import glob
    import os

    from safetensors import safe_open

    from diffuse_worker.diffusion import DiffusionStack, block_count

    total = block_count(config) or 0
    files = DiffusionStack._weight_files(model_id, None)
    per_block = {}
    other = 0
    for path in files:
        with safe_open(path, framework="pt") as handle:
            for name in handle.keys():
                shape = handle.get_slice(name).get_shape()
                size = 1
                for dim in shape:
                    size *= dim
                size *= per
                match = _BLOCK_RE.search(name)
                if match:
                    index = int(match.group(1))
                    per_block[index] = per_block.get(index, 0) + size
                else:
                    other += size
    sizes = [per_block.get(i, 0) for i in range(total)]
    average = (sum(sizes) // total) if total else 0
    return {
        "model_id": model_id,
        "load_dtype": load_dtype,
        "total_layers": total,
        "layer_sizes_bytes": sizes,
        "avg_layer_bytes": average,
        "non_layer_bytes": other,
        "total_bytes": sum(sizes) + other,
    }


def profile_model(model_id: str, load_dtype: str = "bfloat16", hf_token: str | None = None) -> dict:
    from diffuse_worker.diffusion import transformer_config

    config = transformer_config(model_id, hf_token)
    if config is not None:
        return _profile_diffusion(
            model_id, config, load_dtype, _DTYPE_BYTES.get(_normalize_dtype(load_dtype), 2)
        )

    meta = get_safetensors_metadata(model_id, token=hf_token)
    per = _DTYPE_BYTES.get(_normalize_dtype(load_dtype), 2)

    per_layer_bytes = {}
    non_layer_bytes = 0

    for file_meta in meta.files_metadata.values():
        for name, tensor in file_meta.tensors.items():
            size = tensor.parameter_count * per
            # An encoder tower numbers its own blocks, so counting them as
            # decoder layers overstates the model: Qwen2-VL would be profiled
            # with 32 layers for a 28 layer decoder, and the assignment would
            # hand out a slice that does not exist. The tower is a fixed cost
            # carried by whoever holds the embeddings.
            m = None if _is_tower_tensor(name) else _LAYER_RE.search(name)
            if m:
                idx = int(m.group(1))
                per_layer_bytes[idx] = per_layer_bytes.get(idx, 0) + size
            else:
                non_layer_bytes += size

    total_layers = (max(per_layer_bytes) + 1) if per_layer_bytes else 0
    layer_sizes = [per_layer_bytes.get(i, 0) for i in range(total_layers)]
    avg_layer_bytes = (sum(layer_sizes) // total_layers) if total_layers else 0

    return {
        "model_id": model_id,
        "load_dtype": load_dtype,
        "total_layers": total_layers,
        "layer_sizes_bytes": layer_sizes,
        "avg_layer_bytes": avg_layer_bytes,
        "non_layer_bytes": non_layer_bytes,
        "total_bytes": sum(layer_sizes) + non_layer_bytes,
    }


def available_memory_bytes() -> int:
    return psutil.virtual_memory().available


def gpu_memory_bytes() -> tuple[int, str] | None:
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return int(free), "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon: unified memory, use system available as a proxy.
            return None
    except Exception:
        pass
    return None


def available_memory() -> tuple[int, str]:
    gpu = gpu_memory_bytes()
    if gpu is not None:
        return gpu
    return available_memory_bytes(), "cpu"


def plan_capacity(
    model_id: str,
    overhead_fraction: float,
    load_dtype: str = "bfloat16",
    hf_token: str | None = None,
) -> dict:
    profile = profile_model(model_id, load_dtype=load_dtype, hf_token=hf_token)

    gpu = gpu_memory_bytes()
    if gpu is not None:
        available, device = gpu
    else:
        available = available_memory_bytes()
        device = "cpu"

    usable = int(available * (1.0 - overhead_fraction))

    layer_sizes = profile["layer_sizes_bytes"]
    embedding_bytes = profile["non_layer_bytes"]

    max_layers_no_embed = 0
    running = 0
    for size in layer_sizes:
        if running + size <= usable:
            running += size
            max_layers_no_embed += 1
        else:
            break

    budget_with_embed = usable - embedding_bytes
    max_layers_with_embed = 0
    running = 0
    for size in layer_sizes:
        if running + size <= budget_with_embed:
            running += size
            max_layers_with_embed += 1
        else:
            break

    return {
        "model_id": model_id,
        "load_dtype": load_dtype,
        "device": device,
        "total_layers": profile["total_layers"],
        "available_bytes": available,
        "usable_bytes": usable,
        "avg_layer_bytes": profile["avg_layer_bytes"],
        "non_layer_bytes": embedding_bytes,
        "max_layers": max(0, max_layers_no_embed),
        "max_layers_if_holding_embedding": max(0, max_layers_with_embed),
    }