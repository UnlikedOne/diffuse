import json
import queue
import threading

import torch

_BLOCK_ATTRIBUTES = ("blocks", "transformer_blocks", "single_transformer_blocks", "layers")


class _Captured(Exception):
    pass


def block_holder(model):
    for name in _BLOCK_ATTRIBUTES:
        blocks = getattr(model, name, None)
        if isinstance(blocks, torch.nn.ModuleList) and len(blocks) > 0:
            return name, blocks
    raise ValueError("this model exposes no list of transformer blocks")


def _fetch(model_id: str, name: str, token: str | None):
    import os

    if os.path.isdir(model_id):
        path = os.path.join(model_id, name)
        return path if os.path.exists(path) else None
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(model_id, name, token=token)
    except Exception:
        return None


def read_index(model_id: str, token: str | None = None) -> dict | None:
    path = _fetch(model_id, "model_index.json", token)
    if path is None:
        return None
    with open(path) as handle:
        index = json.load(handle)
    return index if "transformer" in index else None


def transformer_config(model_id: str, token: str | None = None) -> dict | None:
    path = _fetch(model_id, "transformer/config.json", token)
    if path is None:
        return None
    with open(path) as handle:
        return json.load(handle)


def block_count(config: dict) -> int | None:
    for key in ("num_layers", "num_hidden_layers", "num_blocks", "depth"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


class PatchAttention:
    def __init__(self, attention):
        self.attention = attention
        self.buffers: dict[str, torch.Tensor] = {}
        self.session = None
        self.offset = 0
        self.sequence = 0

    @staticmethod
    def _rope(x, cos, sin):
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        c = cos[..., 0::2]
        s = sin[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * c - x2 * s
        out[..., 1::2] = x1 * s + x2 * c
        return out.type_as(x)

    def context_for(self, hidden_states):
        buffer = self.buffers.get(self.session)
        if (
            buffer is None
            or buffer.shape[1] != self.sequence
            or buffer.shape[2] != hidden_states.shape[2]
        ):
            buffer = torch.zeros(
                hidden_states.shape[0],
                self.sequence,
                hidden_states.shape[2],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            self.buffers[self.session] = buffer
        buffer[:, self.offset : self.offset + hidden_states.shape[1]] = hidden_states
        return buffer

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None, **kwargs):
        if self.session is None or encoder_hidden_states is not None:
            return self.attention(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, **kwargs)

        context = self.context_for(hidden_states)
        query = attn.to_q(hidden_states)
        key = attn.to_k(context)
        value = attn.to_v(context)
        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            cos, sin = rotary_emb
            stop = self.offset + hidden_states.shape[1]
            query = self._rope(query, cos[:, self.offset : stop], sin[:, self.offset : stop])
            key = self._rope(key, cos, sin)

        out = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        out = out.transpose(1, 2).flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        return attn.to_out[1](out)


class DiffusionStack:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_id = None
        self.start_block = 0
        self.end_block = 0
        self.total_blocks = 0
        self.model = None
        self.blocks = None
        self.patchers: list[PatchAttention] = []
        self.contexts: dict[str, list[torch.Tensor]] = {}
        self._lock = threading.Lock()

    def load(self, model_id: str, start: int, end: int, token: str | None = None):
        config = transformer_config(model_id, token)
        if config is None:
            raise ValueError(f"{model_id} has no diffusion transformer")
        total = block_count(config)
        if total is None:
            raise ValueError(f"cannot determine the block count of {model_id}")
        if end > total:
            raise ValueError(f"end block {end} exceeds total {total}")

        import diffusers

        klass = getattr(diffusers, config["_class_name"])
        wanted = dict(config)
        wanted.pop("_class_name", None)
        wanted.pop("_diffusers_version", None)
        holder_key = None
        for key in ("num_layers", "num_hidden_layers", "num_blocks", "depth"):
            if key in wanted:
                holder_key = key
                break
        kept = end - start if end > start else 0
        wanted[holder_key] = max(kept, 1)

        model = klass(**wanted)
        name, blocks = block_holder(model)
        if kept == 0:
            setattr(model, name, torch.nn.ModuleList())

        self._materialise(model, model_id, start, end, total, name, token)

        self.model_id = model_id
        self.start_block = start
        self.end_block = end
        self.total_blocks = total
        self.model = model
        self.blocks = getattr(model, name)
        self._install_patchers()
        return total

    def _materialise(self, model, model_id, start, end, total, block_attribute, token):
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        prefix = f"{block_attribute}."
        wanted = {}
        for key in model.state_dict():
            if not key.startswith(prefix):
                wanted[key] = key
                continue
            index = int(key[len(prefix) :].split(".")[0])
            wanted[key] = f"{prefix}{start + index}." + key[len(prefix) :].split(".", 1)[1]

        files = self._weight_files(model_id, token)
        tensors = {}
        for path in files:
            with safe_open(path, framework="pt") as handle:
                available = set(handle.keys())
                for local, remote in wanted.items():
                    if remote in available:
                        tensors[local] = handle.get_tensor(remote)

        missing = [k for k in model.state_dict() if k not in tensors]
        if missing:
            raise ValueError(f"checkpoint is missing {len(missing)} tensors, first {missing[0]}")
        model.load_state_dict(tensors)
        model.eval().to(self.device)

    @staticmethod
    def _weight_files(model_id, token):
        import glob
        import os

        if os.path.isdir(model_id):
            found = sorted(glob.glob(os.path.join(model_id, "transformer", "*.safetensors")))
            if not found:
                raise ValueError("no safetensors under transformer/")
            return found

        from huggingface_hub import hf_hub_download, list_repo_files

        names = [
            f
            for f in list_repo_files(model_id, token=token)
            if f.startswith("transformer/") and f.endswith(".safetensors")
        ]
        if not names:
            raise ValueError("no safetensors under transformer/")
        return [hf_hub_download(model_id, name, token=token) for name in names]

    def _install_patchers(self):
        self.patchers = []
        if self.blocks is None:
            return
        for block in self.blocks:
            for module in block.modules():
                processor = getattr(module, "processor", None)
                if processor is None or not hasattr(module, "to_q"):
                    continue
                patcher = PatchAttention(processor)
                module.processor = patcher
                self.patchers.append(patcher)
                break

    def drop(self, session_id: str):
        with self._lock:
            for patcher in self.patchers:
                patcher.buffers.pop(session_id, None)

    @torch.inference_mode()
    def run_patch(self, session_id, hidden, offset, sequence, arguments):
        with self._lock:
            for patcher in self.patchers:
                patcher.session = session_id
                patcher.offset = offset
                patcher.sequence = sequence
            try:
                for block in self.blocks:
                    hidden = block(hidden, *arguments)
            finally:
                for patcher in self.patchers:
                    patcher.session = None
        return hidden


class _Relay(torch.nn.Module):
    def __init__(self, session):
        super().__init__()
        self.session = session

    def forward(self, hidden_states, *args, **kwargs):
        return self.session.exchange(hidden_states, args)


class DiffusionSession:
    def __init__(self, model_id, prompt, steps, token=None, device="cpu", options=None, seed=None):
        self.model_id = model_id
        self.prompt = prompt
        self.steps = steps
        self.token = token
        self.device = device
        self.options = options or {}
        self.seed = seed
        self.pipeline = None
        self.transformer = None
        self.sequence = 0
        self.blocks = 0
        self._requests = queue.Queue(maxsize=1)
        self._answers = queue.Queue(maxsize=1)
        self._thread = None
        self._result = None
        self._error = None
        self._constants = {}

    def load(self):
        import diffusers

        index = read_index(self.model_id, self.token)
        if index is None:
            raise ValueError(f"{self.model_id} is not a diffusion pipeline")
        klass = getattr(diffusers, index["_class_name"])
        self.pipeline = klass.from_pretrained(
            self.model_id, torch_dtype=torch.float32, token=self.token
        )
        self.transformer = self.pipeline.transformer
        name, blocks = block_holder(self.transformer)
        self.blocks = len(blocks)
        setattr(self.transformer, name, torch.nn.ModuleList([_Relay(self)]))
        return self

    def output_kind(self) -> str:
        vae = getattr(self.pipeline, "vae", None)
        config = getattr(vae, "config", None)
        if config is not None and getattr(config, "sampling_rate", None):
            return "audio"
        patch = getattr(getattr(self.transformer, "config", None), "patch_size", None)
        if isinstance(patch, (list, tuple)) and len(patch) >= 3:
            return "video"
        return "image"

    def sampling_rate(self) -> int:
        config = getattr(getattr(self.pipeline, "vae", None), "config", None)
        return int(getattr(config, "sampling_rate", 0) or 44100)

    def exchange(self, hidden_states, arguments):
        self._requests.put((hidden_states, arguments))
        answer = self._answers.get()
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def _run(self):
        try:
            kwargs = dict(self.options)
            kwargs.setdefault("num_inference_steps", self.steps)
            if self.seed is not None:
                kwargs.setdefault(
                    "generator", torch.Generator(device="cpu").manual_seed(int(self.seed))
                )
            with torch.inference_mode():
                self._result = self.pipeline(self.prompt, **kwargs)
        except BaseException as exc:
            self._error = exc
        self._requests.put(None)

    def begin(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self._next()

    def _next(self):
        item = self._requests.get()
        if item is None:
            if self._error is not None:
                raise self._error
            return None
        hidden, arguments = item
        self.sequence = hidden.shape[1]
        return hidden, arguments

    def advance(self, hidden_states):
        self._answers.put(hidden_states)
        return self._next()

    def fail(self, error):
        self._answers.put(error)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def finish(self):
        if self._thread is not None:
            self._thread.join(timeout=600)
        if self._error is not None:
            raise self._error
        return encode_result(self._result, self.sampling_rate())


def changed_arguments(previous, arguments):
    fresh = {}
    for index, value in enumerate(arguments):
        if not torch.is_tensor(value):
            continue
        seen = previous.get(index)
        if seen is None or seen.shape != value.shape or not torch.equal(seen, value):
            fresh[index] = value
            previous[index] = value.clone()
    return fresh


def encode_result(result, rate):
    import io

    import numpy as np

    frames = getattr(result, "frames", None)
    if frames is not None:
        import imageio.v2 as iio

        clip = frames[0]
        if hasattr(clip, "numpy"):
            clip = clip.numpy()
        clip = np.asarray(clip)
        if clip.dtype != np.uint8:
            clip = (np.clip(clip, 0.0, 1.0) * 255).astype(np.uint8)
        import os
        import tempfile

        handle, path = tempfile.mkstemp(suffix=".mp4")
        os.close(handle)
        try:
            with iio.get_writer(path, format="ffmpeg", mode="I", fps=8, codec="libx264") as writer:
                for frame in clip:
                    writer.append_data(frame)
            with open(path, "rb") as fh:
                return fh.read(), "video/mp4"
        finally:
            os.unlink(path)

    audios = getattr(result, "audios", None)
    if audios is None:
        audios = getattr(result, "audio", None)
    if audios is not None:
        import soundfile

        wave = audios[0]
        if hasattr(wave, "detach"):
            wave = wave.detach().to(torch.float32).cpu().numpy()
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim > 1:
            wave = wave.T if wave.shape[0] < wave.shape[-1] else wave
        buffer = io.BytesIO()
        soundfile.write(buffer, wave, rate, format="WAV")
        return buffer.getvalue(), "audio/wav"

    images = getattr(result, "images", None)
    if images is not None:
        buffer = io.BytesIO()
        images[0].save(buffer, format="PNG")
        return buffer.getvalue(), "image/png"

    raise ValueError("the pipeline returned nothing recognisable")


def flatten_arguments(arguments):
    values = []
    layout = []
    for item in arguments:
        if torch.is_tensor(item):
            layout.append("T")
            values.append(item)
        elif isinstance(item, (tuple, list)):
            inner = []
            for piece in item:
                if torch.is_tensor(piece):
                    inner.append("T")
                    values.append(piece)
                else:
                    inner.append("N")
            layout.append("(" + "".join(inner) + ")")
        else:
            layout.append("N")
    return values, ",".join(layout)


def rebuild_arguments(values, layout):
    out = []
    cursor = 0
    for token in layout.split(","):
        if token == "T":
            out.append(values[cursor])
            cursor += 1
        elif token == "N":
            out.append(None)
        elif token.startswith("("):
            inner = []
            for mark in token[1:-1]:
                if mark == "T":
                    inner.append(values[cursor])
                    cursor += 1
                else:
                    inner.append(None)
            out.append(tuple(inner))
    return tuple(out)
