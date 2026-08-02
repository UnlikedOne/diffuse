import json
import queue
import threading

import torch

_NAMED_STACKS = (
    "blocks",
    "transformer_blocks",
    "single_transformer_blocks",
    "temporal_transformer_blocks",
    "layers",
)


def _uniform_blocks(blocks) -> bool:
    kinds = {type(block).__name__ for block in blocks}
    if len(kinds) != 1:
        return False
    return any(hasattr(module, "to_q") for module in blocks[0].modules())


def block_holders(model):
    """Every list of transformer blocks the model runs, in the order it runs them.

    Some models keep one list, others keep two: Flux and HunyuanVideo run a
    joint stack and then a single-stream stack. Both are block lists to split,
    so both are found here rather than only the first."""
    lists = [
        (name, child)
        for name, child in model.named_children()
        if isinstance(child, torch.nn.ModuleList) and len(child) > 0
    ]
    named = [(name, blocks) for name, blocks in lists if name in _NAMED_STACKS]
    if named:
        return named
    return [(name, blocks) for name, blocks in lists if _uniform_blocks(blocks)]


def block_holder(model):
    holders = block_holders(model)
    if not holders:
        raise ValueError("this model exposes no list of transformer blocks")
    return holders[0]


def _meta_build(klass, config):
    try:
        with torch.device("meta"):
            return klass(**config)
    except Exception:
        return None


def _settings(config: dict) -> dict:
    return {k: v for k, v in config.items() if not k.startswith("_")}


def stack_keys(klass, config: dict) -> dict:
    """Which config entry decides the length of which block list.

    The names differ from model to model — num_layers, num_single_layers,
    depth — and nothing in the file says which list each one governs. Rather
    than keep a table that goes stale, each integer setting is nudged by one on
    a weightless copy and the list that grew is the one it controls."""
    base = _meta_build(klass, config)
    if base is None:
        return {}
    lengths = {name: len(blocks) for name, blocks in block_holders(base)}
    keys = {}
    for key, value in config.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            continue
        probe = dict(config)
        probe[key] = value + 1
        model = _meta_build(klass, probe)
        if model is None:
            continue
        grown = {name: len(blocks) for name, blocks in block_holders(model)}
        changed = [name for name, length in lengths.items() if grown.get(name) == length + 1]
        if len(changed) == 1 and set(grown) == set(lengths):
            keys[changed[0]] = key
    return keys


def _key_of_length(settings: dict, length: int) -> str | None:
    """The one setting that already says how long this stack is."""
    found = [
        key
        for key, value in settings.items()
        if isinstance(value, int) and not isinstance(value, bool) and value == length
    ]
    return found[0] if len(found) == 1 else None


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
    """How many blocks this transformer runs in total, across every stack."""
    import diffusers

    klass = getattr(diffusers, config.get("_class_name", ""), None)
    if klass is not None:
        model = _meta_build(klass, _settings(config))
        if model is not None:
            holders = block_holders(model)
            if holders:
                return sum(len(blocks) for _, blocks in holders)
    for key in ("num_layers", "num_hidden_layers", "num_blocks", "depth"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


class PatchAttention:
    """Self-attention over the whole picture while only a patch is being carried.

    A patch cannot attend to itself alone and still mean what it meant: the
    rest of the image is missing. PipeFusion answers that with the previous
    timestep's activations, which are close enough because consecutive steps
    barely differ. The buffer here is that memory, one per session and per
    attention.

    Two ways to read it are kept. The plain one hands the model's own attention
    the full buffer and takes the patch's rows back out, which is right for any
    model because it changes nothing the model does. The quick one asks only
    for the patch's queries, which is what makes patching worth doing but
    assumes how this model applies its rotary positions. Rather than trust that
    assumption per family, both are run once on the first real patch and the
    quick one is kept only if it agreed."""

    def __init__(self, attention):
        self.attention = attention
        self.buffers: dict[str, torch.Tensor] = {}
        self.session = None
        self.offset = 0
        self.sequence = 0
        self.mode = None

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
            or buffer.shape[0] != hidden_states.shape[0]
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

    def __call__(self, attn, hidden_states, *args, **kwargs):
        if self.session is None:
            return self.attention(attn, hidden_states, *args, **kwargs)
        context = self.context_for(hidden_states)
        if context.shape[1] == hidden_states.shape[1]:
            return self.attention(attn, hidden_states, *args, **kwargs)

        start = self.offset
        stop = start + hidden_states.shape[1]
        if self.mode == "fast":
            quick = self._fast(attn, hidden_states, context, args, kwargs, start, stop)
            if quick is not None:
                return quick
        exact = self._exact(attn, context, args, kwargs, start, stop)
        if self.mode is None:
            quick = self._fast(attn, hidden_states, context, args, kwargs, start, stop)
            self.mode = "fast" if _agree(quick, exact) else "exact"
        return exact

    def _exact(self, attn, context, args, kwargs, start, stop):
        out = self.attention(attn, context, *args, **kwargs)
        if isinstance(out, tuple):
            return (out[0][:, start:stop],) + tuple(out[1:])
        return out[:, start:stop]

    def _fast(self, attn, hidden_states, context, args, kwargs, start, stop):
        try:
            return self._patch_query(attn, hidden_states, context, args, kwargs, start, stop)
        except Exception:
            return None

    def _patch_query(self, attn, hidden_states, context, args, kwargs, start, stop):
        extras = [v for v in list(args) + list(kwargs.values()) if v is not None]
        rotary = _rotary_of(extras, context.shape[1])
        if rotary is None or len(extras) != 1 or not hasattr(attn, "to_out"):
            return None

        cos, sin = rotary
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
        query = self._rope(query, cos[:, start:stop], sin[:, start:stop])
        key = self._rope(key, cos, sin)

        out = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        out = out.transpose(1, 2).flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        return attn.to_out[1](out)


def _rotary_of(values, sequence):
    for value in values:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            continue
        cos, sin = value
        if not (torch.is_tensor(cos) and torch.is_tensor(sin)):
            continue
        if cos.dim() >= 2 and cos.shape[1] == sequence and cos.shape == sin.shape:
            return cos, sin
    return None


def _agree(quick, exact) -> bool:
    if quick is None or isinstance(exact, tuple) or not torch.is_tensor(quick):
        return False
    if quick.shape != exact.shape:
        return False
    scale = float(exact.abs().max())
    return float((quick - exact).abs().max()) <= max(1e-4 * scale, 1e-5)


class DiffusionStack:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_id = None
        self.start_block = 0
        self.end_block = 0
        self.total_blocks = 0
        self.model = None
        self.stacks: dict[str, torch.nn.ModuleList] = {}
        self.patchers: list[PatchAttention] = []
        self._lock = threading.Lock()

    def load(self, model_id: str, start: int, end: int, token: str | None = None):
        config = transformer_config(model_id, token)
        if config is None:
            raise ValueError(f"{model_id} has no diffusion transformer")

        import diffusers

        name = config.get("_class_name", "")
        klass = getattr(diffusers, name, None)
        if klass is None:
            raise ValueError(f"this build of diffusers has no {name}")
        settings = _settings(config)
        base = _meta_build(klass, settings)
        if base is None:
            raise ValueError(f"cannot build {name} from its own config")
        holders = block_holders(base)
        if not holders:
            raise ValueError(f"{name} exposes no list of transformer blocks")
        total = sum(len(blocks) for _, blocks in holders)
        if end > total:
            raise ValueError(f"end block {end} exceeds total {total}")
        keys = stack_keys(klass, settings)

        ranges = []
        offset = 0
        for stack, blocks in holders:
            length = len(blocks)
            first = min(max(start - offset, 0), length)
            last = min(max(end - offset, 0), length)
            ranges.append((stack, first, max(last - first, 0), length))
            offset += length

        for stack, _, kept, length in ranges:
            key = keys.get(stack) or _key_of_length(settings, length)
            if key is None:
                raise ValueError(f"cannot tell which setting sizes {stack} on {name}")
            settings[key] = max(kept, 1)
        ranges = [(stack, first, kept) for stack, first, kept, _ in ranges]

        model = klass(**settings)
        for stack, _, kept in ranges:
            if kept == 0:
                setattr(model, stack, torch.nn.ModuleList())

        self._materialise(model, model_id, ranges, token)

        self.model_id = model_id
        self.start_block = start
        self.end_block = end
        self.total_blocks = total
        self.model = model
        self.stacks = {stack: getattr(model, stack) for stack, _, _ in ranges}
        self._install_patchers()
        return total

    def _materialise(self, model, model_id, ranges, token):
        from safetensors import safe_open

        sources = {}
        for stack, first, kept in ranges:
            prefix = f"{stack}."
            for index in range(kept):
                sources[f"{prefix}{index}."] = f"{prefix}{first + index}."

        wanted = {}
        for key in model.state_dict():
            remote = key
            for local, source in sources.items():
                if key.startswith(local):
                    remote = source + key[len(local) :]
                    break
            wanted[key] = remote

        tensors = {}
        for path in self._weight_files(model_id, token):
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
        for blocks in self.stacks.values():
            for block in blocks:
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
                for key in [k for k in patcher.buffers if k.startswith(session_id)]:
                    patcher.buffers.pop(key, None)

    @torch.inference_mode()
    def run_patch(self, session_id, hidden, offset, sequence, arguments, plan):
        blocks = self.stacks.get(plan["stack"])
        if blocks is None:
            raise ValueError(f"this slice holds no stack called {plan['stack']}")
        state = _split_state(hidden, plan["state"])
        with self._lock:
            self._aim(session_id, offset, sequence)
            try:
                for block in blocks:
                    args, kwargs = _bind(plan, arguments, state)
                    out = block(*args, **kwargs)
                    produced = (out,) if plan["single"] else tuple(out)
                    for slot, value in zip(plan["returns"], produced):
                        state[slot] = value
            finally:
                self._aim(None, 0, 0)
        return torch.cat(state, dim=1) if len(state) > 1 else state[0]

    def _aim(self, session_id, offset, sequence):
        for patcher in self.patchers:
            patcher.session = session_id
            patcher.offset = offset
            patcher.sequence = sequence


class _Relay(torch.nn.Module):
    def __init__(self, session, stack):
        super().__init__()
        self.session = session
        self.stack = stack

    def forward(self, *args, **kwargs):
        return self.session.exchange(self.stack, args, kwargs)


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
        self.max_patches = 0
        self.originals = {}
        self.plans = {}
        self._requests = queue.Queue(maxsize=1)
        self._answers = queue.Queue(maxsize=1)
        self._thread = None
        self._result = None
        self._error = None

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
        holders = block_holders(self.transformer)
        if not holders:
            raise ValueError(f"{self.model_id} exposes no list of transformer blocks")
        self.blocks = sum(len(blocks) for _, blocks in holders)
        for name, blocks in holders:
            self.originals[name] = blocks
            setattr(self.transformer, name, torch.nn.ModuleList([_Relay(self, name)]))
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

    def exchange(self, stack, args, kwargs):
        plan = self.plans.get(stack)
        if plan is not None and (
            len(plan["args"]) != len(args) or set(plan["kwargs"]) != set(kwargs)
        ):
            plan = None
        if plan is None:
            plan = build_plan(stack, args, kwargs, self.originals[stack][0])
            self.plans[stack] = plan
            if len(plan["state"]) > 1:
                self.max_patches = 1
        hidden, tensors = pack_call(plan, args, kwargs)
        self._requests.put((hidden, tensors, plan))
        answer = self._answers.get()
        if isinstance(answer, BaseException):
            raise answer
        state = _split_state(answer, plan["state"])
        produced = tuple(state[slot] for slot in plan["returns"])
        return produced[0] if plan["single"] else produced

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
        hidden, tensors, plan = item
        self.sequence = hidden.shape[1]
        return hidden, tensors, plan

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


def build_plan(stack, args, kwargs, block):
    """How one call into a block stack is taken apart for the wire.

    A block is handed positional arguments and keyword arguments, only some of
    which are tensors, and returns one tensor or several. Which of its inputs
    each output replaces is what the loop over the blocks needs to know, and it
    is not written down anywhere: it is learned by running one block once and
    matching the shapes it gave back."""
    with torch.inference_mode():
        out = block(*args, **kwargs)
    single = not isinstance(out, (tuple, list))
    produced = (out,) if single else tuple(out)

    places = []
    for index, value in enumerate(args):
        if torch.is_tensor(value):
            places.append((("a", index), value))
    for name, value in kwargs.items():
        if torch.is_tensor(value):
            places.append((("k", name), value))

    slots = []
    returns = []
    taken = {}
    for item in produced:
        if not torch.is_tensor(item):
            raise ValueError("a block returned something that is not a tensor")
        found = None
        for where, value in places:
            if where in taken:
                continue
            if value.shape == item.shape and value.dtype == item.dtype:
                found = where
                break
        if found is None:
            raise ValueError("a block returned a tensor that replaces none of its inputs")
        taken[found] = len(slots)
        returns.append(len(slots))
        slots.append(found)

    lengths = []
    for where in slots:
        value = args[where[1]] if where[0] == "a" else kwargs[where[1]]
        if value.dim() < 2:
            raise ValueError("a block carries a state tensor with no sequence axis")
        lengths.append(int(value.shape[1]))

    tensors = []
    encoded_args = [_encode(value, slots, tensors, ("a", index)) for index, value in enumerate(args)]
    encoded_kwargs = {
        name: _encode(value, slots, tensors, ("k", name)) for name, value in kwargs.items()
    }
    return {
        "stack": stack,
        "args": encoded_args,
        "kwargs": encoded_kwargs,
        "state": lengths,
        "returns": returns,
        "single": single,
    }


def _encode(value, slots, tensors, where=None):
    if torch.is_tensor(value):
        if where is not None and where in slots:
            return {"s": slots.index(where)}
        tensors.append(value)
        return {"t": len(tensors) - 1}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"c": value}
    if isinstance(value, (tuple, list)):
        kind = "p" if isinstance(value, tuple) else "l"
        return {kind: [_encode(item, slots, tensors) for item in value]}
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {"d": {k: _encode(v, slots, tensors) for k, v in value.items()}}
    raise ValueError(f"a block was handed a {type(value).__name__}, which cannot travel")


def pack_call(plan, args, kwargs):
    """The tensors of one call, split into the state and everything else."""
    state = [None] * len(plan["state"])
    tensors = []
    _collect(plan["args"], args, state, tensors)
    _collect(
        [plan["kwargs"][name] for name in plan["kwargs"]],
        [kwargs[name] for name in plan["kwargs"]],
        state,
        tensors,
    )
    if any(item is None for item in state):
        raise ValueError("the call did not carry every state tensor the plan expects")
    hidden = torch.cat(state, dim=1) if len(state) > 1 else state[0]
    return hidden, tensors


def _collect(encoded, values, state, tensors):
    for shape, value in zip(encoded, values):
        if "s" in shape:
            state[shape["s"]] = value
        elif "t" in shape:
            tensors.append(value)
        elif "l" in shape or "p" in shape:
            _collect(shape.get("l") or shape.get("p"), list(value), state, tensors)
        elif "d" in shape:
            inner = shape["d"]
            _collect([inner[k] for k in inner], [value[k] for k in inner], state, tensors)


def _split_state(hidden, lengths):
    if len(lengths) <= 1 or hidden.shape[1] != sum(lengths):
        return [hidden]
    pieces = []
    position = 0
    for length in lengths:
        pieces.append(hidden[:, position : position + length])
        position += length
    return pieces


def _bind(plan, tensors, state):
    args = [_decode(shape, tensors, state) for shape in plan["args"]]
    kwargs = {name: _decode(shape, tensors, state) for name, shape in plan["kwargs"].items()}
    return args, kwargs


def _decode(shape, tensors, state):
    if "s" in shape:
        return state[shape["s"]]
    if "t" in shape:
        return tensors[shape["t"]]
    if "c" in shape:
        return shape["c"]
    if "l" in shape:
        return [_decode(item, tensors, state) for item in shape["l"]]
    if "p" in shape:
        return tuple(_decode(item, tensors, state) for item in shape["p"])
    if "d" in shape:
        return {k: _decode(v, tensors, state) for k, v in shape["d"].items()}
    raise ValueError("the wire layout carries a shape this build does not know")


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
