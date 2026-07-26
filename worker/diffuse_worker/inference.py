import inspect
import threading
import time

import torch
import torch.nn.functional as F
from transformers import DynamicCache


class SliceRunner:
    def __init__(self, model_slice):
        self.slice = model_slice
        self.caches = {}
        self.cache_seen = {}
        self.seq_lens = {}
        self.rope_pos = {}
        self._sessions_lock = threading.Lock()
        self._layer_params = self._detect_layer_params()
        self._cache_indices = self._detect_cache_indices()

    def _detect_cache_indices(self):
        if not self.slice.layers:
            return []
        indices = []
        for position, layer in enumerate(self.slice.layers):
            found = None
            for module in layer.modules():
                idx = getattr(module, "layer_idx", None)
                if isinstance(idx, int):
                    found = idx
                    break
            indices.append(position if found is None else found)
        return indices

    def _touch_session(self, session_id):
        now = time.monotonic()
        self.cache_seen[session_id] = now
        stale = [s for s, t in list(self.cache_seen.items()) if now - t > 600]
        for s in stale:
            self.caches.pop(s, None)
            self.cache_seen.pop(s, None)
            self.seq_lens.pop(s, None)
            self.rope_pos.pop(s, None)

    def _acquire_session(self, session_id):
        with self._sessions_lock:
            self._touch_session(session_id)
            cache = self.caches.get(session_id)
            if cache is None:
                cache = DynamicCache()
                self.caches[session_id] = cache
                self.seq_lens[session_id] = 0
            return cache, self.seq_lens.get(session_id, 0)

    def _detect_layer_params(self):
        if not self.slice.layers:
            return set()
        layer = self.slice.layers[0]
        try:
            sig = inspect.signature(layer.forward)
            return set(sig.parameters.keys())
        except (ValueError, TypeError):
            return set()

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.slice.embed_tokens(input_ids)
        if getattr(self.slice, "pos_embed", None) is not None:
            positions = torch.arange(input_ids.shape[-1], device=input_ids.device)
            hidden = hidden + self.slice.pos_embed(positions)
        return hidden

    def _position_embeddings(self, hidden: torch.Tensor, start_pos: int, position_ids=None):
        if self.slice.rotary is None:
            return None
        seq_len = hidden.shape[1]
        if position_ids is None:
            position_ids = torch.arange(
                start_pos, start_pos + seq_len, device=hidden.device
            ).unsqueeze(0)
        else:
            return self.slice.rotary(hidden, position_ids.to(hidden.device))
        try:
            return self.slice.rotary(hidden, position_ids)
        except IndexError:
            # Some multimodal decoders index position ids on several axes at
            # once, time, height and width, rather than one. Giving each axis
            # the same sequence is what these models do for text; it is only an
            # approximation once media is in the prompt, where the axes are
            # meant to differ.
            axes = getattr(
                getattr(self.slice.rotary, "config", None), "rope_scaling", None
            )
            sections = (axes or {}).get("mrope_section") if isinstance(axes, dict) else None
            count = 3 if sections is None else max(3, len(sections))
            return self.slice.rotary(hidden, position_ids.expand(count, -1, -1))

    def _run_layers(self, hidden, cache, start_pos, position_ids=None, rope_start=None):
        if self.slice.layers is not None and len(self.slice.layers) > 0:
            target = next(self.slice.layers[0].parameters()).dtype
            if hidden.dtype != target:
                hidden = hidden.to(target)
        pos_emb = self._position_embeddings(
            hidden, start_pos if rope_start is None else rope_start, position_ids
        )
        seq_len = hidden.shape[1]
        params = self._layer_params
        base_kwargs = {}
        if pos_emb is not None and "position_embeddings" in params:
            base_kwargs["position_embeddings"] = pos_emb
        if "position_ids" in params and pos_emb is None:
            base_kwargs["position_ids"] = torch.arange(
                start_pos, start_pos + seq_len, device=hidden.device
            )
        if cache is not None:
            if "past_key_values" in params:
                base_kwargs["past_key_values"] = cache
            elif "past_key_value" in params:
                base_kwargs["past_key_value"] = cache
            if "use_cache" in params:
                base_kwargs["use_cache"] = True
            if "cache_position" in params:
                base_kwargs["cache_position"] = torch.arange(
                    start_pos, start_pos + seq_len, device=hidden.device
                )
        for layer in self.slice.layers:
            out = layer(hidden, **base_kwargs)
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden
    
    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.slice.norm(hidden)
        return self.slice.lm_head(hidden)

    def _topk_row(self, row: torch.Tensor, top_k: int):
        k = min(top_k, row.shape[-1])
        values, indices = torch.topk(row.float(), k)
        by_index = torch.argsort(indices)
        values, indices = values[by_index], indices[by_index]
        by_value = torch.argsort(-values, stable=True)
        return indices[by_value].to(torch.int64), values[by_value]

    def cached_length(self, session_id: str) -> int:
        with self._sessions_lock:
            return self.seq_lens.get(session_id, 0)

    @torch.inference_mode()
    def run(self, tensor_in, is_input_ids, session_id="", use_cache=False, top_k=0, position_ids=None):
        cache = None
        start_pos = 0
        rope_start = 0
        if use_cache and session_id:
            cache, start_pos = self._acquire_session(session_id)
            with self._sessions_lock:
                # Media can compress a long prompt into fewer rotary positions
                # than it has cache entries, so the two counters diverge and the
                # rotary one has to be tracked separately.
                rope_start = self.rope_pos.get(session_id, start_pos)
        device = self.slice.torch_device()
        if tensor_in.device != device:
            tensor_in = tensor_in.to(device)
        if self.slice.is_first() and is_input_ids:
            hidden = self._embed(tensor_in)
        else:
            hidden = tensor_in
        seq_len = hidden.shape[1]
        hidden = self._run_layers(hidden, cache, start_pos, position_ids, rope_start)
        if cache is not None:
            with self._sessions_lock:
                self.seq_lens[session_id] = start_pos + seq_len
                if position_ids is not None:
                    self.rope_pos[session_id] = int(position_ids.max()) + 1
                else:
                    self.rope_pos[session_id] = rope_start + seq_len
        if not self.slice.is_last():
            return hidden
        if top_k > 0:
            logits = self._head(hidden[:, -1:, :])
            return self._topk_row(logits[0, -1], top_k)
        return self._head(hidden)

    @torch.inference_mode()
    def run_batch(self, items, top_k=0):
        sessions = [it.session_id for it in items]
        caches, lengths = [], []
        for sid in sessions:
            cache, start_pos = self._acquire_session(sid)
            caches.append(cache)
            lengths.append(start_pos)

        if min(lengths) == 0 or len(self.slice.layers) == 0:
            return [
                self.run(
                    it.tensor_in,
                    is_input_ids=it.is_input_ids,
                    session_id=it.session_id,
                    use_cache=True,
                    top_k=top_k,
                )
                for it in items
            ]

        device = self.slice.torch_device()
        inputs = [
            it.tensor_in.to(device) if it.tensor_in.device != device else it.tensor_in
            for it in items
        ]
        if self.slice.is_first() and items[0].is_input_ids:
            hidden = torch.cat([self._embed(x) for x in inputs], dim=0)
        else:
            hidden = torch.cat(inputs, dim=0)

        target = next(self.slice.layers[0].parameters()).dtype
        if hidden.dtype != target:
            hidden = hidden.to(target)

        batch = len(items)
        max_len = max(lengths)

        merged = DynamicCache()
        for j in self._cache_indices:
            keys, values = [], []
            for cache, length in zip(caches, lengths):
                k = cache.layers[j].keys
                v = cache.layers[j].values
                pad = max_len - length
                if pad:
                    k = F.pad(k, (0, 0, pad, 0))
                    v = F.pad(v, (0, 0, pad, 0))
                keys.append(k)
                values.append(v)
            merged.update(torch.cat(keys, dim=0), torch.cat(values, dim=0), j)

        mask = torch.zeros(batch, 1, 1, max_len + 1, dtype=hidden.dtype, device=device)
        blocked = torch.finfo(hidden.dtype).min
        for i, length in enumerate(lengths):
            pad = max_len - length
            if pad:
                mask[i, :, :, :pad] = blocked

        position_ids = torch.tensor(
            [[length] for length in lengths], dtype=torch.long, device=device
        )

        kwargs = {}
        params = self._layer_params
        if self.slice.rotary is not None and "position_embeddings" in params:
            kwargs["position_embeddings"] = self.slice.rotary(hidden, position_ids)
        elif "position_ids" in params:
            kwargs["position_ids"] = position_ids
        if "attention_mask" in params:
            kwargs["attention_mask"] = mask
        if "past_key_values" in params:
            kwargs["past_key_values"] = merged
        elif "past_key_value" in params:
            kwargs["past_key_value"] = merged
        if "use_cache" in params:
            kwargs["use_cache"] = True

        for layer in self.slice.layers:
            out = layer(hidden, **kwargs)
            hidden = out[0] if isinstance(out, tuple) else out

        with self._sessions_lock:
            for i, (sid, length) in enumerate(zip(sessions, lengths)):
                split = DynamicCache()
                for j in self._cache_indices:
                    start = max_len - length
                    k = merged.layers[j].keys[i : i + 1, :, start:, :].contiguous()
                    v = merged.layers[j].values[i : i + 1, :, start:, :].contiguous()
                    split.update(k, v, j)
                self.caches[sid] = split
                self.seq_lens[sid] = length + 1

        if not self.slice.is_last():
            return [hidden[i : i + 1] for i in range(batch)]
        logits = self._head(hidden)
        if top_k > 0:
            return [self._topk_row(logits[i, -1], top_k) for i in range(batch)]
        return [logits[i : i + 1] for i in range(batch)]

    def clear_session(self, session_id: str):
        with self._sessions_lock:
            self.caches.pop(session_id, None)
            self.cache_seen.pop(session_id, None)
            self.seq_lens.pop(session_id, None)
            self.rope_pos.pop(session_id, None)

MODALITIES = (
    (
        "get_image_features",
        "image_token_id",
        ("pixel_values", "pixel_attention_mask", "image_grid_thw", "image_sizes"),
    ),
    (
        "get_video_features",
        "video_token_id",
        ("pixel_values_videos", "video_grid_thw", "video_sizes"),
    ),
    (
        "get_audio_features",
        "audio_token_id",
        ("input_features", "feature_attention_mask", "audio_attention_mask"),
    ),
)


def _feature_tensor(out, hidden_size):
    """Pick the tensor that already lives in the decoder's hidden space.

    Feature getters are not consistent about what they hand back. Some return a
    plain projected tensor, others an output object where the raw tower states
    sit in `last_hidden_state` and the projected ones in `pooler_output`.
    Choosing by width rather than by attribute name keeps this working across
    models instead of encoding one model's habits."""
    def as_tensor(value):
        # A getter may hand back one tensor per attachment rather than a single
        # block: Qwen2-VL splits its projected embeds per video. Joining them
        # keeps the rows in the order the placeholders appear.
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)) and value and all(
            isinstance(v, torch.Tensor) for v in value
        ):
            return torch.cat([v.reshape(-1, v.shape[-1]) for v in value])
        return None

    candidates = []
    for attr in ("pooler_output", "last_hidden_state", "image_embeds", "audio_embeds"):
        joined = as_tensor(getattr(out, attr, None))
        if joined is not None:
            candidates.append(joined)
    joined = as_tensor(out)
    if joined is not None:
        candidates.append(joined)

    for tensor in candidates:
        if tensor.shape[-1] == hidden_size:
            return tensor
    if candidates:
        return candidates[0]
    raise ValueError("the feature getter returned nothing tensor-like")


class MediaEmbedder:
    """Turns text plus media into the hidden states the decoder stack expects.

    This runs where the embeddings live, which on a Diffuse client is the
    machine itself: the picture or the recording is consumed here, and what
    leaves is the same transformed activations a text prompt would produce."""

    def __init__(self, model_slice):
        self.slice = model_slice

    def _token_id(self, attr):
        cfg = self.slice.model.config if self.slice.model is not None else None
        for holder in (cfg, getattr(cfg, "text_config", None)):
            value = getattr(holder, attr, None) if holder is not None else None
            if isinstance(value, int):
                return value
        tokenizer = self.slice.tokenizer
        if tokenizer is not None:
            token = getattr(cfg, attr.replace("_id", ""), None)
            if isinstance(token, str):
                resolved = tokenizer.convert_tokens_to_ids(token)
                if isinstance(resolved, int) and resolved >= 0:
                    return resolved
        return None

    def _feature_source(self, method):
        for holder in (self.slice.model, getattr(self.slice.model, "model", None)):
            if holder is not None and hasattr(holder, method):
                return getattr(holder, method)
        return None

    def _project(self, features, hidden_size):
        """Last resort when the getter only exposed raw tower states."""
        for name, module in (self.slice.tower or {}).items():
            if module is None or "vision" in name or "audio" in name:
                continue
            try:
                projected = module(features)
            except Exception:
                continue
            if isinstance(projected, torch.Tensor) and projected.shape[-1] == hidden_size:
                return projected
        raise ValueError(
            f"media features are {features.shape[-1]} wide but the decoder expects "
            f"{hidden_size}, and no projector in the tower bridged them"
        )

    def rope_positions(self, inputs):
        """The multi-axis positions this model wants, when it wants any.

        A decoder that indexes time, height and width separately cannot rebuild
        them from a sequence length: they depend on how the media was laid out.
        They are computed here, where the grids are, and travel with the
        activations."""
        model = self.slice.model
        for holder in (getattr(model, "model", None), model):
            getter = getattr(holder, "get_rope_index", None)
            if getter is None:
                continue
            try:
                accepted = set(inspect.signature(getter).parameters)
                kwargs = {k: v for k, v in inputs.items() if k in accepted}
                result = getter(**kwargs)
            except Exception:
                continue
            positions = result[0] if isinstance(result, tuple) else result
            if isinstance(positions, torch.Tensor) and positions.dim() == 3:
                return positions
        return None

    @torch.inference_mode()
    def embed(self, inputs) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        device = self.slice.torch_device()
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        embeds = self.slice.embed_tokens(input_ids)

        for method, token_attr, keys in MODALITIES:
            primary = keys[0]
            if primary not in inputs:
                continue
            source = self._feature_source(method)
            token_id = self._token_id(token_attr)
            if source is None or token_id is None:
                raise ValueError(
                    f"this build cannot embed {primary}: no {method} on the model "
                    f"or no {token_attr} in its config"
                )
            accepted = set(inspect.signature(source).parameters)
            kwargs = {
                k: (v.to(device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
                if k in keys and (k in accepted or "kwargs" in accepted)
            }
            hidden_size = embeds.shape[-1]
            features = _feature_tensor(source(**kwargs), hidden_size)
            if features.shape[-1] != hidden_size:
                features = self._project(features, hidden_size)
            features = features.reshape(-1, hidden_size).to(embeds.dtype)
            mask = input_ids == token_id
            slots = int(mask.sum())
            if slots != features.shape[0]:
                raise ValueError(
                    f"{primary}: {features.shape[0]} feature rows for {slots} "
                    f"placeholder tokens"
                )
            embeds = embeds.masked_scatter(mask.unsqueeze(-1), features)

        return embeds
