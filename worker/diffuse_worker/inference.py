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

    def _position_embeddings(self, hidden: torch.Tensor, start_pos: int):
        if self.slice.rotary is None:
            return None
        seq_len = hidden.shape[1]
        position_ids = torch.arange(
            start_pos, start_pos + seq_len, device=hidden.device
        ).unsqueeze(0)
        return self.slice.rotary(hidden, position_ids)

    def _run_layers(self, hidden, cache, start_pos):
        if self.slice.layers is not None and len(self.slice.layers) > 0:
            target = next(self.slice.layers[0].parameters()).dtype
            if hidden.dtype != target:
                hidden = hidden.to(target)
        pos_emb = self._position_embeddings(hidden, start_pos)
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
    def run(self, tensor_in, is_input_ids, session_id="", use_cache=False, top_k=0):
        cache = None
        start_pos = 0
        if use_cache and session_id:
            cache, start_pos = self._acquire_session(session_id)
        device = self.slice.torch_device()
        if tensor_in.device != device:
            tensor_in = tensor_in.to(device)
        if self.slice.is_first() and is_input_ids:
            hidden = self._embed(tensor_in)
        else:
            hidden = tensor_in
        seq_len = hidden.shape[1]
        hidden = self._run_layers(hidden, cache, start_pos)
        if cache is not None:
            with self._sessions_lock:
                self.seq_lens[session_id] = start_pos + seq_len
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