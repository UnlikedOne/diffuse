import inspect
import time

import torch
from transformers import DynamicCache


class SliceRunner:
    def __init__(self, model_slice):
        self.slice = model_slice
        self.caches = {}
        self.cache_seen = {}
        self.seq_lens = {}
        self._layer_params = self._detect_layer_params()

    def _touch_session(self, session_id):
        now = time.monotonic()
        self.cache_seen[session_id] = now
        stale = [s for s, t in self.cache_seen.items() if now - t > 600]
        for s in stale:
            self.caches.pop(s, None)
            self.cache_seen.pop(s, None)
            self.seq_lens.pop(s, None)

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

    @torch.no_grad()
    def run(self, tensor_in, is_input_ids, session_id="", use_cache=False):
        cache = None
        start_pos = 0
        if use_cache and session_id:
            self._touch_session(session_id)
            cache = self.caches.get(session_id)
            if cache is None:
                cache = DynamicCache()
                self.caches[session_id] = cache
                self.seq_lens[session_id] = 0
            start_pos = self.seq_lens.get(session_id, 0)
        if self.slice.is_first() and is_input_ids:
            hidden = self._embed(tensor_in)
        else:
            hidden = tensor_in
        seq_len = hidden.shape[1]
        hidden = self._run_layers(hidden, cache, start_pos)
        if cache is not None:
            self.seq_lens[session_id] = start_pos + seq_len
        if self.slice.is_last():
            return self._head(hidden)
        return hidden


    def clear_session(self, session_id: str):
        self.caches.pop(session_id, None)
        self.cache_seen.pop(session_id, None)
        self.seq_lens.pop(session_id, None)