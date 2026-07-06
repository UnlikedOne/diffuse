from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedSlice:
    model_id: str
    start_layer: int
    end_layer: int
    total_layers: int


def _resolve_backbone(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        base = model.model
        return {
            "kind": "llama",
            "layers": base.layers,
            "embed": base.embed_tokens,
            "pos_embed": None,
            "rotary": getattr(base, "rotary_emb", None),
            "norm": base.norm,
            "lm_head": model.lm_head,
        }
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        base = model.transformer
        return {
            "kind": "gpt2",
            "layers": base.h,
            "embed": base.wte,
            "pos_embed": base.wpe,
            "rotary": None,
            "norm": base.ln_f,
            "lm_head": model.lm_head,
        }
    raise ValueError(f"unsupported architecture: {type(model).__name__}")


class ModelSlice:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_id: str | None = None
        self.kind: str | None = None
        self.start_layer: int = 0
        self.end_layer: int = 0
        self.total_layers: int = 0
        self.tokenizer = None
        self.embed_tokens = None
        self.pos_embed = None
        self.rotary = None
        self.layers = None
        self.norm = None
        self.lm_head = None

    def load(
        self,
        model_id: str,
        start_layer: int,
        end_layer: int,
        hf_token: str | None = None,
        cache_dir: str | None = None,
    ) -> LoadedSlice:
        cfg = AutoConfig.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir)
        total = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer")

        if start_layer == 0 and end_layer == 0:
            self.model_id = model_id
            self.start_layer = 0
            self.end_layer = 0
            self.total_layers = total
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=hf_token, cache_dir=cache_dir
            )
            return LoadedSlice(model_id, 0, 0, total)

        if end_layer > total:
            raise ValueError(f"end_layer {end_layer} exceeds total {total}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            cache_dir=cache_dir,
            dtype=torch.float32,
        )
        model.eval()
        parts = _resolve_backbone(model)
        self.model_id = model_id
        self.kind = parts["kind"]
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.total_layers = total
        self.layers = parts["layers"][start_layer:end_layer]
        self.rotary = parts["rotary"]
        if start_layer == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=hf_token, cache_dir=cache_dir
            )
            self.embed_tokens = parts["embed"]
            self.pos_embed = parts["pos_embed"]
        if end_layer == total:
            self.norm = parts["norm"]
            self.lm_head = parts["lm_head"]
        return LoadedSlice(model_id, start_layer, end_layer, total)

    def is_first(self) -> bool:
        return self.start_layer == 0

    def is_last(self) -> bool:
        return self.end_layer == self.total_layers