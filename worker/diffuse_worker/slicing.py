import re
from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub import get_safetensors_metadata, hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_LAYER_RE = re.compile(r"\.(?:layers|h|blocks|block)\.(\d+)\.")


@dataclass
class LoadedSlice:
    model_id: str
    start_layer: int
    end_layer: int
    total_layers: int


def _find_layer_list(module, expected: int):
    for name, child in module.named_children():
        if isinstance(child, nn.ModuleList) and len(child) == expected:
            return name, child
    for _, child in module.named_children():
        found = _find_layer_list(child, expected)
        if found is not None:
            return found
    return None


def _find_layer_parent(module, expected: int):
    for name, child in module.named_children():
        if isinstance(child, nn.ModuleList) and len(child) == expected:
            return module, name
    for _, child in module.named_children():
        found = _find_layer_parent(child, expected)
        if found is not None:
            return found
    return None


def _find_final_norm(backbone, layers_attr: str):
    candidates = []
    for name, child in backbone.named_children():
        if name == layers_attr:
            continue
        if "norm" in name.lower() or name in ("ln_f", "final_layernorm"):
            candidates.append(child)
        elif isinstance(child, nn.Module) and "norm" in type(child).__name__.lower():
            candidates.append(child)
    return candidates[-1] if candidates else None


def _find_position_embeddings(backbone, input_embed):
    for name, child in backbone.named_children():
        if child is input_embed:
            continue
        if isinstance(child, nn.Embedding) and name in (
            "wpe",
            "position_embeddings",
            "embed_positions",
        ):
            return child
    return None


def _resolve_backbone(model):
    cfg = model.config
    total = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    if total is None:
        raise ValueError("cannot determine layer count from config")

    backbone = getattr(model, "base_model", None) or model
    found = _find_layer_list(backbone, total) or _find_layer_list(model, total)
    if found is None:
        raise ValueError(
            f"cannot locate a list of {total} transformer layers in {type(model).__name__}"
        )
    layers_attr, layers = found

    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings() or getattr(model, "lm_head", None)

    rotary = None
    for name, child in backbone.named_children():
        if "rotary" in name.lower() or "rotary" in type(child).__name__.lower():
            rotary = child
            break

    return {
        "kind": type(model).__name__,
        "layers": layers,
        "layers_attr": layers_attr,
        "embed": embed,
        "pos_embed": _find_position_embeddings(backbone, embed),
        "rotary": rotary,
        "norm": _find_final_norm(backbone, layers_attr),
        "lm_head": lm_head,
        "backbone": backbone,
    }


def _is_embedding_tensor(name: str) -> bool:
    return "embed" in name or "wte" in name or "wpe" in name


def _tensor_is_needed(name, start_layer, end_layer, total, tied=False):
    m = _LAYER_RE.search(name)
    if m:
        idx = int(m.group(1))
        return start_layer <= idx < end_layer
    if _is_embedding_tensor(name):
        return start_layer == 0 or (tied and end_layer == total)
    return end_layer == total


def _plan_download(model_id, start_layer, end_layer, total, hf_token, tied=False):
    meta = get_safetensors_metadata(model_id, token=hf_token)
    plan = {}
    for filename, file_meta in meta.files_metadata.items():
        needed = [
            name
            for name in file_meta.tensors
            if _tensor_is_needed(name, start_layer, end_layer, total, tied)
        ]
        if needed:
            plan[filename] = needed
    return plan


def _remap_key(name: str, start_layer: int) -> str:
    m = _LAYER_RE.search(name)
    if not m:
        return name
    idx = int(m.group(1))
    return name[: m.start(1)] + str(idx - start_layer) + name[m.end(1) :]

def _detach_unused_modules(model, parts, start_layer, end_layer, total):
    unused = []
    if start_layer != 0:
        unused.append(parts.get("embed"))
        unused.append(parts.get("pos_embed"))
    if end_layer != total:
        unused.append(parts.get("norm"))
        unused.append(parts.get("lm_head"))
    targets = {id(m) for m in unused if m is not None}
    if not targets:
        return
    for module in model.modules():
        for attr_name, child in list(module.named_children()):
            if id(child) in targets:
                setattr(module, attr_name, None)

def _rebuild_meta_buffers(model, cfg):
    for module in list(model.modules()):
        stale = [
            bname
            for bname, buf in module.named_buffers(recurse=False)
            if buf.device.type == "meta"
        ]
        if not stale:
            continue
        rebuilt = None
        for attempt in (lambda: type(module)(config=cfg), lambda: type(module)(cfg)):
            try:
                rebuilt = attempt()
                break
            except Exception:
                continue
        if rebuilt is None:
            continue
        for bname, buf in rebuilt.named_buffers(recurse=False):
            if bname in stale and buf.device.type != "meta":
                module.register_buffer(bname, buf, persistent=False)


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

        tied = bool(getattr(cfg, "tie_word_embeddings", False))
        plan = None
        try:
            plan = _plan_download(
                model_id, start_layer, end_layer, total, hf_token, tied
            )
        except Exception as exc:
            print(f"partial load unavailable ({exc}), downloading the full model")

        if plan:
            try:
                self._load_partial(
                    model_id,
                    cfg,
                    plan,
                    start_layer,
                    end_layer,
                    total,
                    tied,
                    hf_token,
                    cache_dir,
                )
            except Exception as exc:
                print(f"partial load failed ({exc}), downloading the full model")
                self._load_full(
                    model_id, start_layer, end_layer, total, hf_token, cache_dir
                )
        else:
            self._load_full(
                model_id, start_layer, end_layer, total, hf_token, cache_dir
            )

        if start_layer == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=hf_token, cache_dir=cache_dir
            )

        self.model_id = model_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.total_layers = total
        return LoadedSlice(model_id, start_layer, end_layer, total)

    def _load_partial(
        self,
        model_id,
        cfg,
        plan,
        start_layer,
        end_layer,
        total,
        tied,
        hf_token,
        cache_dir,
    ):
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg)
        model.eval()

        parts = _resolve_backbone(model)
        parent_attr = _find_layer_parent(parts["backbone"], total)
        kept = nn.ModuleList(list(parts["layers"])[start_layer:end_layer])
        if parent_attr is not None:
            parent, attr = parent_attr
            setattr(parent, attr, kept)

        state = {}
        for filename, tensor_names in plan.items():
            path = hf_hub_download(
                model_id, filename, token=hf_token, cache_dir=cache_dir
            )
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in tensor_names:
                    key = _remap_key(name, start_layer)
                    state[key] = f.get_tensor(name)

        model.load_state_dict(state, strict=False, assign=True)

        head = model.get_output_embeddings()
        src = model.get_input_embeddings()
        if (
            tied
            and head is not None
            and src is not None
            and head.weight.device.type == "meta"
            and src.weight.device.type != "meta"
        ):
            head.weight = src.weight

        _detach_unused_modules(model, parts, start_layer, end_layer, total)
        _rebuild_meta_buffers(model, cfg)

        leftover_params = [
            n for n, p in model.named_parameters() if p.device.type == "meta"
        ]
        leftover_buffers = [
            n for n, b in model.named_buffers() if b.device.type == "meta"
        ]
        if leftover_params or leftover_buffers:
            raise RuntimeError(
                f"unmaterialised params {leftover_params[:5]}, "
                f"buffers {leftover_buffers[:5]}"
            )

        self.kind = parts["kind"]
        self.layers = kept
        self.rotary = parts["rotary"]
        if start_layer == 0:
            self.embed_tokens = parts["embed"]
            self.pos_embed = parts["pos_embed"]
        if end_layer == total:
            self.norm = parts["norm"]
            self.lm_head = parts["lm_head"]

    def _load_full(self, model_id, start_layer, end_layer, total, hf_token, cache_dir):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            cache_dir=cache_dir,
            dtype="auto",
        )
        model.eval()
        parts = _resolve_backbone(model)
        self.kind = parts["kind"]
        self.layers = parts["layers"][start_layer:end_layer]
        self.rotary = parts["rotary"]
        if start_layer == 0:
            self.embed_tokens = parts["embed"]
            self.pos_embed = parts["pos_embed"]
        if end_layer == total:
            self.norm = parts["norm"]
            self.lm_head = parts["lm_head"]

    def is_first(self) -> bool:
        return self.start_layer == 0

    def is_last(self) -> bool:
        return self.end_layer == self.total_layers