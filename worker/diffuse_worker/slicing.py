import re
from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub import get_safetensors_metadata, hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_LAYER_RE = re.compile(r"\.(?:layers|h|blocks|block)\.(\d+)\.")

# A multimodal checkpoint carries an encoder tower (vision, audio) and a
# projector alongside the language stack. The tower numbers its own blocks, so
# it has to be recognised before the layer regex ever runs, or those blocks
# would be mistaken for decoder layers and sliced apart.
_TOWER_MARKERS = (
    "vision_model",
    "vision_tower",
    "visual",
    "audio_tower",
    "audio_model",
    "image_encoder",
    "connector",
    "multi_modal_projector",
    "mm_projector",
    "modality_projection",
    "merger",
    "perceiver",
)

_TEXT_MODULE_NAMES = ("language_model", "text_model", "model", "transformer")


# Where a checkpoint keeps the stack Diffuse slices. Multimodal models put it
# under `text_config`, MusicGen under `decoder`, others elsewhere; the names are
# tried in order and then any sub-config that declares a depth, so a family
# nobody has named here is still found by its shape.
_DECODER_SECTIONS = (
    "text_config",
    "decoder_config",
    "decoder",
    "talker_config",
    "language_model_config",
    "llm_config",
)

_DEPTH_KEYS = ("num_hidden_layers", "n_layer")


def _depth_of(section) -> int | None:
    for key in _DEPTH_KEYS:
        value = getattr(section, key, None) if not isinstance(section, dict) else section.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def text_config(cfg):
    """The sub-config describing the decoder stack Diffuse slices.

    Reading `num_hidden_layers` off the top level fails on every checkpoint that
    wraps its language model: multimodal ones expose nothing at the root, and an
    encoder-decoder keeps two stacks side by side."""
    for name in _DECODER_SECTIONS:
        section = getattr(cfg, name, None)
        if section is not None and _depth_of(section) is not None:
            return section
    if _depth_of(cfg) is not None:
        return cfg
    for name in dir(cfg):
        if name.startswith("_") or "encoder" in name:
            continue
        section = getattr(cfg, name, None)
        if hasattr(section, "to_dict") and _depth_of(section) is not None:
            return section
    return cfg


def layer_count(cfg):
    return _depth_of(text_config(cfg))


def is_multimodal(cfg) -> bool:
    return any(
        getattr(cfg, attr, None) is not None
        for attr in ("vision_config", "audio_config", "video_config")
    )


def model_class_for(cfg):
    """The class that can build this checkpoint.

    AutoModelForCausalLM covers text-only models but rejects every multimodal
    config. Rather than maintain a mapping of auto classes per modality, take
    the class the checkpoint names for itself, which works for image, audio and
    video alike."""
    import transformers

    for arch in getattr(cfg, "architectures", None) or []:
        klass = getattr(transformers, arch, None)
        if klass is not None:
            return klass
    for fallback in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForSpeechSeq2Seq",
    ):
        klass = getattr(transformers, fallback, None)
        if klass is None:
            continue
        try:
            if type(cfg) in klass._model_mapping:
                return klass
        except Exception:
            continue
    return AutoModelForCausalLM


def _build_empty(klass, cfg):
    """An unmaterialised model, however the class prefers to be built.

    `from_config` only exists on the Auto classes; the concrete multimodal
    classes are constructed directly. Without this the partial loader falls
    over and every node downloads the whole checkpoint."""
    if _sdpa_available():
        try:
            cfg._attn_implementation = "sdpa"
        except Exception:
            pass
    if hasattr(klass, "from_config"):
        return klass.from_config(cfg)
    return klass(cfg)


def _is_tower_tensor(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _TOWER_MARKERS)


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


def _find_text_module(model, expected: int):
    """The module that owns the decoder stack.

    Scoping matters on multimodal models: the final norm and the rotary
    embedding must come from the language model, not from the vision tower,
    which carries modules of the same names and shapes."""
    for name in _TEXT_MODULE_NAMES:
        child = getattr(model, name, None)
        if child is None or _is_tower_tensor(name):
            continue
        if _find_layer_list(child, expected) is not None:
            deeper = _find_text_module(child, expected)
            return deeper if deeper is not None else child
    for name, child in model.named_children():
        if _is_tower_tensor(name):
            continue
        if _find_layer_list(child, expected) is not None:
            deeper = _find_text_module(child, expected)
            return deeper if deeper is not None else child
    return None


def _find_tower(model):
    """Encoder tower and projector, whatever the modality."""
    modules = {}
    for name, child in model.named_children():
        if _is_tower_tensor(name):
            modules[name] = child
    inner = getattr(model, "model", None)
    if inner is not None:
        for name, child in inner.named_children():
            if _is_tower_tensor(name):
                modules.setdefault(name, child)
    return modules


def _resolve_backbone(model):
    cfg = model.config
    total = layer_count(cfg)
    if total is None:
        raise ValueError("cannot determine layer count from config")

    scoped = _find_text_module(model, total)
    backbone = scoped or getattr(model, "base_model", None) or model
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
        "tower": _find_tower(model),
    }


def _is_embedding_tensor(name: str) -> bool:
    return "embed" in name or "wte" in name or "wpe" in name


def _tensor_is_needed(name, start_layer, end_layer, total, tied=False):
    # The tower rides with the slice holding the embeddings, since that is where
    # media becomes hidden states. Checked before the layer regex, which its own
    # numbered blocks would otherwise match.
    if _is_tower_tensor(name):
        return start_layer == 0
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
    if _is_tower_tensor(name):
        return name
    m = _LAYER_RE.search(name)
    if not m:
        return name
    idx = int(m.group(1))
    return name[: m.start(1)] + str(idx - start_layer) + name[m.end(1) :]

def _align_state_keys(model, state):
    """Map checkpoint names onto the names this model actually uses.

    A checkpoint does not have to spell its tensors the way the class does:
    Voxtral ships `audio_tower.conv1.weight` for a module the model calls
    `model.audio_tower.conv1.weight`. from_pretrained reconciles that; loading
    a state dict by hand does not, and the mismatch shows up as parameters left
    unmaterialised. Matching on the longest unique suffix bridges the two
    without hardcoding any one checkpoint's habits."""
    expected = [n for n, _ in model.named_parameters()]
    expected += [n for n, _ in model.named_buffers()]
    known = set(expected)

    suffixes = {}
    for name in expected:
        parts = name.split(".")
        for i in range(len(parts)):
            suffixes.setdefault(".".join(parts[i:]), []).append(name)

    aligned = {}
    for key, tensor in state.items():
        if key in known:
            aligned[key] = tensor
            continue
        parts = key.split(".")
        from_tower = _is_tower_tensor(key)
        for i in range(len(parts)):
            candidates = suffixes.get(".".join(parts[i:]))
            if not candidates:
                continue
            # An encoder tower carries layers named exactly like the decoder's,
            # down to `layers.0.self_attn.q_proj.weight`, so a suffix alone is
            # ambiguous. Keeping only candidates on the same side of that line
            # makes the match unique again.
            candidates = [c for c in candidates if _is_tower_tensor(c) == from_tower]
            if len(candidates) == 1:
                aligned[candidates[0]] = tensor
                break
        else:
            aligned[key] = tensor
    return aligned


def _detach_unused_modules(model, parts, start_layer, end_layer, total):
    unused = []
    if start_layer != 0:
        unused.append(parts.get("embed"))
        unused.append(parts.get("pos_embed"))
        # A middle or tail slice never sees media, so the tower is dead weight.
        unused.extend((parts.get("tower") or {}).values())
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
        # On a multimodal checkpoint the rotary buffers belong to the language
        # model, so they have to be rebuilt from the text sub-config; the root
        # config describes the wrapper and does not carry the right fields.
        # A buffer can belong to the language model or to an encoder tower, and
        # each is described by its own sub-config. Rebuilding a vision rotary
        # from the text config, or from the root one, silently fails and sends
        # the loader back to downloading the whole checkpoint.
        candidates = [text_config(cfg)]
        for sub in ("vision_config", "audio_config", "video_config"):
            inner_cfg = getattr(cfg, sub, None)
            if inner_cfg is not None:
                candidates.append(inner_cfg)
        candidates.append(cfg)
        attempts = []
        for inner in candidates:
            attempts.append(lambda inner=inner: type(module)(config=inner))
            attempts.append(lambda inner=inner: type(module)(inner))
        for attempt in attempts:
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


def _build_kwargs(hf_token, cache_dir):
    kwargs = {"token": hf_token, "cache_dir": cache_dir}
    if _sdpa_available():
        kwargs["attn_implementation"] = "sdpa"
    return kwargs


def _sdpa_available() -> bool:
    return hasattr(torch.nn.functional, "scaled_dot_product_attention")


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
        self.tower = {}
        self.processor = None
        self.multimodal = False
        self.model = None

    def load(
        self,
        model_id: str,
        start_layer: int,
        end_layer: int,
        hf_token: str | None = None,
        cache_dir: str | None = None,
    ) -> LoadedSlice:
        cfg = AutoConfig.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir)
        total = layer_count(cfg)
        if total is None:
            raise ValueError(f"cannot determine the layer count of {model_id}")
        self.multimodal = is_multimodal(cfg)

        if start_layer == 0 and end_layer == 0:
            self.model_id = model_id
            self.start_layer = 0
            self.end_layer = 0
            self.total_layers = total
            self._load_frontend(model_id, hf_token, cache_dir)
            # A text client needs nothing but the tokenizer. A multimodal one
            # has to hold the embeddings and the encoder tower, because that is
            # what turns a picture or a recording into activations, and doing it
            # anywhere else would mean handing the raw media to a stranger.
            if self.multimodal:
                tied = bool(getattr(cfg, "tie_word_embeddings", False))
                try:
                    plan = _plan_download(model_id, 0, 0, total, hf_token, tied)
                    if plan:
                        self._load_partial(
                            model_id, cfg, plan, 0, 0, total, tied, hf_token, cache_dir
                        )
                    else:
                        raise RuntimeError("no partial plan available")
                except Exception as exc:
                    # Some towers hold buffers that cannot be rebuilt from a
                    # config, so the slice-by-slice loader cannot materialise
                    # them. Falling back costs the whole checkpoint but leaves
                    # the client able to embed media, which is the point.
                    print(f"partial media frontend unavailable ({exc}), loading in full")
                    try:
                        self._load_full(model_id, 0, 0, total, hf_token, cache_dir)
                    except Exception as inner:
                        print(f"could not load the media frontend ({inner}), text only")
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
            self._load_frontend(model_id, hf_token, cache_dir)

        self.model_id = model_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.total_layers = total
        return LoadedSlice(model_id, start_layer, end_layer, total)

    def _load_frontend(self, model_id, hf_token, cache_dir):
        """Tokenizer, plus the processor when the model takes media.

        The processor is what turns an image, an audio clip or a video into the
        placeholder tokens and pixel or feature tensors the tower expects."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, cache_dir=cache_dir
        )
        if not self.multimodal:
            return
        try:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                model_id, token=hf_token, cache_dir=cache_dir
            )
        except Exception as exc:
            print(f"no processor available for {model_id} ({exc}), text only")

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
        klass = model_class_for(cfg)
        with torch.device("meta"):
            model = _build_empty(klass, cfg)
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

        model.load_state_dict(_align_state_keys(model, state), strict=False, assign=True)

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

        model.to(self.device)

        self.kind = parts["kind"]
        self.layers = kept
        self.rotary = parts["rotary"]
        if start_layer == 0:
            self.embed_tokens = parts["embed"]
            self.pos_embed = parts["pos_embed"]
            self.tower = parts.get("tower") or {}
            self.model = model
        if end_layer == total:
            self.norm = parts["norm"]
            self.lm_head = parts["lm_head"]

    def _load_full(self, model_id, start_layer, end_layer, total, hf_token, cache_dir):
        cfg = AutoConfig.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir)
        model = model_class_for(cfg).from_pretrained(
            model_id,
            dtype="auto",
            **_build_kwargs(hf_token, cache_dir),
        )
        model.eval()
        model.to(self.device)
        parts = _resolve_backbone(model)
        self.kind = parts["kind"]
        self.layers = parts["layers"][start_layer:end_layer]
        self.rotary = parts["rotary"]
        if start_layer == 0:
            self.embed_tokens = parts["embed"]
            self.pos_embed = parts["pos_embed"]
            self.tower = parts.get("tower") or {}
            self.model = model
        if end_layer == total:
            self.norm = parts["norm"]
            self.lm_head = parts["lm_head"]

    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def is_first(self) -> bool:
        return self.start_layer == 0

    def is_last(self) -> bool:
        return self.end_layer == self.total_layers