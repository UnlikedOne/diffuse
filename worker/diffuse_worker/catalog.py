import json
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download


_GENERATIVE_SUFFIXES = (
    "ForCausalLM",
    "ForConditionalGeneration",
    "ForImageTextToText",
    "ForSpeechSeq2Seq",
    "LMHeadModel",
)

_MEDIA_SECTIONS = {
    "vision_config": "image",
    "audio_config": "audio",
    "video_config": "video",
}

_RECURRENT_KEYS = ("state_size", "conv_kernel", "time_step_rank", "time_mix_extra_dim")


_DECODER_SECTIONS = (
    "text_config",
    "decoder_config",
    "decoder",
    "talker_config",
    "language_model_config",
    "llm_config",
)


def _depth_of(section):
    if not isinstance(section, dict):
        return None
    for key in ("num_hidden_layers", "n_layer"):
        value = section.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _text_section(config: dict) -> dict:
    for name in _DECODER_SECTIONS:
        section = config.get(name)
        if _depth_of(section) is not None:
            return section
    if _depth_of(config) is not None:
        return config
    for name, section in config.items():
        if "encoder" in name:
            continue
        if _depth_of(section) is not None:
            return section
    return config


def _layer_count(config: dict):
    return _depth_of(_text_section(config))


def _architecture(config: dict) -> str:
    archs = config.get("architectures") or []
    return archs[0] if archs else ""


def describe(config: dict) -> dict:
    """Read a checkpoint's own config and say what Diffuse can do with it."""
    arch = _architecture(config)
    layers = _layer_count(config)
    inner = _text_section(config)

    inputs = ["text"]
    if config.get("num_mel_bins") or config.get("input_feat_per_channel"):
        inputs = ["audio"]
    for section, modality in _MEDIA_SECTIONS.items():
        if isinstance(config.get(section), dict) and modality not in inputs:
            inputs.append(modality)
    for modality in ("image", "audio", "video"):
        marker = f"{modality}_token_id"
        if (marker in config or marker in inner) and modality not in inputs:
            inputs.append(modality)

    generative = arch.endswith(_GENERATIVE_SUFFIXES)
    recurrent = any(key in inner for key in _RECURRENT_KEYS)
    encoder_decoder = bool(config.get("is_encoder_decoder"))

    if layers is None:
        support, note = "unsupported", "no layer stack in its config to split"
    elif not generative:
        support, note = "unsupported", f"{arch or 'this architecture'} does not generate"
    elif encoder_decoder:
        support, note = (
            "ready",
            "encoder-decoder: its encoder runs on your machine, its output travels",
        )
    elif recurrent:
        support, note = (
            "unsupported",
            "recurrent state, which one slice cannot hand to the next",
        )
    elif len(inputs) > 1:
        support, note = "ready", "the encoder tower rides with the first slice"
    else:
        support, note = "ready", "a plain decoder stack, splits cleanly"

    return {
        "architecture": arch,
        "model_type": config.get("model_type") or "",
        "layers": layers or 0,
        "hidden_size": inner.get("hidden_size") or 0,
        "vocab_size": inner.get("vocab_size") or 0,
        "inputs": inputs,
        "outputs": ["text"],
        "support": support,
        "note": note,
    }


def _fetch_config(model_id: str, token: str | None):
    try:
        path = hf_hub_download(model_id, "config.json", token=token)
        with open(path) as handle:
            return json.load(handle)
    except Exception:
        return None


def search(
    query: str = "",
    limit: int = 40,
    hf_token: str | None = None,
    supported_only: bool = True,
) -> list[dict]:
    api = HfApi(token=hf_token)
    listed = api.list_models(
        search=query or None,
        filter="text-generation",
        sort="downloads",
        limit=max(limit * 3, limit + 20),
        expand=["config", "safetensors", "downloads", "likes", "gated"],
    )

    skip = ("-gguf", "-awq", "-gptq", "-mlx", "-onnx", "embedding", "reranker")
    entries = []
    for model in listed:
        if any(marker in model.id.lower() for marker in skip):
            continue
        params = 0
        if model.safetensors is not None and model.safetensors.total:
            params = int(model.safetensors.total)
        entries.append(
            {
                "id": model.id,
                "params": params,
                "downloads": int(model.downloads or 0),
                "likes": int(model.likes or 0),
                "gated": bool(model.gated),
            }
        )
        if len(entries) >= limit * 2:
            break

    with ThreadPoolExecutor(max_workers=16) as pool:
        configs = list(pool.map(lambda e: _fetch_config(e["id"], hf_token), entries))

    cards = []
    for entry, config in zip(entries, configs):
        if config is None:
            entry.update(
                architecture="",
                model_type="",
                layers=0,
                hidden_size=0,
                vocab_size=0,
                inputs=["text"],
                outputs=["text"],
                support="unknown",
                note="config unreachable, a token may be required",
            )
        else:
            entry.update(describe(config))
        if supported_only and entry["support"] == "unsupported":
            continue
        cards.append(entry)
        if len(cards) >= limit:
            break
    return cards
