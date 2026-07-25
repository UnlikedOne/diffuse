from huggingface_hub import HfApi

MULTIMODAL = {
    "idefics3",
    "smolvlm",
    "llava",
    "llava_next",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen2_audio",
    "gemma3",
    "mllama",
    "internvl",
    "video_llava",
    "phi4_multimodal",
}

VALIDATED = {
    "llama",
    "qwen2",
    "qwen3",
    "mistral",
    "gemma",
    "gemma2",
    "gemma3",
    "phi",
    "phi3",
    "gpt2",
    "gptj",
    "gpt_neox",
    "stablelm",
    "olmo",
    "olmo2",
    "starcoder2",
    "cohere",
    "granite",
    "smollm3",
    "falcon",
    "exaone",
    "minicpm",
    "internlm2",
}

UNSUPPORTED = {
    "mamba",
    "mamba2",
    "rwkv",
    "jamba",
    "recurrent_gemma",
    "t5",
    "mt5",
    "bart",
    "whisper",
    "bert",
    "roberta",
    "clip",
}

UNSUPPORTED_NOTE = {
    "mamba": "state space model, no attention layers to slice",
    "mamba2": "state space model, no attention layers to slice",
    "rwkv": "recurrent architecture, no transformer layers to slice",
    "jamba": "hybrid mamba and attention, slicing is not uniform",
    "recurrent_gemma": "recurrent architecture, no uniform layer stack",
    "t5": "encoder-decoder, Diffuse slices decoder-only stacks",
    "mt5": "encoder-decoder, Diffuse slices decoder-only stacks",
    "bart": "encoder-decoder, Diffuse slices decoder-only stacks",
    "whisper": "speech model, not a causal language model",
    "bert": "encoder only, cannot generate",
    "roberta": "encoder only, cannot generate",
    "clip": "vision-text encoder, cannot generate",
}


def classify(model_type: str, architectures: list[str]) -> tuple[str, str]:
    kind = (model_type or "").lower()
    arch = architectures[0] if architectures else ""

    if kind in MULTIMODAL:
        return "validated", "multimodal, the encoder tower rides with the first slice"
    # Checked before the ForConditionalGeneration heuristic below: encoder
    # decoder stacks carry that same suffix and still cannot be sliced.
    if kind in UNSUPPORTED:
        return "unsupported", UNSUPPORTED_NOTE.get(kind, "architecture cannot be split by layer")
    if arch.endswith(("ForConditionalGeneration", "ForImageTextToText")):
        return "likely", "multimodal, untested here but the decoder stack should slice"
    if kind in VALIDATED:
        return "validated", "runs on Diffuse"
    if arch.endswith("ForCausalLM"):
        return "likely", "decoder-only, should slice but is untested here"
    if arch:
        return "unsupported", f"{arch} is not a causal language model"
    return "likely", "architecture unknown, will be checked when hosting"


def search(
    query: str = "",
    limit: int = 40,
    hf_token: str | None = None,
    supported_only: bool = True,
) -> list[dict]:
    api = HfApi(token=hf_token)
    models = api.list_models(
        search=query or None,
        filter="text-generation",
        sort="downloads",
        limit=max(limit * 3, limit + 20),
        expand=["config", "safetensors", "downloads", "likes", "gated"],
    )

    skip_markers = ("embedding", "reranker", "rerank", "-gguf", "-awq", "-gptq")

    cards = []
    for model in models:
        if any(marker in model.id.lower() for marker in skip_markers):
            continue
        config = model.config or {}
        architectures = config.get("architectures") or []
        model_type = config.get("model_type") or ""
        support, note = classify(model_type, architectures)
        if supported_only and support == "unsupported":
            continue
        params = 0
        if model.safetensors is not None and model.safetensors.total:
            params = int(model.safetensors.total)
        cards.append(
            {
                "id": model.id,
                "architecture": architectures[0] if architectures else "",
                "model_type": model_type,
                "params": params,
                "downloads": int(model.downloads or 0),
                "likes": int(model.likes or 0),
                "gated": bool(model.gated),
                "support": support,
                "note": note,
            }
        )
        if len(cards) >= limit:
            break
    return cards
