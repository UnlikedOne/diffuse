import numpy as np
import pytest
import torch
from PIL import Image

from diffuse_worker.catalog import describe
from diffuse_worker.inference import MediaEmbedder, SliceRunner
from diffuse_worker.slicing import ModelSlice, is_multimodal, layer_count
from transformers import AutoConfig

VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"


def test_layer_count_reads_the_text_sub_config():
    cfg = AutoConfig.from_pretrained(VLM)
    assert not hasattr(cfg, "num_hidden_layers")
    assert layer_count(cfg) == 30
    assert is_multimodal(cfg)


def test_tower_rides_with_the_first_slice_only():
    first = ModelSlice(device="cpu")
    first.load(VLM, 0, 3)
    assert first.tower, "the slice holding the embeddings must carry the encoder tower"
    assert first.embed_tokens is not None

    middle = ModelSlice(device="cpu")
    middle.load(VLM, 10, 13)
    assert middle.tower == {}, "a middle slice must not pay for the tower"
    assert middle.embed_tokens is None
    assert middle.lm_head is None


def test_decoder_layers_are_sliced_not_the_tower():
    sliced = ModelSlice(device="cpu")
    sliced.load(VLM, 4, 9)
    assert len(sliced.layers) == 5, "the vision blocks must not be mistaken for decoder layers"


def test_image_becomes_hidden_states_in_the_decoder_space():
    slice_ = ModelSlice(device="cpu")
    slice_.load(VLM, 0, 2)
    if slice_.processor is None:
        pytest.skip("processor unavailable, multimodal extra not installed")

    image = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hi"}]}]
    text = slice_.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = slice_.processor(text=text, images=[image], return_tensors="pt")

    embeds = MediaEmbedder(slice_).embed(inputs)
    hidden = slice_.embed_tokens.embedding_dim
    assert embeds.shape[0] == 1
    assert embeds.shape[1] == inputs["input_ids"].shape[1]
    assert embeds.shape[2] == hidden, "media features must land in the decoder's hidden space"


def test_embedded_media_flows_through_a_slice_as_activations():
    slice_ = ModelSlice(device="cpu")
    slice_.load(VLM, 0, 2)
    if slice_.processor is None:
        pytest.skip("processor unavailable, multimodal extra not installed")
    runner = SliceRunner(slice_)

    image = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hi"}]}]
    text = slice_.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = slice_.processor(text=text, images=[image], return_tensors="pt")
    embeds = MediaEmbedder(slice_).embed(inputs)

    out = runner.run(embeds, is_input_ids=False, session_id="mm", use_cache=True)
    assert out.shape == embeds.shape
    assert runner.cached_length("mm") == embeds.shape[1]


def test_capability_is_read_off_the_config_not_a_list():
    """No family is hardcoded: the same rules judge a checkpoint published today."""
    vision = describe(
        {
            "architectures": ["Idefics3ForConditionalGeneration"],
            "model_type": "idefics3",
            "text_config": {"num_hidden_layers": 30, "hidden_size": 576},
            "vision_config": {"num_hidden_layers": 12},
        }
    )
    assert vision["support"] == "ready"
    assert vision["inputs"] == ["text", "image"]
    assert vision["layers"] == 30

    # A model may reuse one tower for several modalities and say so only with a
    # placeholder token id.
    both = describe(
        {
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "text_config": {"num_hidden_layers": 28},
            "vision_config": {},
            "video_token_id": 151656,
        }
    )
    assert both["inputs"] == ["text", "image", "video"]

    encoder_decoder = describe(
        {
            "architectures": ["WhisperForConditionalGeneration"],
            "num_hidden_layers": 12,
            "is_encoder_decoder": True,
        }
    )
    assert encoder_decoder["support"] == "unsupported"
    assert "encoder" in encoder_decoder["note"]

    recurrent = describe(
        {"architectures": ["MambaForCausalLM"], "num_hidden_layers": 24, "state_size": 16}
    )
    assert recurrent["support"] == "unsupported"
    assert "recurrent" in recurrent["note"]

    not_generative = describe(
        {"architectures": ["BertForMaskedLM"], "num_hidden_layers": 12}
    )
    assert not_generative["support"] == "unsupported"

    # An architecture nobody has seen before is judged on its shape alone.
    unseen = describe(
        {"architectures": ["BrandNewForCausalLM"], "num_hidden_layers": 42}
    )
    assert unseen["support"] == "ready"
    assert unseen["layers"] == 42


class _FakeModel:
    """Stands in for a built model: only its parameter names matter here."""

    def __init__(self, names):
        self._names = names

    def named_parameters(self):
        return [(n, None) for n in self._names]

    def named_buffers(self):
        return []


def test_checkpoint_keys_are_aligned_onto_the_model_names():
    from diffuse_worker.slicing import _align_state_keys

    model = _FakeModel(
        [
            "model.audio_tower.conv1.weight",
            "model.audio_tower.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.embed_tokens.weight",
        ]
    )
    # How Voxtral actually spells them, which is not how the class does.
    state = {
        "audio_tower.conv1.weight": 1,
        "audio_tower.layers.0.self_attn.q_proj.weight": 2,
        "language_model.model.layers.0.self_attn.q_proj.weight": 3,
        "language_model.model.embed_tokens.weight": 4,
    }
    aligned = _align_state_keys(model, state)

    assert aligned["model.audio_tower.conv1.weight"] == 1
    # The tower names its blocks exactly like the decoder does; a tower tensor
    # must not land on a decoder layer, nor the other way round.
    assert aligned["model.audio_tower.layers.0.self_attn.q_proj.weight"] == 2
    assert aligned["model.language_model.layers.0.self_attn.q_proj.weight"] == 3
    assert aligned["model.language_model.embed_tokens.weight"] == 4


def test_alignment_leaves_already_correct_names_alone():
    from diffuse_worker.slicing import _align_state_keys

    model = _FakeModel(["model.layers.0.mlp.up_proj.weight"])
    state = {"model.layers.0.mlp.up_proj.weight": 7}
    assert _align_state_keys(model, state) == state
