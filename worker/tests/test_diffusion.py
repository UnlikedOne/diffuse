import json

import pytest
import torch

from diffuse_worker.diffusion import (
    DiffusionStack,
    block_count,
    block_holders,
    build_plan,
    pack_call,
    stack_keys,
)

diffusers = pytest.importorskip("diffusers")

WAN = dict(
    patch_size=[1, 2, 2],
    num_attention_heads=2,
    attention_head_dim=16,
    in_channels=4,
    out_channels=4,
    text_dim=32,
    freq_dim=32,
    ffn_dim=64,
    num_layers=4,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    rope_max_seq_len=64,
)


def build_checkpoint(path):
    from diffusers import WanTransformer3DModel

    torch.manual_seed(5)
    model = WanTransformer3DModel(**WAN).eval()
    model.save_pretrained(path / "transformer")
    (path / "model_index.json").write_text(
        json.dumps(
            {"_class_name": "WanPipeline", "transformer": ["diffusers", "WanTransformer3DModel"]}
        )
    )
    return model


def block_inputs(model):
    latent = torch.randn(1, 4, 2, 16, 16)
    rotary = model.rope(latent)
    sequence = 2 * 8 * 8
    hidden = torch.randn(1, sequence, 32)
    encoder = torch.randn(1, 6, 32)
    temb = torch.randn(1, 6, 32)
    return hidden, (encoder, temb, rotary)


def wire(model, hidden, arguments):
    plan = build_plan("blocks", (hidden,) + arguments, {}, model.blocks[0])
    carried, tensors = pack_call(plan, (hidden,) + arguments, {})
    return plan, carried, tensors


def test_a_whole_sequence_through_slices_matches_the_unsliced_stack(tmp_path):
    model = build_checkpoint(tmp_path)
    torch.manual_seed(9)
    hidden, arguments = block_inputs(model)

    with torch.inference_mode():
        expected = hidden
        for block in model.blocks:
            expected = block(expected, *arguments)

    plan, carried, tensors = wire(model, hidden, arguments)
    front = DiffusionStack()
    front.load(str(tmp_path), 0, 2)
    back = DiffusionStack()
    back.load(str(tmp_path), 2, 4)

    sequence = hidden.shape[1]
    produced = front.run_patch("s", carried, 0, sequence, tensors, plan)
    produced = back.run_patch("s", produced, 0, sequence, tensors, plan)

    assert torch.equal(produced, expected)


def test_patches_stay_close_to_the_whole_sequence(tmp_path):
    model = build_checkpoint(tmp_path)
    torch.manual_seed(9)
    hidden, arguments = block_inputs(model)

    with torch.inference_mode():
        expected = hidden
        for block in model.blocks:
            expected = block(expected, *arguments)

    plan, carried, tensors = wire(model, hidden, arguments)
    stack = DiffusionStack()
    stack.load(str(tmp_path), 0, 4)
    sequence = hidden.shape[1]

    stack.run_patch("s", carried, 0, sequence, tensors, plan)

    size = sequence // 4
    produced = torch.zeros_like(expected)
    for index in range(4):
        start = index * size
        stop = start + size
        produced[:, start:stop] = stack.run_patch(
            "s", carried[:, start:stop], start, sequence, tensors, plan
        )

    assert not torch.equal(produced, expected)
    assert (produced - expected).abs().max() < 0.5 * expected.abs().max()


def test_the_quick_attention_is_only_kept_when_it_agrees(tmp_path):
    model = build_checkpoint(tmp_path)
    stack = DiffusionStack()
    stack.load(str(tmp_path), 0, 4)
    assert stack.patchers
    assert all(patcher.mode is None for patcher in stack.patchers)

    torch.manual_seed(9)
    hidden, arguments = block_inputs(model)
    plan, carried, tensors = wire(model, hidden, arguments)
    sequence = hidden.shape[1]

    stack.run_patch("s", carried, 0, sequence, tensors, plan)
    assert all(patcher.mode is None for patcher in stack.patchers)

    stack.run_patch("s", carried[:, :16], 0, sequence, tensors, plan)
    assert all(patcher.mode is None for patcher in stack.patchers)

    stack.run_patch("s", carried[:, 16:32], 16, sequence, tensors, plan)
    assert {patcher.mode for patcher in stack.patchers} == {"fast"}


def test_keyword_arguments_and_constants_survive_the_wire():
    class Block(torch.nn.Module):
        def forward(self, hidden_states, encoder_hidden_states, scale=1.0, extras=None):
            return hidden_states + scale, encoder_hidden_states * 2

    hidden = torch.randn(1, 8, 4)
    encoder = torch.randn(1, 3, 4)
    args = (hidden,)
    kwargs = {"encoder_hidden_states": encoder, "scale": 2.5, "extras": {"depth": 3}}

    plan = build_plan("blocks", args, kwargs, Block())
    carried, tensors = pack_call(plan, args, kwargs)

    assert plan["state"] == [8, 3]
    assert plan["returns"] == [0, 1]
    assert plan["single"] is False
    assert plan["kwargs"]["scale"] == {"c": 2.5}
    assert plan["kwargs"]["extras"] == {"d": {"depth": {"c": 3}}}
    assert tensors == []
    assert tuple(carried.shape) == (1, 11, 4)
    assert json.loads(json.dumps(plan)) == plan


def test_a_stack_called_by_keyword_survives_being_split(tmp_path):
    from diffusers import PixArtTransformer2DModel

    settings = dict(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        num_layers=4,
        cross_attention_dim=16,
        caption_channels=16,
        sample_size=8,
        patch_size=2,
        norm_type="ada_norm_single",
    )
    torch.manual_seed(3)
    model = PixArtTransformer2DModel(**settings).eval()
    model.save_pretrained(tmp_path / "transformer")

    torch.manual_seed(4)
    hidden = torch.randn(1, 16, 16)
    kwargs = {
        "attention_mask": None,
        "encoder_hidden_states": torch.randn(1, 5, 16),
        "encoder_attention_mask": None,
        "timestep": torch.randn(1, 96),
        "cross_attention_kwargs": None,
        "class_labels": None,
    }

    with torch.inference_mode():
        expected = hidden
        for block in model.transformer_blocks:
            expected = block(expected, **kwargs)

    plan = build_plan("transformer_blocks", (hidden,), kwargs, model.transformer_blocks[0])
    carried, tensors = pack_call(plan, (hidden,), kwargs)
    assert plan["single"] is True
    assert plan["state"] == [16]
    assert plan["kwargs"]["class_labels"] == {"c": None}

    produced = carried
    for start, end in ((0, 2), (2, 4)):
        stack = DiffusionStack()
        stack.load(str(tmp_path), start, end)
        produced = stack.run_patch("s", produced, 0, hidden.shape[1], tensors, plan)

    assert torch.equal(produced, expected)


def test_a_block_carrying_two_streams_is_never_cut_into_patches():
    from diffusers.models.transformers.transformer_flux import FluxTransformerBlock

    torch.manual_seed(6)
    block = FluxTransformerBlock(dim=16, num_attention_heads=2, attention_head_dim=8).eval()
    hidden = torch.randn(1, 12, 16)
    encoder = torch.randn(1, 5, 16)
    kwargs = {
        "encoder_hidden_states": encoder,
        "temb": torch.randn(1, 16),
        "image_rotary_emb": (torch.randn(17, 8), torch.randn(17, 8)),
    }

    plan = build_plan("transformer_blocks", (hidden,), kwargs, block)
    carried, tensors = pack_call(plan, (hidden,), kwargs)

    assert plan["single"] is False
    assert plan["state"] == [5, 12]
    assert plan["returns"] == [0, 1]
    assert tuple(carried.shape) == (1, 17, 16)


def test_every_block_stack_is_counted_and_named():
    from diffusers import FluxTransformer2DModel

    settings = dict(
        patch_size=1,
        in_channels=4,
        num_layers=2,
        num_single_layers=3,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        pooled_projection_dim=8,
        axes_dims_rope=(2, 2, 4),
    )
    with torch.device("meta"):
        model = FluxTransformer2DModel(**settings)

    assert [name for name, _ in block_holders(model)] == [
        "transformer_blocks",
        "single_transformer_blocks",
    ]
    assert stack_keys(FluxTransformer2DModel, settings) == {
        "transformer_blocks": "num_layers",
        "single_transformer_blocks": "num_single_layers",
    }
    assert block_count({"_class_name": "FluxTransformer2DModel", **settings}) == 5
