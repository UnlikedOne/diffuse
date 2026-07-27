import json

import pytest
import torch

from diffuse_worker.diffusion import DiffusionStack, block_holder, rebuild_arguments, flatten_arguments

diffusers = pytest.importorskip("diffusers")


def build_checkpoint(path):
    from diffusers import WanTransformer3DModel

    torch.manual_seed(5)
    model = WanTransformer3DModel(
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
    ).eval()
    model.save_pretrained(path / "transformer")
    (path / "model_index.json").write_text(
        json.dumps({"_class_name": "WanPipeline", "transformer": ["diffusers", "WanTransformer3DModel"]})
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


def test_a_whole_sequence_through_slices_matches_the_unsliced_stack(tmp_path):
    model = build_checkpoint(tmp_path)
    torch.manual_seed(9)
    hidden, arguments = block_inputs(model)

    with torch.inference_mode():
        expected = hidden
        for block in model.blocks:
            expected = block(expected, *arguments)

    front = DiffusionStack()
    front.load(str(tmp_path), 0, 2)
    back = DiffusionStack()
    back.load(str(tmp_path), 2, 4)

    sequence = hidden.shape[1]
    produced = front.run_patch("s", hidden, 0, sequence, arguments)
    produced = back.run_patch("s", produced, 0, sequence, arguments)

    assert torch.equal(produced, expected)


def test_patches_stay_close_to_the_whole_sequence(tmp_path):
    model = build_checkpoint(tmp_path)
    torch.manual_seed(9)
    hidden, arguments = block_inputs(model)

    with torch.inference_mode():
        expected = hidden
        for block in model.blocks:
            expected = block(expected, *arguments)

    stack = DiffusionStack()
    stack.load(str(tmp_path), 0, 4)
    sequence = hidden.shape[1]

    # A first pass over the whole sequence fills the stale buffers, the way the
    # first denoising step does before any patch is ever cut.
    stack.run_patch("s", hidden, 0, sequence, arguments)

    size = sequence // 4
    produced = torch.zeros_like(expected)
    for index in range(4):
        start = index * size
        stop = start + size
        produced[:, start:stop] = stack.run_patch(
            "s", hidden[:, start:stop], start, sequence, arguments
        )

    assert not torch.equal(produced, expected)
    assert (produced - expected).abs().max() < 0.5 * expected.abs().max()


def test_arguments_survive_the_wire_layout():
    values = [torch.randn(2, 3), torch.randn(4), torch.randn(1, 5)]
    original = (values[0], (values[1], values[2]), None)

    flat, layout = flatten_arguments(original)
    rebuilt = rebuild_arguments(flat, layout)

    assert layout == "T,(TT),N"
    assert torch.equal(rebuilt[0], values[0])
    assert torch.equal(rebuilt[1][0], values[1])
    assert torch.equal(rebuilt[1][1], values[2])
    assert rebuilt[2] is None
