import torch

from diffuse_worker.inference import SliceRunner
from diffuse_worker.slicing import ModelSlice

TOY_MODEL = "sshleifer/tiny-gpt2"


def test_full_forward_single_slice():
    s = ModelSlice()
    info = s.load(TOY_MODEL, 0, s_total := _total(TOY_MODEL))
    runner = SliceRunner(s)

    input_ids = s.tokenizer("A cat sat", return_tensors="pt")["input_ids"]
    logits = runner.run(input_ids, is_input_ids=True)

    assert logits.shape[0] == input_ids.shape[0]
    assert logits.shape[1] == input_ids.shape[1]
    assert logits.shape[2] == s.lm_head.out_features
    assert torch.isfinite(logits).all()


def test_split_matches_single():
    total = _total(TOY_MODEL)
    mid = total // 2

    whole = ModelSlice()
    whole.load(TOY_MODEL, 0, total)
    wr = SliceRunner(whole)
    input_ids = whole.tokenizer("A cat sat", return_tensors="pt")["input_ids"]
    ref = wr.run(input_ids, is_input_ids=True)

    front = ModelSlice()
    front.load(TOY_MODEL, 0, mid)
    fr = SliceRunner(front)
    back = ModelSlice()
    back.load(TOY_MODEL, mid, total)
    br = SliceRunner(back)

    hidden = fr.run(input_ids, is_input_ids=True)
    out = br.run(hidden, is_input_ids=False)

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-4)


def _total(model_id: str) -> int:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    return getattr(cfg, "num_hidden_layers", None) or cfg.n_layer

def test_a_sinusoidal_position_table_is_found_even_though_it_is_not_an_embedding():
    import torch.nn as nn

    from diffuse_worker.slicing import _find_position_embeddings

    class Sinusoidal(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weights", torch.zeros(8, 4))

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(10, 4)
            self.embed_positions = Sinusoidal()

    backbone = Backbone()
    found = _find_position_embeddings(backbone, backbone.embed_tokens)

    assert found is backbone.embed_positions


def test_positions_are_asked_for_with_the_embeddings_not_the_stream_ids():
    import torch.nn as nn

    from diffuse_worker.inference import SliceRunner

    seen = {}

    class Table(nn.Module):
        def forward(self, x, offset=0):
            seen["shape"] = tuple(x.shape)
            return torch.zeros(x.shape[0], x.shape[1], 4)

    runner = SliceRunner.__new__(SliceRunner)
    runner.slice = type("S", (), {"pos_embed": Table()})()

    hidden = torch.zeros(2, 1, 4)
    ids = torch.zeros(2, 4, 1, dtype=torch.long)
    runner._absolute_positions(hidden, ids, 0)

    assert seen["shape"] == (2, 1, 4)
