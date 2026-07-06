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