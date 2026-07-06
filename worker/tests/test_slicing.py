import pytest
import torch

from diffuse_worker.slicing import ModelSlice

TOY_MODEL = "sshleifer/tiny-gpt2"


def test_load_full_range():
    s = ModelSlice()
    info = s.load(TOY_MODEL, 0, 2)
    assert info.total_layers >= 2
    assert s.is_first()
    assert s.embed_tokens is not None


def test_load_first_slice_has_tokenizer():
    s = ModelSlice()
    s.load(TOY_MODEL, 0, 1)
    assert s.tokenizer is not None
    assert s.is_first()
    assert not s.is_last()