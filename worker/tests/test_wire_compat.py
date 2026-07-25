import torch

from diffuse_worker.server import proto_to_tensor, tensor_to_proto


def test_legacy_requester_gets_float32():
    hidden = torch.randn(1, 3, 8, dtype=torch.bfloat16)
    msg = tensor_to_proto(hidden, accepts_bf16=False)
    assert msg.dtype == "float32"
    assert len(msg.data) == 3 * 8 * 4
    assert proto_to_tensor(msg).dtype is torch.float32


def test_current_requester_gets_bfloat16():
    hidden = torch.randn(1, 3, 8, dtype=torch.bfloat16)
    msg = tensor_to_proto(hidden, accepts_bf16=True)
    assert msg.dtype == "bfloat16"
    assert len(msg.data) == 3 * 8 * 2
    assert proto_to_tensor(msg).dtype is torch.bfloat16


def test_default_is_the_legacy_format():
    hidden = torch.randn(1, 2, 4, dtype=torch.bfloat16)
    assert tensor_to_proto(hidden).dtype == "float32"


def test_float32_is_never_downgraded():
    hidden = torch.randn(1, 2, 4, dtype=torch.float32)
    assert tensor_to_proto(hidden, accepts_bf16=True).dtype == "float32"
