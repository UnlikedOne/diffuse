import time
from concurrent import futures

import grpc
import numpy as np
import pytest
import torch

from diffuse_worker import data_pb2, data_pb2_grpc
from diffuse_worker.config import WorkerConfig
from diffuse_worker.server import InferenceWorkerServicer, proto_to_tensor, tensor_to_proto

TOY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture
def worker_channel():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    data_pb2_grpc.add_InferenceWorkerServicer_to_server(
        InferenceWorkerServicer(WorkerConfig()), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    yield channel
    channel.close()
    server.stop(None)


def test_load_and_run_over_grpc(worker_channel):
    stub = data_pb2_grpc.InferenceWorkerStub(worker_channel)

    load = stub.LoadSlice(
        data_pb2.LoadSliceRequest(model_id=TOY_MODEL, start_layer=0, end_layer=2)
    )
    assert load.ok, load.error
    assert load.total_layers == 2

    ids = np.array([[32, 4758, 3332]], dtype=np.int64)
    req = data_pb2.SliceRequest(
        model_id=TOY_MODEL,
        start_layer=0,
        end_layer=2,
        session_id="test-session",
        position=0,
        activations=data_pb2.Tensor(
            shape=list(ids.shape), dtype="int64", data=ids.tobytes()
        ),
    )
    resp = stub.RunSlice(req)
    assert resp.ok, resp.error

    out = proto_to_tensor(resp.activations)
    assert out.shape[0] == 1
    assert out.shape[1] == 3
    assert torch.isfinite(out).all()

class _PartsOnlyTemplate:
    """Templates the way a multimodal checkpoint does: only parts are read.

    Handed a plain string it does not fail, it writes an empty turn, which is
    exactly how SmolVLM's tokenizer silently dropped the user's question."""

    def apply_chat_template(
        self, conversation, tokenize=False, add_generation_prompt=False
    ):
        rendered = []
        for message in conversation:
            content = message["content"]
            parts = content if isinstance(content, list) else []
            said = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
            rendered.append(f"{message['role']}: {said}")
        return "\n".join(rendered)


class _PlainTokenizer:
    def apply_chat_template(
        self, conversation, tokenize=False, add_generation_prompt=False
    ):
        return "\n".join(f"{m['role']}: {m['content']}" for m in conversation)


def test_prompt_survives_a_multimodal_chat_template():
    # Both halves of the checkpoint template parts; only the processor is asked
    # with parts, so asking the tokenizer with a string loses the question.
    servicer = InferenceWorkerServicer(WorkerConfig())
    servicer.slice.processor = _PartsOnlyTemplate()
    servicer.slice.tokenizer = _PartsOnlyTemplate()

    text = servicer._apply_template([{"role": "user", "content": "Name three colours."}])

    assert "Name three colours." in text


def test_a_text_only_checkpoint_still_uses_its_tokenizer():
    servicer = InferenceWorkerServicer(WorkerConfig())
    servicer.slice.processor = None
    servicer.slice.tokenizer = _PlainTokenizer()

    text = servicer._apply_template([{"role": "user", "content": "Name three colours."}])

    assert text == "user: Name three colours."
