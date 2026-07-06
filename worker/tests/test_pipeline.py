from concurrent import futures

import grpc
import numpy as np
import pytest
import torch

from diffuse_worker import data_pb2, data_pb2_grpc
from diffuse_worker.config import WorkerConfig
from diffuse_worker.inference import SliceRunner
from diffuse_worker.server import InferenceWorkerServicer, proto_to_tensor
from diffuse_worker.slicing import ModelSlice

TOY_MODEL = "sshleifer/tiny-gpt2"


def _total(model_id: str) -> int:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    return getattr(cfg, "num_hidden_layers", None) or cfg.n_layer


def _start_worker():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    data_pb2_grpc.add_InferenceWorkerServicer_to_server(
        InferenceWorkerServicer(WorkerConfig()), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


def _tensor_msg(arr):
    return data_pb2.Tensor(shape=list(arr.shape), dtype=str(arr.dtype), data=arr.tobytes())


@pytest.fixture
def two_workers():
    s1, p1 = _start_worker()
    s2, p2 = _start_worker()
    c1 = grpc.insecure_channel(f"127.0.0.1:{p1}")
    c2 = grpc.insecure_channel(f"127.0.0.1:{p2}")
    yield c1, c2
    c1.close()
    c2.close()
    s1.stop(None)
    s2.stop(None)


def test_pipeline_two_nodes(two_workers):
    c1, c2 = two_workers
    total = _total(TOY_MODEL)
    mid = total // 2

    stub1 = data_pb2_grpc.InferenceWorkerStub(c1)
    stub2 = data_pb2_grpc.InferenceWorkerStub(c2)

    assert stub1.LoadSlice(
        data_pb2.LoadSliceRequest(model_id=TOY_MODEL, start_layer=0, end_layer=mid)
    ).ok
    assert stub2.LoadSlice(
        data_pb2.LoadSliceRequest(model_id=TOY_MODEL, start_layer=mid, end_layer=total)
    ).ok

    ref_slice = ModelSlice()
    ref_slice.load(TOY_MODEL, 0, total)
    ref_runner = SliceRunner(ref_slice)
    ids = ref_slice.tokenizer("A cat sat", return_tensors="pt")["input_ids"]
    ref = ref_runner.run(ids, is_input_ids=True)

    ids_np = ids.numpy().astype(np.int64)
    r1 = stub1.RunSlice(
        data_pb2.SliceRequest(
            model_id=TOY_MODEL,
            start_layer=0,
            end_layer=mid,
            session_id="pipe",
            position=0,
            activations=_tensor_msg(ids_np),
        )
    )
    assert r1.ok, r1.error

    hidden = proto_to_tensor(r1.activations)
    r2 = stub2.RunSlice(
        data_pb2.SliceRequest(
            model_id=TOY_MODEL,
            start_layer=mid,
            end_layer=total,
            session_id="pipe",
            position=0,
            activations=_tensor_msg(hidden.numpy()),
        )
    )
    assert r2.ok, r2.error

    out = proto_to_tensor(r2.activations)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-4)