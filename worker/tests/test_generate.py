from concurrent import futures

import grpc
import pytest

from diffuse_worker import data_pb2_grpc
from diffuse_worker.config import WorkerConfig
from diffuse_worker.pipeline import Pipeline, PipelineStage
from diffuse_worker.server import InferenceWorkerServicer
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


def test_generate_across_two_nodes(two_workers):
    c1, c2 = two_workers
    total = _total(TOY_MODEL)
    mid = total // 2

    stage1 = PipelineStage(c1, TOY_MODEL, 0, mid)
    stage2 = PipelineStage(c2, TOY_MODEL, mid, total)
    stage1.load()
    stage2.load()

    ref = ModelSlice()
    ref.load(TOY_MODEL, 0, total)

    pipe = Pipeline([stage1, stage2], ref.tokenizer)
    out = pipe.generate("A cat sat", max_new_tokens=10)

    assert isinstance(out, str)
    assert out.startswith("A cat sat")
    assert len(out) > len("A cat sat")
    print(f"\n[Diffuse generated]: {out!r}")