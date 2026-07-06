import numpy as np
import torch

from diffuse_worker import data_pb2, data_pb2_grpc
from diffuse_worker.server import proto_to_tensor


def _tensor_msg(arr: np.ndarray) -> data_pb2.Tensor:
    return data_pb2.Tensor(shape=list(arr.shape), dtype=str(arr.dtype), data=arr.tobytes())


class PipelineStage:
    def __init__(self, channel, model_id: str, start_layer: int, end_layer: int):
        self.stub = data_pb2_grpc.InferenceWorkerStub(channel)
        self.model_id = model_id
        self.start_layer = start_layer
        self.end_layer = end_layer

    def load(self, hf_token: str | None = None) -> None:
        resp = self.stub.LoadSlice(
            data_pb2.LoadSliceRequest(
                model_id=self.model_id,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                hf_token=hf_token or "",
            )
        )
        if not resp.ok:
            raise RuntimeError(f"load failed: {resp.error}")

    def run(self, tensor: torch.Tensor, session_id: str) -> torch.Tensor:
        resp = self.stub.RunSlice(
            data_pb2.SliceRequest(
                model_id=self.model_id,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                session_id=session_id,
                position=0,
                activations=_tensor_msg(tensor.numpy()),
            )
        )
        if not resp.ok:
            raise RuntimeError(f"run failed: {resp.error}")
        return proto_to_tensor(resp.activations)


class Pipeline:
    def __init__(self, stages: list[PipelineStage], tokenizer):
        self.stages = stages
        self.tokenizer = tokenizer

    def _forward(self, input_ids: torch.Tensor, session_id: str) -> torch.Tensor:
        tensor = input_ids
        for stage in self.stages:
            tensor = stage.run(tensor, session_id)
        return tensor

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        session_id: str = "gen",
        greedy: bool = True,
    ) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        eos = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            logits = self._forward(input_ids.to(torch.int64), session_id)
            next_logits = logits[:, -1, :]
            if greedy:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos is not None and next_id.item() == eos:
                break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)