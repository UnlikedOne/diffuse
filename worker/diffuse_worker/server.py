import logging
from concurrent import futures

import grpc
import numpy as np
import torch

from diffuse_worker import data_pb2, data_pb2_grpc
from diffuse_worker.config import WorkerConfig
from diffuse_worker.inference import SliceRunner
from diffuse_worker.slicing import ModelSlice

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("diffuse.worker")

MAX_MESSAGE_BYTES = 128 * 1024 * 1024


def tensor_to_proto(t: torch.Tensor) -> data_pb2.Tensor:
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(torch.float32)
    arr = t.detach().cpu().numpy()
    return data_pb2.Tensor(
        shape=list(arr.shape),
        dtype=str(arr.dtype),
        data=arr.tobytes(),
    )

def proto_to_tensor(p: data_pb2.Tensor) -> torch.Tensor:
    np_dtype = np.dtype(p.dtype)
    arr = np.frombuffer(p.data, dtype=np_dtype).reshape(tuple(p.shape))
    return torch.from_numpy(arr.copy())


class InferenceWorkerServicer(data_pb2_grpc.InferenceWorkerServicer):
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.slice = ModelSlice(device=config.device)
        self.runner: SliceRunner | None = None
        self.loaded = False

    def LoadSlice(self, request, context):
        try:
            info = self.slice.load(
                model_id=request.model_id,
                start_layer=request.start_layer,
                end_layer=request.end_layer,
                hf_token=request.hf_token or self.config.hf_token or None,
                cache_dir=self.config.cache_dir,
            )
            self.runner = SliceRunner(self.slice)
            self.loaded = True
            log.info(
                "loaded %s layers %d:%d of %d",
                info.model_id,
                info.start_layer,
                info.end_layer,
                info.total_layers,
            )
            return data_pb2.LoadSliceResponse(ok=True, total_layers=info.total_layers)
        except Exception as exc:
            log.exception("load failed")
            return data_pb2.LoadSliceResponse(ok=False, error=str(exc))

    def RunSlice(self, request, context):
        if not self.loaded or self.runner is None:
            return data_pb2.SliceResponse(
                session_id=request.session_id, ok=False, error="slice not loaded"
            )
        try:
            tensor_in = proto_to_tensor(request.activations)
            is_input_ids = tensor_in.dtype == torch.int64
            out = self.runner.run(
                tensor_in,
                is_input_ids=is_input_ids,
                session_id=request.session_id,
                use_cache=request.use_cache,
            )
            return data_pb2.SliceResponse(
                session_id=request.session_id,
                activations=tensor_to_proto(out),
                ok=True,
            )
        except Exception as exc:
            log.exception("run failed")
            return data_pb2.SliceResponse(
                session_id=request.session_id, ok=False, error=str(exc)
            )

    def Encode(self, request, context):
        if self.slice.tokenizer is None:
            return data_pb2.EncodeResponse(ok=False, error="tokenizer not available on this slice")
        try:
            if len(request.messages) > 0:
                messages = [{"role": m.role, "content": m.content} for m in request.messages]
                text = self.slice.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            elif request.apply_chat_template:
                messages = [{"role": "user", "content": request.text}]
                text = self.slice.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = request.text
            ids = self.slice.tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
            eos = self.slice.tokenizer.eos_token_id
            return data_pb2.EncodeResponse(token_ids=ids, ok=True, eos_token_id=eos if eos is not None else -1)
        except Exception as exc:
            log.exception("encode failed")
            return data_pb2.EncodeResponse(ok=False, error=str(exc))

    def Decode(self, request, context):
        if self.slice.tokenizer is None:
            return data_pb2.DecodeResponse(ok=False, error="tokenizer not available on this slice")
        try:
            text = self.slice.tokenizer.decode(
                list(request.token_ids), skip_special_tokens=request.skip_special_tokens
            )
            return data_pb2.DecodeResponse(text=text, ok=True)
        except Exception as exc:
            log.exception("decode failed")
            return data_pb2.DecodeResponse(ok=False, error=str(exc))

    def Health(self, request, context):
        return data_pb2.HealthResponse(
            ok=True,
            slice_loaded=self.loaded,
            start_layer=self.slice.start_layer if self.loaded else 0,
            end_layer=self.slice.end_layer if self.loaded else 0,
        )
    
    def ClearSession(self, request, context):
        if self.runner is not None:
            self.runner.clear_session(request.session_id)
        return data_pb2.ClearSessionResponse(ok=True)
    
    def ProfileModel(self, request, context):
        try:
            from diffuse_worker.capacity import plan_capacity

            overhead = request.overhead_fraction if request.overhead_fraction > 0 else 0.3
            plan = plan_capacity(
                request.model_id,
                overhead_fraction=overhead,
                load_dtype="bfloat16",
                hf_token=request.hf_token or self.config.hf_token or None,
            )
            return data_pb2.ProfileResponse(
                ok=True,
                total_layers=plan["total_layers"],
                avg_layer_bytes=plan["avg_layer_bytes"],
                non_layer_bytes=plan["non_layer_bytes"],
                available_bytes=plan["available_bytes"],
                max_layers=plan["max_layers"],
                max_layers_if_holding_embedding=plan["max_layers_if_holding_embedding"],
                device=plan["device"],
            )
        except Exception as exc:
            log.exception("profile failed")
            return data_pb2.ProfileResponse(ok=False, error=str(exc))

def serve(config: WorkerConfig | None = None) -> None:
    config = config or WorkerConfig.from_env()
    options = [
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4), options=options
    )
    data_pb2_grpc.add_InferenceWorkerServicer_to_server(
        InferenceWorkerServicer(config), server
    )
    addr = f"{config.host}:{config.port}"
    server.add_insecure_port(addr)
    server.start()
    log.info("worker listening on %s", addr)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()