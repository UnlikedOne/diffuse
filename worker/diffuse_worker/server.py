import logging
from concurrent import futures

import grpc
import numpy as np
import torch

from diffuse_worker import data_pb2, data_pb2_grpc
from diffuse_worker.batching import BatchScheduler
from diffuse_worker.config import WorkerConfig
from diffuse_worker.inference import MediaEmbedder, SliceRunner
from diffuse_worker.slicing import ModelSlice

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("diffuse.worker")

MAX_MESSAGE_BYTES = 128 * 1024 * 1024


TOPK_DTYPE = "topk_i64_f32"


def tensor_to_proto(t: torch.Tensor, accepts_bf16: bool = False) -> data_pb2.Tensor:
    t = t.detach().cpu()
    if t.dtype in (torch.bfloat16, torch.float16) and not accepts_bf16:
        t = t.to(torch.float32)
    if t.dtype is torch.bfloat16:
        return data_pb2.Tensor(
            shape=list(t.shape),
            dtype="bfloat16",
            data=t.view(torch.uint16).numpy().tobytes(),
        )
    arr = t.numpy()
    return data_pb2.Tensor(
        shape=list(arr.shape),
        dtype=str(arr.dtype),
        data=arr.tobytes(),
    )


def topk_to_proto(indices: torch.Tensor, values: torch.Tensor) -> data_pb2.Tensor:
    ids = indices.detach().cpu().to(torch.int64).numpy()
    scores = values.detach().cpu().to(torch.float32).numpy()
    return data_pb2.Tensor(
        shape=[int(ids.shape[0])],
        dtype=TOPK_DTYPE,
        data=ids.tobytes() + scores.tobytes(),
    )


def proto_to_tensor(p: data_pb2.Tensor) -> torch.Tensor:
    if p.dtype == "bfloat16":
        arr = np.frombuffer(p.data, dtype=np.uint16).reshape(tuple(p.shape))
        return torch.from_numpy(arr.copy()).view(torch.bfloat16)
    np_dtype = np.dtype(p.dtype)
    arr = np.frombuffer(p.data, dtype=np_dtype).reshape(tuple(p.shape))
    return torch.from_numpy(arr.copy())


class InferenceWorkerServicer(data_pb2_grpc.InferenceWorkerServicer):
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.slice = ModelSlice(device=config.device)
        self.runner: SliceRunner | None = None
        self.scheduler: BatchScheduler | None = None
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
            self.scheduler = (
                BatchScheduler(self.runner, max_batch=self.config.max_batch)
                if self.config.max_batch > 1
                else None
            )
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
            positions = (
                proto_to_tensor(request.position_ids)
                if request.HasField("position_ids")
                else None
            )
            if positions is not None:
                # Multi-axis positions cannot be batched with other sessions,
                # which carry their own; run this one on its own.
                out = self.runner.run(
                    tensor_in,
                    is_input_ids=is_input_ids,
                    session_id=request.session_id,
                    use_cache=request.use_cache,
                    top_k=request.top_k,
                    position_ids=positions,
                )
            elif self.scheduler is not None:
                out = self.scheduler.submit(
                    tensor_in,
                    is_input_ids=is_input_ids,
                    session_id=request.session_id,
                    use_cache=request.use_cache,
                    top_k=request.top_k,
                )
            else:
                out = self.runner.run(
                    tensor_in,
                    is_input_ids=is_input_ids,
                    session_id=request.session_id,
                    use_cache=request.use_cache,
                    top_k=request.top_k,
                )
            if isinstance(out, tuple):
                payload = topk_to_proto(*out)
            else:
                payload = tensor_to_proto(out, accepts_bf16=request.accepts_bf16)
            return data_pb2.SliceResponse(
                session_id=request.session_id,
                activations=payload,
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
            templated = False
            if len(request.messages) > 0:
                messages = [{"role": m.role, "content": m.content} for m in request.messages]
                text = self.slice.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                templated = True
            elif request.apply_chat_template:
                messages = [{"role": "user", "content": request.text}]
                text = self.slice.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                templated = True
            else:
                text = request.text
            ids = self.slice.tokenizer(
                text, return_tensors="pt", add_special_tokens=not templated
            )["input_ids"][0].tolist()
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
    
    def EmbedMedia(self, request, context):
        """Turn text and attachments into hidden states, where the media stays.

        Diffuse runs this on the machine that owns the prompt, so a picture or a
        recording is consumed locally and only activations travel."""
        if self.slice.processor is None or self.slice.embed_tokens is None:
            return data_pb2.EmbedMediaResponse(
                ok=False,
                error="this slice cannot embed media: it holds no processor or no embeddings",
            )
        try:
            import io

            images, audio, videos = [], [], []
            for item in request.media:
                if item.kind == "audio":
                    import soundfile

                    samples, _rate = soundfile.read(io.BytesIO(item.data))
                    audio.append(samples)
                elif item.kind == "video":
                    # A processor wants decoded frames, not a container. The
                    # bytes are written out because the decoders read files.
                    import tempfile

                    import imageio.v2 as iio

                    suffix = "." + (item.mime.split("/")[-1] or "mp4")
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
                        fh.write(item.data)
                        clip_path = fh.name
                    try:
                        import numpy as np

                        videos.append(
                            [np.asarray(f) for f in iio.mimread(clip_path, memtest=False)]
                        )
                    finally:
                        import os

                        os.unlink(clip_path)
                else:
                    from PIL import Image

                    images.append(Image.open(io.BytesIO(item.data)).convert("RGB"))

            # The chat template is what puts a placeholder in the prompt for every
            # attachment. Sending the text alone would leave the processor with
            # media it has nowhere to insert.
            parts = [{"type": item.kind or "image"} for item in request.media]
            if len(request.messages) > 0:
                conversation = [
                    {"role": m.role, "content": m.content} for m in request.messages
                ]
                if parts and conversation:
                    last = conversation[-1]
                    last["content"] = parts + [{"type": "text", "text": last["content"]}]
                text = self.slice.processor.apply_chat_template(
                    conversation, add_generation_prompt=True
                )
            elif request.apply_chat_template:
                content = parts + [{"type": "text", "text": request.text}]
                text = self.slice.processor.apply_chat_template(
                    [{"role": "user", "content": content}], add_generation_prompt=True
                )
            else:
                text = request.text

            kwargs = {"text": text, "return_tensors": "pt"}
            if images:
                kwargs["images"] = images
            if audio:
                kwargs["audio"] = audio
            if videos:
                kwargs["videos"] = videos
            inputs = self.slice.processor(**kwargs)

            embedder = MediaEmbedder(self.slice)
            embeds = embedder.embed(inputs)
            positions = embedder.rope_positions(inputs)
            return data_pb2.EmbedMediaResponse(
                ok=True,
                embeddings=tensor_to_proto(embeds, accepts_bf16=request.accepts_bf16),
                token_count=int(embeds.shape[1]),
                position_ids=(
                    tensor_to_proto(positions) if positions is not None else None
                ),
            )
        except Exception as exc:
            log.exception("media embedding failed")
            return data_pb2.EmbedMediaResponse(ok=False, error=str(exc))

    def SearchModels(self, request, context):
        try:
            from diffuse_worker.catalog import search

            cards = search(
                query=request.query,
                limit=request.limit or 40,
                hf_token=request.hf_token or self.config.hf_token or None,
                supported_only=request.supported_only,
            )
            from diffuse_worker.capacity import available_memory

            available, device = available_memory()
            token = request.hf_token or self.config.hf_token or None
            account = ""
            if token:
                try:
                    from huggingface_hub import HfApi

                    account = HfApi(token=token).whoami().get("name", "")
                except Exception:
                    account = ""
            return data_pb2.SearchModelsResponse(
                ok=True,
                models=[
                    # The descriptor may carry more than the wire cares about;
                    # keep the fields the message actually declares.
                    data_pb2.ModelCard(
                        **{
                            k: v
                            for k, v in card.items()
                            if k in data_pb2.ModelCard.DESCRIPTOR.fields_by_name
                        }
                    )
                    for card in cards
                ],
                available_bytes=available,
                device=device,
                authenticated=bool(token),
                account=account,
            )
        except Exception as exc:
            log.exception("model search failed")
            return data_pb2.SearchModelsResponse(ok=False, error=str(exc))

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
    if config.torch_threads > 0:
        torch.set_num_threads(config.torch_threads)
    options = [
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
        ("grpc.so_reuseport", 0),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_concurrency), options=options
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