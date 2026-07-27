import logging
import threading
from concurrent import futures

import grpc
import numpy as np
import torch

from diffuse_worker import data_pb2, data_pb2_grpc
from diffuse_worker.batching import BatchScheduler
from diffuse_worker.config import WorkerConfig
from diffuse_worker.diffusion import (
    DiffusionSession,
    DiffusionStack,
    changed_arguments,
    flatten_arguments,
    read_index,
    rebuild_arguments,
)
from diffuse_worker.generation import GenerationSession, SessionStore
from diffuse_worker.inference import MediaEmbedder, SliceRunner
from diffuse_worker.slicing import ModelSlice

logging.basicConfig(level=logging.INFO)
def _branch_of(values):
    for value in values:
        if value.dim() >= 2 and value.shape[1] > 1:
            flat = value.reshape(-1)[:64].to(torch.float32)
            return f"{tuple(value.shape)}:{float(flat.sum()):.6e}"
    return "single"


log = logging.getLogger("diffuse.worker")

MAX_MESSAGE_BYTES = 128 * 1024 * 1024


TOPK_DTYPE = "topk_i64_f32"
MULTI_TOPK_DTYPE = "topk_multi_i64_f32"


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


def multi_topk_to_proto(rows) -> data_pb2.Tensor:
    """One shortlist per stream, packed as ids then scores."""
    ids = torch.stack([r[0] for r in rows]).to(torch.int64).numpy()
    scores = torch.stack([r[1] for r in rows]).to(torch.float32).numpy()
    return data_pb2.Tensor(
        shape=list(ids.shape),
        dtype=MULTI_TOPK_DTYPE,
        data=ids.tobytes() + scores.tobytes(),
    )


def proto_to_streams(p: data_pb2.Tensor) -> torch.Tensor:
    """Candidate ids per stream, whatever form the last slice sent."""
    if p.dtype == MULTI_TOPK_DTYPE:
        streams, k = int(p.shape[0]), int(p.shape[1])
        count = streams * k
        ids = np.frombuffer(p.data, dtype=np.int64, count=count).reshape(streams, k)
        return torch.from_numpy(ids.copy())
    if p.dtype == TOPK_DTYPE:
        k = int(p.shape[0])
        ids = np.frombuffer(p.data, dtype=np.int64, count=k)
        return torch.from_numpy(ids.copy()).reshape(1, k)
    return proto_to_tensor(p)


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
        self.sessions = SessionStore()
        self.loaded = False
        self.stack = None
        self.patch_state = {}
        self.diffusions = {}
        self._diffusion_lock = threading.Lock()

    def LoadSlice(self, request, context):
        token = request.hf_token or self.config.hf_token or None
        try:
            if read_index(request.model_id, token) is not None:
                stack = DiffusionStack(device=self.config.device)
                total = stack.load(
                    request.model_id, request.start_layer, request.end_layer, token
                )
                self.stack = stack
                self.loaded = True
                log.info(
                    "loaded diffusion blocks %d:%d of %d for %s",
                    request.start_layer,
                    request.end_layer,
                    total,
                    request.model_id,
                )
                return data_pb2.LoadSliceResponse(ok=True, total_layers=total)
        except Exception as exc:
            log.exception("diffusion load failed")
            return data_pb2.LoadSliceResponse(ok=False, error=str(exc))
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
        if request.patch_sequence:
            return self._run_patch(request)
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
            memory = (
                proto_to_tensor(request.encoder_memory)
                if request.HasField("encoder_memory")
                else None
            )
            if positions is not None or memory is not None:
                # Multi-axis positions cannot be batched with other sessions,
                # which carry their own; run this one on its own.
                out = self.runner.run(
                    tensor_in,
                    is_input_ids=is_input_ids,
                    session_id=request.session_id,
                    use_cache=request.use_cache,
                    top_k=request.top_k,
                    position_ids=positions,
                    memory=memory,
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
            if isinstance(out, list) and out and isinstance(out[0], tuple):
                payload = multi_topk_to_proto(out)
            elif isinstance(out, tuple):
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
                text = self._apply_template(messages)
                templated = True
            elif request.apply_chat_template:
                text = self._apply_template([{"role": "user", "content": request.text}])
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

    def _apply_template(self, messages) -> str:
        """Lay out a conversation the way this particular checkpoint expects.

        A multimodal checkpoint templates content as a list of parts. Handing
        its tokenizer a plain string does not fail: the template simply finds no
        part it recognises and writes an empty turn, so the model answers a
        question it was never asked. The processor's template understands parts,
        so it is the one asked whenever the checkpoint ships one."""
        processor = self.slice.processor
        if processor is not None and hasattr(processor, "apply_chat_template"):
            parts = [
                {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                for m in messages
            ]
            try:
                return processor.apply_chat_template(parts, add_generation_prompt=True)
            except Exception:
                log.warning("processor template failed; falling back to the tokenizer")
        return self.slice.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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

    def BeginGeneration(self, request, context):
        """Open a generation on this machine and say what it will produce."""
        if self.slice.model is None:
            # A plain decoder needs nothing special: say so plainly rather than
            # failing, so the caller takes the ordinary path without a warning.
            return data_pb2.BeginGenerationResponse(
                ok=True, streams=1, output_kind="text"
            )
        try:
            processor = self.slice.processor or self.slice.tokenizer
            inputs = self._build_inputs(processor, request)
            session = GenerationSession(
                self.slice, max_new_tokens=request.max_new_tokens or 256
            )
            first = session.begin(inputs)
            self.sessions.start(request.session_id, session)
            return data_pb2.BeginGenerationResponse(
                ok=True,
                input_ids=tensor_to_proto(first),
                encoder_memory=(
                    tensor_to_proto(session.memory, accepts_bf16=request.accepts_bf16)
                    if session.memory is not None
                    else None
                ),
                streams=session.stream_count(),
                output_kind=session.output_kind(),
            )
        except Exception as exc:
            log.exception("begin generation failed")
            return data_pb2.BeginGenerationResponse(ok=False, error=str(exc))

    def _build_inputs(self, processor, request):
        """Hand the processor the media under the names it knows.

        Introspection does not help here: a processor's __call__ is usually
        (*args, **kwargs), so the natural names are passed and the sampling rate
        always travels with audio, which is what decides whether a spectrogram
        or a tokenizer is used."""
        import io

        images, audio, rate = [], [], 16000
        for item in request.media:
            if item.kind == "audio":
                import soundfile

                samples, sample_rate = soundfile.read(io.BytesIO(item.data))
                audio.append(samples)
                rate = int(sample_rate)
            elif item.kind == "video":
                import tempfile

                import imageio.v2 as iio
                import numpy as np

                suffix = "." + (item.mime.split("/")[-1] or "mp4")
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
                    fh.write(item.data)
                    path = fh.name
                try:
                    images.extend(np.asarray(f) for f in iio.mimread(path, memtest=False))
                finally:
                    import os

                    os.unlink(path)
            else:
                from PIL import Image

                images.append(Image.open(io.BytesIO(item.data)).convert("RGB"))

        kwargs = {"return_tensors": "pt"}
        if audio:
            kwargs["audio"] = audio[0] if len(audio) == 1 else audio
            kwargs["sampling_rate"] = rate
        if images:
            kwargs["images"] = images
        text = self._prompt_with_placeholders(processor, request)
        if request.text or not kwargs.get("audio") is not None and not images:
            kwargs["text"] = [text]
        if "text" not in kwargs and not audio and not images:
            kwargs["text"] = [text]

        try:
            return processor(**kwargs)
        except TypeError:
            if audio:
                return processor(audio[0], sampling_rate=rate, return_tensors="pt")
            return processor(request.text, return_tensors="pt")

    def _run_patch(self, request):
        if self.stack is None:
            return data_pb2.SliceResponse(
                session_id=request.session_id, ok=False, error="no diffusion blocks loaded"
            )
        try:
            key = f"{request.session_id}:{request.branch}"
            with self._diffusion_lock:
                state = self.patch_state.setdefault(key, {"values": {}, "layout": ""})
                for item in request.arguments:
                    state["values"][item.index] = proto_to_tensor(item.value)
                if request.layout:
                    state["layout"] = request.layout
                values = [state["values"][i] for i in sorted(state["values"])]
                layout = state["layout"]
            arguments = rebuild_arguments(values, layout) if layout else tuple(values)
            out = self.stack.run_patch(
                key,
                proto_to_tensor(request.activations),
                int(request.patch_offset),
                int(request.patch_sequence),
                arguments,
            )
            return data_pb2.SliceResponse(
                session_id=request.session_id,
                ok=True,
                activations=tensor_to_proto(out, accepts_bf16=request.accepts_bf16),
            )
        except Exception as exc:
            log.exception("diffusion patch failed")
            return data_pb2.SliceResponse(
                session_id=request.session_id, ok=False, error=str(exc)
            )

    def _prompt_with_placeholders(self, processor, request) -> str:
        """The prompt with one placeholder per attachment, when the model uses them.

        A vision processor refuses a picture it has nowhere to put: the count of
        <image> markers in the text has to match the count of images. The chat
        template is what writes those markers, so it is applied here whenever
        media travels. Processors that do not template (Whisper, MusicGen) fall
        back to the plain text, which is what they expect."""
        if not request.media:
            return request.text
        template = getattr(processor, "apply_chat_template", None)
        if template is None:
            return request.text
        content = [{"type": item.kind or "image"} for item in request.media]
        content.append({"type": "text", "text": request.text})
        try:
            return template(
                [{"role": "user", "content": content}], add_generation_prompt=True
            )
        except Exception:
            return request.text

    def AdvanceGeneration(self, request, context):
        session = self.sessions.get(request.session_id)
        if session is None:
            return data_pb2.AdvanceGenerationResponse(ok=False, error="unknown session")
        try:
            nxt = session.advance(proto_to_streams(request.streams))
            return data_pb2.AdvanceGenerationResponse(
                ok=True,
                input_ids=tensor_to_proto(nxt) if nxt is not None else None,
                finished=nxt is None,
            )
        except Exception as exc:
            log.exception("advance generation failed")
            return data_pb2.AdvanceGenerationResponse(ok=False, error=str(exc))

    def FinishGeneration(self, request, context):
        session = self.sessions.drop(request.session_id)
        if session is None:
            return data_pb2.FinishGenerationResponse(ok=False, error="unknown session")
        try:
            data, mime, text = session.finish()
            return data_pb2.FinishGenerationResponse(ok=True, data=data, mime=mime, text=text)
        except Exception as exc:
            log.exception("finish generation failed")
            return data_pb2.FinishGenerationResponse(ok=False, error=str(exc))

    def _diffusion_step(self, session, item, first):
        if item is None:
            return None
        hidden, arguments = item
        values, layout = flatten_arguments(arguments)
        branch = _branch_of(values)
        previous = session.branches.setdefault(branch, {})
        fresh = changed_arguments(previous, values)
        packed = [
            data_pb2.DiffusionArgument(index=index, value=tensor_to_proto(value))
            for index, value in sorted(fresh.items())
        ]
        return hidden, packed, layout if first or branch not in session.told else "", branch

    def BeginDiffusion(self, request, context):
        token = self.config.hf_token or None
        try:
            options = {}
            if request.height:
                options["height"] = int(request.height)
            if request.width:
                options["width"] = int(request.width)
            if request.frames:
                options["num_frames"] = int(request.frames)
            if request.guidance:
                options["guidance_scale"] = float(request.guidance)
            session = DiffusionSession(
                request.model_id,
                request.prompt,
                max(int(request.steps), 1),
                token=token,
                device=self.config.device,
                options=options,
                seed=int(request.seed) if request.seed else None,
            )
            session.load()
            session.branches = {}
            session.told = set()
            kind = session.output_kind()
            blocks = session.blocks
            step = self._diffusion_step(session, session.begin(), True)
            if step is None:
                raise ValueError("the pipeline produced no transformer call")
            hidden, packed, layout, branch = step
            session.told.add(branch)
            with self._diffusion_lock:
                self.diffusions[request.session_id] = session
            return data_pb2.BeginDiffusionResponse(
                ok=True,
                hidden=tensor_to_proto(hidden, accepts_bf16=request.accepts_bf16),
                arguments=packed,
                layout=layout,
                sequence=hidden.shape[1],
                blocks=blocks,
                output_kind=kind,
                branch=branch,
            )
        except Exception as exc:
            log.exception("begin diffusion failed")
            return data_pb2.BeginDiffusionResponse(ok=False, error=str(exc))

    def AdvanceDiffusion(self, request, context):
        with self._diffusion_lock:
            session = self.diffusions.get(request.session_id)
        if session is None:
            return data_pb2.AdvanceDiffusionResponse(ok=False, error="unknown session")
        try:
            step = self._diffusion_step(
                session, session.advance(proto_to_tensor(request.hidden)), False
            )
            if step is None:
                return data_pb2.AdvanceDiffusionResponse(ok=True, finished=True)
            hidden, packed, layout, branch = step
            session.told.add(branch)
            return data_pb2.AdvanceDiffusionResponse(
                ok=True,
                finished=False,
                hidden=tensor_to_proto(hidden),
                arguments=packed,
                layout=layout,
                sequence=hidden.shape[1],
                branch=branch,
            )
        except Exception as exc:
            log.exception("advance diffusion failed")
            return data_pb2.AdvanceDiffusionResponse(ok=False, error=str(exc))

    def FinishDiffusion(self, request, context):
        with self._diffusion_lock:
            session = self.diffusions.pop(request.session_id, None)
        if session is None:
            return data_pb2.FinishDiffusionResponse(ok=False, error="unknown session")
        try:
            data, mime = session.finish()
            return data_pb2.FinishDiffusionResponse(ok=True, data=data, mime=mime)
        except Exception as exc:
            log.exception("finish diffusion failed")
            return data_pb2.FinishDiffusionResponse(ok=False, error=str(exc))

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