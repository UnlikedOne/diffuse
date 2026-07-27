import io
import threading

import torch


class GenerationSession:
    """Owns the generation state for one request, on the client's own machine."""

    def __init__(self, model_slice, max_new_tokens: int = 256, seed: int | None = None):
        self.slice = model_slice
        self.max_new_tokens = max_new_tokens
        self.memory = None
        self.buffer = None
        self.mask = None
        self.produced: list[int] = []
        self.finished = False
        self._effective = None
        self.seed = seed

    def _config(self):
        """The settings generation would actually run with.

        A checkpoint often leaves sampling unset and lets the library fill it
        in, so reading generation_config alone reports no top-k where the model
        in fact truncates at fifty. Asking for the prepared config gives what
        model.generate() would use, without this having to know any defaults."""
        if self._effective is None:
            model = self.slice.model
            prepare = getattr(model, "_prepare_generation_config", None)
            if prepare is not None:
                try:
                    self._effective = prepare(None)[0]
                except Exception:
                    self._effective = None
            if self._effective is None:
                self._effective = getattr(model, "generation_config", None)
        return self._effective

    def top_k(self) -> int:
        value = getattr(self._config(), "top_k", None)
        return int(value) if value else 0

    def top_p(self) -> float:
        value = getattr(self._config(), "top_p", None)
        return float(value) if value else 1.0

    def guidance(self) -> float:
        scale = getattr(self._config(), "guidance_scale", None)
        return float(scale) if scale and float(scale) > 1.0 else 1.0

    def sampling(self) -> bool:
        return bool(getattr(self._config(), "do_sample", False))

    def temperature(self) -> float:
        value = getattr(self._config(), "temperature", None)
        return float(value) if value else 1.0

    def shortlist_size(self) -> int:
        return 1 if self.sampling() else 8

    def _pick(self, scores: torch.Tensor) -> torch.Tensor:
        return scores.argmax(-1)


    def _decoder(self):
        model = self.slice.model
        for holder in (getattr(model, "decoder", None), model):
            if holder is not None and hasattr(holder, "build_delay_pattern_mask"):
                return holder
        return None

    def _encoder(self):
        """The half that reads, wherever the checkpoint keeps it.

        MusicGen calls it text_encoder, Whisper keeps it under model.encoder.
        Both are the same thing to Diffuse: something the client runs once, on
        its own machine, whose output is what actually travels."""
        model = self.slice.model
        for holder, name in (
            (model, "text_encoder"),
            (model, "encoder"),
            (getattr(model, "model", None), "encoder"),
        ):
            if holder is None:
                continue
            encoder = getattr(holder, name, None)
            if encoder is not None and callable(encoder):
                return encoder
        return None

    def _codec(self):
        """The thing that turns generated tokens back into bytes."""
        model = self.slice.model
        for name in ("audio_encoder", "audio_tower", "vqmodel", "vqgan"):
            codec = getattr(model, name, None)
            if codec is not None and hasattr(codec, "decode"):
                return codec
        return None

    def output_kind(self) -> str:
        if self._codec() is not None:
            return "audio" if self._is_audio() else "image"
        return "text"

    def _is_audio(self) -> bool:
        config = getattr(self.slice.model, "config", None)
        for name in ("audio_encoder", "audio_config"):
            section = getattr(config, name, None)
            if section is not None and getattr(section, "sampling_rate", None):
                return True
        return False

    def sampling_rate(self) -> int:
        config = getattr(self.slice.model, "config", None)
        for name in ("audio_encoder", "audio_config"):
            section = getattr(config, name, None)
            rate = getattr(section, "sampling_rate", None) if section is not None else None
            if rate:
                return int(rate)
        return 32000

    def stream_count(self) -> int:
        for holder in (getattr(self.slice.model, "decoder", None), self.slice.model):
            count = getattr(holder, "num_codebooks", None) if holder is not None else None
            if isinstance(count, int) and count > 0:
                return count
        return 1


    @torch.inference_mode()
    def begin(self, inputs) -> torch.Tensor:
        """Prepare the first thing to feed the network, and the encoder memory."""
        import inspect

        model = self.slice.model
        encoder = self._encoder()
        if encoder is not None:
            accepted = set(inspect.signature(encoder.forward).parameters)
            kwargs = {k: v for k, v in inputs.items() if k in accepted}
            encoded = encoder(**kwargs).last_hidden_state
            projection = getattr(model, "enc_to_dec_proj", None)
            self.memory = projection(encoded) if projection is not None else encoded
            if self.guidance() > 1.0:
                self.memory = torch.cat(
                    [self.memory, torch.zeros_like(self.memory)], dim=0
                )

        decoder = self._decoder()
        if decoder is None:
            if self.memory is not None:
                config = getattr(model, "generation_config", None)
                start = getattr(config, "decoder_start_token_id", None)
                if start is not None:
                    self.buffer = torch.tensor([[int(start)]], dtype=torch.long)
                    self.produced = []
                    return self.buffer
            self.buffer = inputs["input_ids"]
            return self.buffer

        streams = self.stream_count()
        config = getattr(model, "generation_config", None)
        start = getattr(config, "decoder_start_token_id", 0)
        pad = getattr(config, "pad_token_id", 0)
        seed = torch.full((streams, 1), start, dtype=torch.long)
        self.buffer, self.mask = decoder.build_delay_pattern_mask(
            seed, pad_token_id=pad, max_length=self.max_new_tokens + 1
        )
        self.buffer = decoder.apply_delay_pattern_mask(
            self.buffer, self.mask[:, : self.buffer.shape[-1]]
        )
        return self._feed(self.buffer[:, -1:], streams)

    def _feed(self, columns, streams):
        step = columns.reshape(1, streams, 1)
        if self.guidance() > 1.0:
            step = step.repeat(2, 1, 1)
        return step

    @torch.inference_mode()
    def advance(self, streams: torch.Tensor) -> torch.Tensor | None:
        """Take what the last slice returned and say what to feed next."""
        decoder = self._decoder()
        shortlist = isinstance(streams, tuple)
        if shortlist:
            ids, scores = streams
            chosen = ids.gather(-1, self._pick(scores).unsqueeze(-1)).squeeze(-1)
        picked_ids = not shortlist and streams.dtype == torch.int64 and streams.dim() == 2
        if decoder is None:
            if shortlist:
                token = int(chosen.reshape(-1)[0])
            elif picked_ids:
                token = int(streams[0, 0])
            else:
                token = int(streams.reshape(1, -1, streams.shape[-1])[0, -1].argmax())
            eos = getattr(
                getattr(self.slice.model, "generation_config", None), "eos_token_id", None
            )
            self.produced.append(token)
            if len(self.produced) >= self.max_new_tokens or token == eos:
                self.finished = True
                return None
            return torch.tensor([[token]], dtype=torch.long)

        if shortlist:
            picked = chosen.reshape(-1, 1)
        elif picked_ids:
            picked = streams[:, 0].reshape(-1, 1)
        else:
            picked = streams[:, :, -1, :].argmax(-1).reshape(-1, 1)
        self.buffer = torch.cat([self.buffer, picked], dim=-1)
        self.buffer = decoder.apply_delay_pattern_mask(
            self.buffer, self.mask[:, : self.buffer.shape[-1]]
        )
        if self.buffer.shape[-1] >= self.mask.shape[-1]:
            self.finished = True
            return None
        return self._feed(self.buffer[:, -1:], self.stream_count())

    @torch.inference_mode()
    def finish(self) -> tuple[bytes, str, str]:
        """Turn what was generated into bytes the caller can keep."""
        decoder = self._decoder()
        codec = self._codec()
        if decoder is None or codec is None:
            tokenizer = self.slice.tokenizer
            text = tokenizer.decode(self.produced, skip_special_tokens=True) if tokenizer else ""
            return b"", "text/plain", text

        config = getattr(self.slice.model, "generation_config", None)
        pad = getattr(config, "pad_token_id", 0)
        codes = decoder.apply_delay_pattern_mask(self.buffer, self.mask)
        codes = codes[codes != pad].reshape(1, self.stream_count(), -1)
        audio = codec.decode(codes.unsqueeze(0), [None]).audio_values

        import soundfile

        buffer = io.BytesIO()
        soundfile.write(
            buffer, audio[0, 0].to(torch.float32).numpy(), self.sampling_rate(), format="WAV"
        )
        return buffer.getvalue(), "audio/wav", ""


class SessionStore:
    """Generation sessions in flight, keyed by the caller's session id."""

    def __init__(self):
        self._sessions: dict[str, GenerationSession] = {}
        self._lock = threading.Lock()

    def start(self, session_id: str, session: GenerationSession) -> None:
        with self._lock:
            self._sessions[session_id] = session

    def get(self, session_id: str) -> GenerationSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def drop(self, session_id: str) -> GenerationSession | None:
        with self._lock:
            return self._sessions.pop(session_id, None)
