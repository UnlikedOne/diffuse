import io
import threading

import torch

# The loop that turns streams of tokens back into an answer is the one part of
# inference that stays model-specific: a music model interleaves its codebooks,
# a text model does not. It lives here, on the machine that asked the question,
# so the orchestrator only ever relays tensors and never learns what any
# particular model expects.


class GenerationSession:
    """Owns the generation state for one request, on the client's own machine."""

    def __init__(self, model_slice, max_new_tokens: int = 256):
        self.slice = model_slice
        self.max_new_tokens = max_new_tokens
        self.memory = None
        self.buffer = None
        self.mask = None
        self.produced: list[int] = []
        self.finished = False

    # -- what the model is shaped like, asked rather than assumed -------------

    def _decoder(self):
        model = self.slice.model
        for holder in (getattr(model, "decoder", None), model):
            if holder is not None and hasattr(holder, "build_delay_pattern_mask"):
                return holder
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

    # -- the loop ------------------------------------------------------------

    @torch.inference_mode()
    def begin(self, inputs) -> torch.Tensor:
        """Prepare the first thing to feed the network, and the encoder memory."""
        model = self.slice.model
        encoder = getattr(model, "text_encoder", None)
        if encoder is not None:
            kwargs = {
                k: v
                for k, v in inputs.items()
                if k in ("input_ids", "attention_mask")
            }
            encoded = encoder(**kwargs).last_hidden_state
            projection = getattr(model, "enc_to_dec_proj", None)
            self.memory = projection(encoded) if projection is not None else encoded

        decoder = self._decoder()
        if decoder is None:
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
        return self.buffer[:, -1:].reshape(1, streams, 1)

    @torch.inference_mode()
    def advance(self, streams: torch.Tensor) -> torch.Tensor | None:
        """Take what the last slice returned and say what to feed next."""
        decoder = self._decoder()
        if decoder is None:
            token = int(streams.reshape(-1)[0]) if streams.numel() else 0
            self.produced.append(token)
            if len(self.produced) >= self.max_new_tokens:
                self.finished = True
                return None
            return torch.tensor([[token]], dtype=torch.long)

        # [batch, streams, position, vocab] -> one token per stream
        picked = streams[:, :, -1, :].argmax(-1).reshape(-1, 1)
        self.buffer = torch.cat([self.buffer, picked], dim=-1)
        self.buffer = decoder.apply_delay_pattern_mask(
            self.buffer, self.mask[:, : self.buffer.shape[-1]]
        )
        if self.buffer.shape[-1] >= self.mask.shape[-1]:
            self.finished = True
            return None
        count = self.stream_count()
        return self.buffer[:, -1:].reshape(1, count, 1)

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
