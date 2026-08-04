# Changelog

## v0.2.9

The release where Diffuse stops being a text-only network. Models that see,
listen, speak, and models that answer by denoising rather than by predicting a
token, all run split across machines now. Alongside that: the wire got much
smaller, the nodes talk to each other instead of through you, and there is a
prebuilt binary for every platform people actually asked for.

Every number below was measured, and the conditions are stated. A number without
its conditions is not worth having.

### Install without a toolchain

Binaries are now built by CI for five platforms and published with the release.
Nobody needs Rust installed any more.

| Platform | Binary |
|---|---|
| Linux x86_64 | `diffuse-linux-x86_64` |
| Linux aarch64 | `diffuse-linux-aarch64` |
| macOS Apple Silicon | `diffuse-macos-aarch64` |
| macOS Intel | `diffuse-macos-x86_64` |
| Windows x86_64 | `diffuse-windows-x86_64.exe` |

Each ships with a `.sha256`, and both installers verify it before putting the
binary anywhere.

```bash
# Linux and macOS
curl -fsSL https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.sh | bash
```

```powershell
# Windows
irm https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.ps1 | iex
```

The Windows path used to require Rust and build from source; it no longer does.
The daemon also looked for its Python worker at `.venv/bin/python`, which does
not exist on Windows — it now looks in `Scripts` there, so the Windows binary can
actually start its worker.

### Diffusion models across nodes

Image, video and audio models that denoise — Wan, Flux, SD3, LTX, Mochi,
CogVideoX, PixArt, HunyuanVideo — are split across machines like any other
model, using patch-level pipeline parallelism
([PipeFusion](https://arxiv.org/abs/2405.14430)).

```bash
diffuse query --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "a paper boat on a puddle" --steps 20 --seed 7
```

Measured on **Wan2.1-T2V-1.3B**, thirty blocks over two nodes, 20 steps,
256×256, nine frames, against the same model run whole on one machine:

| | Difference |
|---|---|
| `--patches 1` (the default) | **none — bit-identical** |
| `--patches 2` | mean 36.5 / 255 |
| `--patches 4` | mean 52.8 / 255 |

Splitting a model across machines costs nothing. Splitting each denoising step
into patches keeps the scene — same boat, same rain, same ripples — and shifts
its tone warmer and harder. That is a trade to choose, so `--patches` defaults
to 1.

Nothing in the mechanism knows what a Flux is. Block stacks are found by shape,
the config entry that sizes each one is found by nudging it on a weightless copy
of the model, and the two ends run by calling the pipeline's own code with the
stacks replaced by a stand-in.

### Models that are not text

- **Sight, hearing, video in.** `diffuse query --image`, `--audio`, `--video`,
  `--media`, and the same attachments from inside `diffuse chat`. Multi-axis
  (M-RoPE) positions travel with the activations, so a sliced video model answers
  exactly like the unsliced one.
- **Speech in.** Whisper transcribes through slices.
- **Sound out.** A model that answers with audio rather than words returns a
  file. MusicGen writes four codebooks per step across the network.
- **Encoder-decoder models** are sliced, with the encoder tower running on your
  own machine, where the prompt already is.
- Checkpoints whose tensors are not named the way their class is, and whose
  sliceable stack is somewhere unusual, load anyway.

### A marketplace instead of a hardcoded list

`diffuse host` with no model opens a browser over live Hugging Face results
rather than a catalogue written into the binary. It reports what your machine
can hold, remembers where you have been (Esc goes back), and no longer lets the
daemon's logs scribble over the screen while it is up.

### Less on the wire, and off your link

| Per token, last hop | Before | After |
|---|---|---|
| Qwen2.5-0.5B | 608 KB | **130 B** |
| Mistral-Small-24B, 8 nodes | 652 KB total | **70 KB total** |
| MusicGen-small, 4 audio streams | 32.8 KB | **74 B** |

Only the top candidates travel, not the whole vocabulary, and ties break the way
`argmax` breaks them, so the answer is unchanged. Activations travel in bfloat16
— the precision the model computes in.

Nodes now hand activations to each other instead of returning through you. Two
nodes, 25 ms client link, twelve tokens: **4059 ms → 2707 ms**. The gain is
entirely from asymmetry; if your client sits in the same datacenter as the nodes
it gains you nothing.

Concurrent sessions are grouped into one pass through the layers. Eight sessions
through the encrypted compute plane: **23.1 → 44.4 tokens/s**, median latency
**344 → 179 ms**. A single stream pays nothing for it — the scheduler never waits
for a batch to fill.

Relay connections stay open for a whole session instead of being rebuilt per hop.

### Correctness fixes worth naming

- **Keyword arguments reached the nodes.** The stand-in that replaces a
  diffusion block stack forwarded only positional arguments. Wan passes
  everything positionally, so it worked; Flux, SD3, LTX, Mochi, CogVideoX and
  PixArt pass by keyword, and on those the text conditioning, the timestep and
  the positions never arrived. No error, just noise. Blocks that return two
  tensors, models with more than one block stack, and non-tensor arguments are
  all handled now.
- **Audio models got their positions.** A sinusoidal position table is a plain
  module, not an `nn.Embedding`, and the search for one required an
  `nn.Embedding` — so no slice ever found it and the right weights ran in the
  wrong order. Sliced MusicGen logits are now **0.0** away from the whole model.
- **Guidance and sampling happen where the numbers are.** A model that asks for
  classifier-free guidance runs both branches and the last slice combines them;
  sampling draws from the whole distribution rather than from a shortlist, since
  truncating moves the mass onto the strongest candidates. Against
  `transformers`' own `generate`, the waveform now matches to one step of 16-bit
  PCM quantisation.
- **A multimodal model asked in text kept the question.** The tokenizer's
  template rendered `User: <end_of_utterance>` and dropped the prompt; the
  processor's template is used when there is one.
- **Sessions survive a lost replica.** A broken route is replayed on a repaired
  one rather than failing the request.
- **Old and new nodes coexist.** The wire format is negotiated, and a peer's
  version is learned from the peer rather than from gossip about it.

### Also

- The client no longer holds the blocks it just handed to the nodes — 6 GB it
  never read, on a video model.
- The installers now pull the media and diffusion extras, so audio, video and
  image generation work on a fresh install.
- Documentation rewritten for what the network actually does, including a page
  on [how diffusion is split](https://unlikedone.github.io/diffuse/concepts/diffusion).

### A correction

The diffusion page used to claim patch splitting cost about a third of one
greyscale level. That was measured on a four-block transformer with random
weights, where a right picture and a wrong one are both noise. It said nothing
about a real model and was presented as though it did. The table above replaces
it with Wan2.1-1.3B.

---

## v0.2.8 and earlier

See the [releases page](https://github.com/UnlikedOne/diffuse/releases).
