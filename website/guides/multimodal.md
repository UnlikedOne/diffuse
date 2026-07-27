# Images, audio, and video

Diffuse is not a text-only network. A model that reads pictures reads them, a
model that listens listens, and a model that answers with sound gives you a
file. The same rule holds throughout: **the media never leaves your machine.**

## The short version

```bash
# Ask a question about a picture
diffuse query --model HuggingFaceTB/SmolVLM-256M-Instruct \
  --prompt "What is in this image?" --image photo.png

# Transcribe a recording
diffuse query --model openai/whisper-small \
  --prompt "" --audio meeting.wav

# Ask about a video
diffuse query --model Qwen/Qwen2-VL-2B-Instruct \
  --prompt "What happens in this clip?" --video clip.mp4

# Generate music, and get a .wav back
diffuse query --model facebook/musicgen-small \
  --prompt "a calm piano melody" --max-tokens 256
```

## The flags

| Flag | What it takes | Repeatable |
|------|---------------|------------|
| `--image <path>` | any image the model's processor accepts | yes |
| `--audio <path>` | wav, mp3, flac, ogg, m4a | yes |
| `--video <path>` | mp4, mov, mkv, webm, avi | yes |
| `--media <path>` | anything; the kind comes from the file extension | yes |

`--media` is the universal one. It reads the extension and decides for itself,
so an unfamiliar format is a matter of naming a file rather than of changing
code. If the extension means nothing to it, it says so and tells you which flag
would settle the question:

```
Error: cannot tell what kind of media notes.xyz is;
pass it with --image, --audio or --video
```

You can mix and repeat:

```bash
diffuse query --model Qwen/Qwen2-VL-2B-Instruct \
  --prompt "Which of these two pictures is brighter?" \
  --image left.png --image right.png
```

## The same thing, in a conversation

`diffuse chat` takes attachments too, as commands rather than flags. Attach
first, then ask; the files apply to the next message only.

```
● › /image left.png
● › /image right.png
● › Which of these two pictures is brighter?
```

`/attach` is the universal one, `/files` shows what is queued, `/detach` drops
it. A model that answers with audio or an image writes its answer to a file and
prints the path, exactly as `query` does. The [chat guide](/guides/chat) has the
complete list.

## Where your picture actually goes

Nowhere. That is the whole point, and it is worth being precise about.

When you attach a file, your own machine runs the model's **encoder tower** —
the vision or audio front end — and turns the picture or the recording into
hidden states. Those are the same kind of numbers a text prompt produces after
the first few layers. That is what crosses the network.

```
your machine                       the network
┌─────────────────────┐
│ photo.png           │
│   ↓ encoder tower   │
│ hidden states  ─────┼──(encrypted)──▶  node   ──▶  node   ──▶  answer
└─────────────────────┘
```

You will see it say so as it runs:

```
◆ image photo.png stays on this machine, only activations leave
→ embedded into 1139 hidden states
→ generating over encrypted channel...
```

A node holding layers 12 to 24 receives a block of floats. It has no image
decoder, no copy of your file, and no way to recover one from what it is given.
This is the same guarantee text prompts already had, extended to pixels and
sound.

::: tip Why the client does the heavy front end
It costs you a few hundred megabytes of model to hold the encoder. That is the
price of the guarantee. If a node did the encoding, your photo would exist in
the clear on a stranger's machine, and no amount of transport encryption would
change that.
:::

## When the answer is not text

Some models answer with sound. `diffuse query` writes the file and tells you
where it went:

```
→ this model answers with audio on 4 streams
→ generating over encrypted channel...

answer:
  36.3 KB of audio/wav after 31 steps
  /home/you/diffuse-query-a1.wav
```

The same rule applies in reverse. The network returns **tokens**, not sound.
Your machine turns those tokens back into a waveform with the model's own
codec. A node never holds the finished answer any more than it held your
original file.

"Four streams" is worth explaining. An audio model does not emit one token per
step the way a text model does; it emits several in parallel, one per codebook,
which the codec later reads together. Text is the one-stream case, not the
normal case.

## What works today

| Model | Takes | Returns | Verified |
|-------|-------|---------|----------|
| `Qwen/Qwen2.5-*` | text | text | yes |
| `HuggingFaceTB/SmolVLM-*` | text, image | text | yes |
| `Qwen/Qwen2-VL-*` | text, image, video | text | yes |
| `mistralai/Voxtral-Mini-*` | text, audio | text | yes |
| `openai/whisper-*` | audio | text | yes |
| `facebook/musicgen-*` | text | **audio** | yes |

Every row was run end to end and compared against the same model running whole
on one machine. Where the comparison is exact, it is noted in
[model support](/concepts/model-support).

Image generation is not verified. The mechanism is the same as audio — a model
that emits image tokens, and a decoder on your machine that turns them into
pixels — but no such model small enough to test has been run. Do not assume it
works until it appears in this table.

## Getting the extras

Media handling needs a few Python packages the text-only path does not:

```bash
cd worker && .venv/bin/pip install -e ".[multimodal]"
```

That pulls in `torchvision` for pixels, `soundfile` and `librosa` for audio,
`imageio` for video containers, and `mistral-common` for Mistral checkpoints,
which ship a `tekken.json` instead of a tokenizer config.

::: warning torchvision must match your torch
Install it with `--no-deps` if pip wants to replace your torch. A mismatched
`torchvision` imports but fails at the first operation with
`operator torchvision::nms does not exist`.
:::

## Common problems

**"this slice cannot embed media: it holds no processor or no embeddings."**
Your client did not load the model's front end. This happens when the local
worker is an older build than the binary; see
[troubleshooting](/troubleshooting#the-installed-worker-is-not-the-one-you-edited).

**A multimodal model gives the same answer whatever you ask it in text.** Your
worker predates the fix for this. A multimodal checkpoint lays out a
conversation as a list of parts; handed a plain string, its tokenizer's template
does not fail, it writes an empty turn — so the model answers a question it
never received. Update the worker, and read
[troubleshooting](/troubleshooting#the-installed-worker-is-not-the-one-you-edited)
if you are not sure which copy is running.

**The answer ignores the picture.** Check that the model actually takes images.
`diffuse models` and the marketplace both show what a model reads. A text model
handed an image will answer from the text alone.

**A video answer that is right for the wrong reason.** Small vision models often
cannot read motion, only frames. If you ask which way something moves and get a
confident answer, try the mirrored clip before believing it. This is a limit of
the model, not of Diffuse.
