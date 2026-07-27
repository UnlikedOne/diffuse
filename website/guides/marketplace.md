# The marketplace

Run `diffuse host` without naming a model and you land in a browser for the
Hugging Face Hub, in your terminal, filtered to what Diffuse can actually split
and to what your machine can actually hold.

```bash
diffuse host
```

```
  ▞▞  DIFFUSE  marketplace
  CPU  ·  11.2 GB free  ·  signed in as unlikedone

┌ press / to search ──────────────────────────────────────────────────┐
│ most downloaded models                                              │
└─────────────────────────────────────────────────────────────────────┘
┌ 40 models, live from hugging face ─────────┐┌ model ────────────────┐
│  Qwen/Qwen2.5-0.5B-Instruct  494M whole…   ││Qwen/Qwen2-VL-2B       │
│  HuggingFaceTB/SmolVLM-256M  256M whole…   ││architecture Qwen2VL…  │
│▍ Qwen/Qwen2-VL-2B-Instruct   2.2B whole…   ││layers       28        │
│  mistralai/Voxtral-Mini-3B   4.7B slice    ││takes  text, image, …  │
│  meta-llama/Llama-3.1-70B   70.0B slice    ││this machine holds the │
│  openai/whisper-small        244M ready    ││whole model            │
└────────────────────────────────────────────┘└───────────────────────┘
 up down  move    /  search    enter  host it    q  quit
```

## Keys

| Key | Does |
|-----|------|
| `↑` `↓` or `j` `k` | move through the list |
| `/` | search the Hub; `enter` runs the search, `esc` cancels |
| `enter` | profile the selected model and offer to host it |
| `q` or `esc` | leave without hosting |

## What the colours mean

Each row is judged against the machine you are on.

| Badge | Colour | Meaning |
|-------|--------|---------|
| `whole model` | green | every layer fits here; you could serve it alone |
| `as a slice` | cyan | too big alone, but you can hold part of it |
| `cannot run` | coral | this machine cannot hold even one layer, or Diffuse cannot split it |

The column after that lists what the model **reads** beyond text — `image`,
`audio`, `video`, or a combination. A `gated` marker means the repository needs
you to accept a licence.

If other people already serve a model, the row says how many, and whether the
model is complete. A gap is an opportunity: hosting the missing range is what
makes a model servable for everyone.

## Nothing here is a list in the code

This matters more than it sounds. Diffuse does not ship a curated set of
blessed models. Every row you see was fetched from the Hub a second ago, and
each model's own `config.json` decides the verdict:

- the depth of its stack, wherever the checkpoint keeps it
- whether it has an encoder tower, and for which modality
- whether its architecture generates at all

Sixty configs are fetched in parallel, which costs about a second. A checkpoint
published tomorrow is judged by the same rules as one published last year, with
nobody adding its name anywhere. That is why `Qwen2-VL` reports video: it says
so in its own config, through a `video_token_id`, and the rule reads it.

A model Diffuse cannot serve is shown with the reason rather than hidden:

```
openai/whisper-small     encoder-decoder: its encoder runs on your machine
state-spaces/mamba-130m  recurrent state, which one slice cannot hand to the next
google-bert/bert-base    BertForMaskedLM does not generate
```

## Signing in to Hugging Face

Set `HF_TOKEN` before you start and the banner says so:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
diffuse host
```

```
CPU  ·  11.2 GB free  ·  signed in as unlikedone
```

Without it:

```
CPU  ·  11.2 GB free  ·  anonymous, gated models stay hidden
```

Signed in, gated repositories resolve and appear normally. Anonymous, they are
listed with `config unreachable, a token may be required` rather than silently
dropped, so you know something is there rather than wondering why a model you
expected is missing.

The token stays on your machine. It is used to talk to Hugging Face and is
never sent to another node.

## What happens when you press enter

Selecting a model does not immediately download anything. Diffuse profiles it
against your machine first:

```
◆ Qwen/Qwen2.5-7B-Instruct
    28 layers, ~520.0 MB per layer, 11.2 GB available on cpu
    this machine can hold 21 layers (75% of the model)
    ◍ the network needs layers 21:28

host this model? (Y/n)
```

The list view's verdict is an estimate from the parameter count. This number is
measured: it reads the checkpoint's real tensor sizes and your real free
memory. It can disagree with the list, and when it does, this one is right.

Say yes and the node downloads only the slice it was assigned, announces itself,
and starts serving.

## Skipping the marketplace

If you already know what you want, name it and the browser never opens:

```bash
diffuse host --model Qwen/Qwen2.5-0.5B-Instruct
```
