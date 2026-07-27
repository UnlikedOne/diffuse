# Diffusion across nodes

A chat model and a diffusion model look alike from a distance — both are a
stack of near-identical transformer blocks — and splitting them across machines
could not be more different. This page explains why, and what Diffuse does
instead.

## Why the ordinary way fails

Splitting a chat model works because there is always more work to start. While
the last stage finishes token five, the first stage begins token six. The
pipeline stays full.

Diffusion has nothing to fill it with. One picture is twenty to fifty passes
through the entire stack, and pass seventeen cannot begin until pass sixteen has
finished, because it denoises what sixteen produced. Cut the blocks across four
machines and three of them are idle at any moment, waiting their turn. Worse,
what moves between the stages is the whole latent — for video, tens of megabytes
per pass.

So the obvious split gives you a pipeline that is three-quarters empty and
saturates the link between the machines. That is not a tuning problem, it is the
shape of the computation.

## What Diffuse does instead

The picture is cut into **patches**, and the patches move through the stages.
While stage two works on patch A, stage one works on patch B. The pipeline fills
up again — not with future tokens, which do not exist, but with pieces of the
same picture.

One difficulty remains, and it is the whole trick. Attention over a patch needs
the rest of the picture: patch B has to know what patch A looks like at this
stage. Waiting for it would put the idle stages straight back.

**So each stage answers with what it saw one denoising step ago.** Between two
consecutive steps a latent barely moves — that is what denoising means, small
corrections to something already nearly right. A stage attends over the previous
step's picture for the patches it has not been given yet, and over this step's
values for the ones it has.

This is the technique published as
[PipeFusion](https://arxiv.org/abs/2405.14430), itself descended from
[DistriFusion](https://arxiv.org/abs/2402.19481). It was designed for exactly
the situation Diffuse is in: machines connected by something slower than a
datacenter fabric.

## What it costs

Substituting one step's activations for another's is an approximation, so the
question is how much the picture moves. Measured on a four-block Wan video
transformer split over two stages, six denoising steps, against the same model
run whole on one machine:

| Patches per step | Difference from the whole model | Bytes per transfer |
|---|---|---|
| 1 | **none — bit-identical** | 6,144 |
| 2 | 0.032 / 255 | 3,072 |
| 4 | 0.034 / 255 | 1,536 |
| 8 | 0.037 / 255 | 768 |
| 16 | 0.038 / 255 | 384 |

Read the first row first: with a single patch there is no stale data to use, and
the answer is **byte-for-byte** what the unsliced model produced. That is the row
that says the machinery is right rather than merely plausible — the slicing, the
attention rewrite, the client-side ends, the encryption, all of it.

Then read down. The transfer shrinks sixteenfold and the difference does not
grow: about a third of one greyscale level, on pixels that run from 0 to 255.
Nobody has ever seen a third of a level.

::: warning The first step is not cut
There is no previous step to borrow from, so the first denoising pass runs as a
single patch over the whole picture. It is the slowest pass of the generation
and it is unavoidable.
:::

## Where the pieces live

| Piece | Lives on | Why |
|-------|----------|-----|
| Text encoder | the client | the prompt is yours |
| Patch embedding, timestep conditioning | the client | cheap, and it starts the stack |
| Transformer blocks | the nodes | this is the expensive part |
| Output projection, unpatchify | the client | it closes the stack |
| VAE decoder | the client | the finished picture belongs to the asker |
| Scheduler | the client | it decides what the next step denoises |

A node holding blocks 12 to 24 of a video model never sees a frame, never runs a
VAE, and never learns the prompt. It receives a patch of hidden states and
returns a patch of hidden states.

## Nothing here knows what a Flux is

There is no list of supported diffusion families, and no per-model adapter. The
mechanism finds what it needs by shape:

- **The block stack** is whichever `ModuleList` of transformer blocks the model
  exposes — `blocks`, `transformer_blocks`, `single_transformer_blocks`.
- **The two ends** are run by calling the pipeline's own code with the block
  stack replaced by a stand-in. The pipeline prepares latents, conditions on the
  timestep, runs its scheduler and decodes with its VAE exactly as it always
  does; Diffuse only intercepts the stack in the middle. No family-specific
  logic is reimplemented, so nothing drifts out of date.
- **The patch attention** is written against the projections every diffusers
  attention exposes (`to_q`, `to_k`, `to_v`, `to_out`, and the optional query
  and key norms), with rotary positions sliced to the patch for the query and
  left whole for the context.
- **What the model answers with** comes from its own parts: a VAE that declares
  a sampling rate means audio, a patch size with a time axis means video,
  anything else is a picture.

A model that follows those conventions works without anyone adding it to a list.
One that does not will fail on load, loudly, rather than produce nonsense.

## Guidance doubles the work

Most diffusion models use classifier-free guidance: every step is run twice,
once with your prompt and once without, and the two are combined. Diffuse runs
both, which is why a twenty-step generation makes forty passes through the
network.

The two branches are independent, and each keeps its own stale buffers — mixing
them would have one branch borrowing the other's picture. They are told apart by
the text conditioning they carry, which is the thing that actually differs.

Running the two branches on different groups of nodes would nearly halve the
wall-clock time and is the obvious next step. It is not implemented.

## Trying it

```bash
diffuse query --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "a paper boat on a puddle" \
  --steps 20 --patches 4 --seed 7
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--steps` | 20 | denoising steps; more is slower and usually better |
| `--patches` | 4 | pieces each step is cut into across the nodes |
| `--seed` | 0 | same seed and prompt give the same answer |

Raise `--patches` when the link between your nodes is the bottleneck; lower it
when it is not. The answer is written next to you as an `.mp4`, `.wav` or
`.png`, depending on what the model makes.

::: danger Diffusion on CPU is slow
A 1.3B video model at 480p spends minutes per denoising step on a CPU node.
This path is worth using on GPU nodes; on CPU it is worth testing with a small
model and a low step count, and not much else.
:::
