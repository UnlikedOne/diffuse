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

## What it costs, measured on a real model

Substituting one step's activations for another's is an approximation, so the
question is how much the picture moves. Measured on **Wan2.1-T2V-1.3B**, thirty
blocks split over two stages, 256×256, nine frames, twenty denoising steps,
against the same model run whole on one machine:

| Patches per step | Difference from the whole model |
|---|---|
| 1 | **none — bit-identical** |
| 2 | mean **36.5 / 255** |
| 4 | mean **52.8 / 255** |

Read the first row first: with a single patch there is no stale data to use, and
the answer is **byte-for-byte** what the unsliced model produced. That is the row
that says the machinery is right rather than merely plausible — the slicing, the
plan that carries the call, the client-side ends, the encryption, all of it. A
model cut across machines costs nothing in fidelity.

Now read the other rows, and be careful about what those numbers mean. Thirty-six
levels of mean error sounds like a ruined picture. It is not. Side by side, the
patched runs are **the same video**: the same paper boat, at the same angle, in
the same rain, rippling the same way. What moves is tone. The colours come out
warmer and more saturated, the soft shading on the paper hardens, and the
reflection in the water changes character. The prompt is followed just as well;
the picture is graded differently.

So the honest summary is a visible quality cost, not a failure — but a cost you
should choose deliberately, which is why `--patches` defaults to **1**. Twenty
steps is barely better than six (66 / 255 at four patches), so denoising longer
does not buy it back.

Why the tone shifts is worth stating. A video latent is flattened with time as
the outer axis, so cutting the sequence into contiguous pieces cuts it mostly
**along time**: each patch is roughly its own frame, and every patch attends to
the other frames as they were one step ago. PipeFusion was published on image
transformers, where a patch is a region of one picture and the neighbouring
regions really do barely move. On video the same trick still holds the scene
together, but it drifts the global statistics of the latent, and the VAE turns
that drift into colour.

::: warning Earlier numbers here were worthless
This table used to report about a third of a greyscale level at sixteen patches.
That was measured on a four-block toy transformer with random weights, where
there is no picture to damage. It said nothing about a real model, and it was
presented as though it did.
:::

::: warning The first step is never cut
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

- **The block stacks** are whichever `ModuleList`s of transformer blocks the
  model exposes. Note the plural: Flux and HunyuanVideo run a joint stack and
  then a single-stream stack, and both are split.
- **How long each stack is** comes from the model's own config, but the config
  does not say which setting governs which stack — `num_layers`,
  `num_single_layers`, `depth`, and the names differ per family. Each integer
  setting is nudged by one on a weightless copy of the model and the stack that
  grew is the one it controls. Nothing to keep up to date.
- **The two ends** are run by calling the pipeline's own code with the block
  stacks replaced by stand-ins. The pipeline prepares latents, conditions on the
  timestep, runs its scheduler and decodes with its VAE exactly as it always
  does; Diffuse only intercepts the stacks in the middle. No family-specific
  logic is reimplemented, so nothing drifts out of date.
- **What the model answers with** comes from its own parts: a VAE that declares
  a sampling rate means audio, a patch size with a time axis means video,
  anything else is a picture.

### What a block is handed, and what it gives back

This is where being generic is actually hard. Diffusers calls its blocks in
whichever style the family was written in: Wan passes everything positionally,
Flux, SD3, LTX and Mochi pass by keyword, and the arguments are not all tensors
— there are floats, integers and dictionaries in there.

So a call is taken apart into a **plan**: which positions and which names it
carried, which of them were tensors, and which small constants sit between them.
The plan travels as JSON with the first call of each branch and is remembered by
the node; only tensors that changed are sent again.

The other half is subtler. A block returns one tensor for some families and two
for others — Flux, SD3, CogVideoX, Mochi and HunyuanVideo carry an image stream
and a text stream side by side through their blocks, and the order they come
back in differs between them. Nothing declares this. It is learned instead: one
block is run once, locally, and each tensor it returned is matched to the input
it replaces. That mapping goes into the plan, so the node knows how to thread
its loop.

::: warning Two streams mean whole steps
A model that carries two streams through its blocks is not cut into patches:
splitting the image stream would give each patch its own version of the text
stream, and there is only one right answer. Those models are still split across
nodes block by block, which is exact — the step simply stays whole. The node
says so when the generation starts and the client obeys, whatever `--patches`
asked for.
:::

### Attention over a patch, checked rather than assumed

Attention is the one place where being generic and being fast pull apart. The
plain way is to hand the model's own attention the whole stale buffer and take
the patch's rows back out of what it returns. That is right for every model,
because it changes nothing the model does — but it computes the whole picture's
attention for every patch.

The quick way asks only for the patch's queries against the buffer's keys, which
is what makes patching worth doing. It has to slice the rotary positions to the
patch, and that means assuming how this family lays its rotary out.

Diffuse does not take that on faith. On the first real patch of a session both
are computed and compared; the quick one is kept only if it agreed with the
plain one to a part in ten thousand. A family whose rotary convention is not the
one assumed silently gets the correct path instead of a wrong picture.

A model that follows these conventions works without anyone adding it to a list.
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
  --steps 20 --seed 7
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--steps` | 20 | denoising steps; more is slower and usually better |
| `--patches` | 1 | pieces each step is cut into across the nodes |
| `--seed` | 0 | same seed and prompt give the same answer |

Leave `--patches` at 1 unless the link between your nodes is the bottleneck.
Raising it keeps the scene and costs you tone, which is a trade worth making
when bandwidth is what you are short of. The answer is written next to you as an
`.mp4`, `.wav` or `.png`, depending on what the model makes.

::: danger Diffusion on CPU is slow
A 1.3B video model at 480p spends minutes per denoising step on a CPU node.
This path is worth using on GPU nodes; on CPU it is worth testing with a small
model and a low step count, and not much else.
:::
