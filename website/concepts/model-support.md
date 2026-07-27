# What Diffuse can run

Diffuse does not keep a list of approved models. It looks at the shape of a
model's computation and decides whether that shape can be cut into pieces that
run on different machines. This page explains the rule, so you can predict the
answer for a model nobody has tried.

## The one thing that has to be true

Diffuse assumes a model is **an ordered stack of near-identical layers**, where
what passes between two layers is a single tensor, and where per-request state
belongs to the layer that made it.

That is what makes a cut possible anywhere. Everything else — embeddings,
encoders, output heads, codecs — sits at the ends, and the ends belong to the
machine that asked the question.

## The three shapes

**Autoregressive.** One step is one pass through the stack; state is the
key-value cache. This is every ordinary chat model. Fully supported.

**Encoder memory.** The stack reads a second stack's output at every layer.
Whisper does this: an audio encoder produces a memory, and the decoder consults
it while it writes words. Supported. The encoder runs on your machine and its
output travels once per session, held by each node the way the cache is.

**Iterative refinement.** The whole stack runs twenty to fifty times to produce
one result, with no cache carried between passes. This is modern image
diffusion — Flux, SD3, PixArt. Their stacks *are* homogeneous transformers and
would slice, but a picture costs fifty full traversals instead of one per
token, which is only worth the network cost on GPU nodes. **Deferred by
decision, not refused on principle.**

## What is refused, and why

**U-Net diffusion** (Stable Diffusion XL and its family). Its skip connections
cross resolutions, so what passes between two points is several tensors at
different scales rather than one. It does not fit a chain. Serving it would
mean a second data plane, which is a different project.

**Recurrent stacks** (Mamba, RWKV). Their layers carry a running state instead
of a key-value cache. The shape is otherwise ideal for splitting — the state is
a fixed size, which is friendlier than a growing cache — but one slice cannot
currently hand that state to the next. This is an engineering gap, not a
barrier.

**Encoders alone** (BERT and relatives). They do not generate. They would slice
happily; there is simply no answer to stream back yet.

**`trust_remote_code`.** A checkpoint that ships its own Python. Running a
stranger's code on volunteer machines contradicts the promise the network is
built on. This is a security decision and it is not going to change.

## How the verdict is reached

For any model, Diffuse fetches `config.json` and reads:

| Question | Where it looks |
|----------|----------------|
| How deep is the stack? | `num_hidden_layers`, wherever the checkpoint keeps it |
| Does it generate? | the architecture name's suffix |
| Does it read media? | `vision_config`, `audio_config`, and placeholder token ids |
| Does it read an encoder? | `is_encoder_decoder` |
| Does it carry recurrent state? | `state_size`, `conv_kernel`, and friends |

The depth is the subtle one. A plain decoder declares it at the root. A
multimodal wrapper hides it under `text_config`. MusicGen keeps it under
`decoder`, next to an encoder of its own. Each family invents a name, so the
known ones are tried in order and then any sub-config that declares a depth,
with encoder sections skipped.

That last exclusion is not a detail. MusicGen's T5 encoder has twelve blocks
and its decoder has twenty-four; measuring the wrong one would make a node
claim a slice that does not exist. Whisper is worse: twelve layers on each
side, so a search that did not know an encoder when it saw one would split the
half that reads instead of the half that answers, and produce confident
nonsense with no error at all.

## Where the ends live

A slice in the middle of a model holds layers and nothing else. The two ends
are special, and so is your own machine.

| Piece | Lives on | Why |
|-------|----------|-----|
| Embedding table | the first slice | it turns tokens into the first hidden states |
| Encoder tower (vision, audio) | the client, and the first slice | media must be consumed where it is owned |
| Output head | the last slice | it turns hidden states into logits |
| Codec (audio, image decoder) | the client | the finished answer belongs to the asker |

A node holding layers 12 to 24 of a vision model never downloads the vision
tower. It would be dead weight: no media ever reaches it.

## Verified, not assumed

Every claim of support in these docs was checked by running the model whole on
one machine and then again split across slices, and comparing.

| Model | Shape | Result |
|-------|-------|--------|
| Qwen2.5 | autoregressive | identical text |
| SmolVLM | autoregressive + vision tower | identical caption |
| Qwen2-VL | autoregressive + vision, video | identical answer, token for token |
| Voxtral | autoregressive + audio tower | correct transcript |
| Whisper | encoder memory | identical transcript |
| MusicGen | encoder memory + 4 output streams | identical codes, token for token |

"Identical" means the sliced pipeline produced the same tokens as the same
model running whole, not merely a plausible answer. That is the property
Diffuse owes you: **splitting a model must not change what it says.**
