# How it works

A transformer processes text by turning tokens into vectors and passing them
through a tall stack of identical layers. Diffuse cuts that stack into contiguous
ranges and hands each range to a different machine.

```
   your device                 the network                  your device
  ┌───────────┐   activations  ┌───────┐   ┌───────┐  candidates ┌───────────┐
  │ tokenize  │───(encrypted)─▶│ node  │──▶│ node  │──(enc.)──▶  │  decode   │
  │           │                │ 0-14  │   │ 14-24 │             │ the token │
  └───────────┘                └───────┘   └───────┘             └───────────┘
   prompt starts here      blind, encrypted middle          answer forms here
```

## The flow of one token

1. **Tokenize locally.** Your machine turns your text into token ids using the
   model's tokenizer. This happens on your device. If you attached a picture or
   a recording, your machine also runs the model's encoder and turns it into
   hidden states — the media itself never leaves.
2. **Enter the pipeline.** The request goes, encrypted, to the node holding the
   first slice, carrying the route for the rest. That node embeds the tokens and
   runs its layers.
3. **Pass along the chain.** Each node runs its slice, encrypts the result for
   the next node, and hands it over directly. The answer travels back down the
   chain. Only the first and last trips touch your connection, whatever the
   number of nodes.
4. **Return the best candidates.** The last slice does not send back a score for
   every word it knows — that would be hundreds of kilobytes per token. It sends
   the few best, which is all your machine reads.
5. **Decode and repeat.** Your machine picks the next token and feeds it back
   until the answer is complete.

A key-value cache on each node keeps the attention state between tokens, so the
pipeline does not reprocess the whole prompt every step. If a node dies mid
answer, the surviving replicas hold no cache for your session, so Diffuse clears
it and replays what was said so far rather than continuing from a blank state
and quietly producing nonsense.

## When the answer is not text

A model that answers with sound emits several streams of tokens per step
instead of one. Those tokens come back to your machine, and your machine turns
them into a waveform with the model's own codec. A node never holds the
finished file, exactly as it never held the picture you sent in. See
[images, audio and video](/guides/multimodal).

## Roles a machine can play

- **Client.** Runs `diffuse chat`, `query`, or `serve`. Tokenizes locally and
  drives the generation loop, but holds no model weights by default.
- **Host.** Runs `diffuse host`. Holds a slice of a model, announces it to the
  network, and serves compute to clients. This is how the network exists.
- **Sentinel.** A well-known node used for bootstrap discovery and, when needed,
  as an encrypted relay for peers behind NAT.

A single machine can be several of these at once.

## Three planes

| Plane | Responsibility |
|-------|----------------|
| **Control** | liveness, gossip discovery, capacity analysis, and deciding where a new node is most useful |
| **Data** | activation flow through the pipeline, with a KV cache for speed |
| **Trust** | node identity and the encryption that seals every hop |

The control, gossip, orchestration, and encrypted transport are written in Rust.
Model execution runs in a separate Python worker on PyTorch and Transformers,
isolated from the network behind a local boundary.

## Next

- [Slices and the pipeline](/concepts/pipeline)
- [Gossip and discovery](/concepts/gossip)
- [Trust and encryption](/concepts/trust)
