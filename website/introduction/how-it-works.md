# How it works

A transformer processes text by turning tokens into vectors and passing them
through a tall stack of identical layers. Diffuse cuts that stack into contiguous
ranges and hands each range to a different machine.

```
   your device                 the network                  your device
  ┌───────────┐   activations  ┌───────┐   ┌───────┐   logits   ┌───────────┐
  │ tokenize  │───(encrypted)─▶│ node  │──▶│ node  │──(enc.)──▶ │  decode   │
  │           │                │ 0-14  │   │ 14-24 │            │  the token│
  └───────────┘                └───────┘   └───────┘            └───────────┘
   prompt starts here      blind, encrypted middle         answer forms here
```

## The flow of one token

1. **Tokenize locally.** Your machine turns your text into token ids using the
   model's tokenizer. This happens on your device.
2. **Enter the pipeline.** The request is sent, encrypted, to the node holding the
   first slice. That node embeds the tokens and runs its layers.
3. **Hop through the middle.** Each node runs its slice on the activations it
   receives, then passes the result to the node holding the next range. Every hop
   is sealed end to end.
4. **Return the logits.** The last slice produces the output distribution, which
   comes back to your device.
5. **Decode and repeat.** Your machine picks the next token and feeds it back
   through the pipeline until the answer is complete.

A KV cache on each node keeps the attention state between tokens, so the pipeline
does not reprocess the whole prompt every step.

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
