# Diffuse: Design & Decisions

Single source of truth for the project. Every architecture decision is frozen here.

## Identity

Diffuse is a distributed, privacy-first, sovereign LLM inference protocol.
One Core, one binary, from local mode to the internet by activating layers.

- Author: UnlikedOne
- License: AGPL-3.0-or-later
- Inspiration: Petals (BigScience). Concepts reimplemented, no shared source code.

## Guiding principle: separation into three planes

- Control plane: who is alive, who holds which layers, routing.
- Data plane: activation flow during inference.
- Trust plane: attestation, reputation, incentives (active in open mode).

The data plane is identical in local and public modes. Only the trust perimeter
changes. Each plane works without the plane above it.

## Languages

- Core (daemon, control, data, trust, api, contribution): Rust.
- Inference worker: Python (PyTorch/transformers), separate process.
- Rust to Python boundary: gRPC, defined in proto/.
- The Python worker is replaceable by Rust/candle later without a rewrite.

## Fault tolerance

- k=2 replication of layer slices.
- Route built at the last moment among live nodes.
- Failover by simple replay first; activation checkpointing later.
- Voluntary departure is a graceful announced departure.
- Graceful degradation to a small local model when coverage is insufficient.

## Bootstrap

- Network: sentinels (bootstrap nodes, --bootstrap flag).
- Model: continuous degraded scale. Solo (mini model) to 2 nodes (medium model,
  fragile k=1) to more nodes (large model, reliable k=2).
- The Core never refuses to start; it offers the best possible model.

## Multi-model

- One sub-network per model.
- Hosting and querying are chosen separately.
- A model is servable only if all its slices are covered.

## Modalities

- Whatever a model accepts on the way in, whatever it produces on the way out.
  Text is one case, not the shape of the system.
- Media never travels raw. It becomes activations where it is owned: the
  encoder tower rides with the slice holding the embeddings, which on a client
  is its own machine. A node receives the same transformed numbers a text
  prompt would produce.
- Output travels as tokens, never as decoded bytes. Kilobytes instead of
  megabytes, and the picture or the waveform only ever exists on the machine
  that asked for it.
- The detokenizer (VQ decoder, vocoder) is a client concern, symmetric to the
  encoder tower. A node never holds the finished answer.
- An answer is typed: text, or bytes with a media type. The chat, the CLI and
  the OpenAI facade all carry that type through.
- A model may emit several token streams per step (audio codebooks). The last
  slice returns K streams; text is K=1, not a special case.

## Computation shapes

The data plane assumes one invariant: an ordered stack of homogeneous layers,
one tensor between two layers, state carried per session. A model is supported
when its computation has one of the accepted shapes, not when it is on a list.

Accepted:
- Autoregressive. One step is one traversal, state is the KV cache.
- Encoder memory. The decoder reads an encoder's output at every layer
  (Whisper and the encoder-decoder family). That memory is a session-scoped
  side input, like the cache: sent once, held for the session.

Deferred:
- Iterative refinement (DiT diffusion: Flux, SD3, PixArt). The stack is
  homogeneous and does slice, but an image costs 20 to 50 full traversals
  instead of one per token, and there is no cache to carry. Worth it on GPU
  nodes only. Deferred on cost, not refused on principle.

Refused:
- U-Net diffusion (SDXL and its family). Skip connections across resolutions
  mean the interface between two points is several tensors at different scales
  rather than one. It does not fit a chain; forcing it would mean a second data
  plane.
- trust_remote_code. Arbitrary Python on volunteer machines contradicts the
  promise the network is built on. A security decision, not a technical one.

## Model descriptor

- What a model eats and produces is derived from its Hugging Face config, never
  from a hand-kept registry: layer count from the text sub-config, encoder tower
  from the module names, output kind from the head.
- A model that cannot be sliced is refused with the reason, in the catalogue,
  rather than failing at load time.
- An unknown architecture is offered as untested rather than hidden. The
  network learns what works by being used.

## Contribution

- User-controlled: max / moderate / min / pause.
- Opportunistic: automatic withdrawal when the machine is used for other tasks.
- Background daemon (systemd / launchd / Windows service).
- --client-only mode: consume without hosting.

## Privacy

- Cleartext compute on the node.
- TEE as the safeguard in open mode; mutual trust in private mode.
- KV-cache in volatile memory, never on disk, wiped at session end.
- Crypto: Ed25519 identity + SHA-256 integrity; Noise transport via libp2p;
  TEE for execution.
- No Shamir, no ZKP.
- Honest framing: surveillance impossible at rest, costly at execution,
  residual metadata.

## Incentives

- Usage credit (contribution to consumption) at first, no token.
- Pure consumer: small initial credit, slow regeneration.
- DIF token deferred (Layer 3), aware of regulatory risk (MiCA).

## History vault

- Local, optional, encrypted (Argon2 + XChaCha20-Poly1305).
- Never synced, can be disabled, ephemeral mode possible.
- Key distinct from the node identity.
- vault module in the client facade.

## Model download

- Worker downloads from the Hugging Face Hub.
- PoC: full toy model, loads the assigned slice.
- Later: partial safetensors loading of only the assigned layers.
- Optional HF token field in the worker config (gated models such as Llama).
  Token is local to the node, never sent over the network.
- Layer 2: P2P distribution of weights with signed-hash verification.
- Preference for fully open models (Qwen, Mistral) for consistency.

## Deployment modes

- One binary. Flags: --mode private|public, --bootstrap, --client-only.
- Private mode by default. Public mode activates the trust plane.

## Delivery layers

- Layer 1 (v0.1): distributed inference, private/team mode. No token, no
  blockchain, optional TEE. This is the first push.
- Layer 2: trust plane (TEE attestation, reputation, activation encryption,
  anti-surveillance).
- Layer 3: economic plane (DIF token, staking, slashing, L2 settlement).

## Future work (do not code before the PoC)

- Weight integrity via signed hash (can be pulled into late Layer 1).
- Anti-Sybil (open mode).
- Probabilistic compute verification by sampling.
- Session recovery after client disconnect.
- Protocol versioning from the first commit.
- Observability (Prometheus metrics).
- Excluded: distributed fine-tuning, polished UI.

Multimodality was excluded here originally. That decision is reversed: it cost
nothing in the data plane, because a multimodal checkpoint is the same stack of
layers with an encoder tower bolted to the front, and the tower is an endpoint
concern. See Modalities and Computation shapes above.

## Build order

1. Python worker: slice inference on one machine, toy model.
2. Rust to worker transport, local.
3. Two Rust daemons talking to each other.
4. Orchestrator + failover.
5. OpenAI-compatible API facade.
6. Rest (auto scheduler, distributed KV-cache, vault, contribution).

PoC: 3 Docker containers, k=1, tiny model. Validates logic, not performance.

## README (to be written later)

Emotional speech, measured claims:
- PRISM / Snowden: centralization as a surveillance point.
- Anthropic-Pentagon conflict: privacy must not depend on a CEO or a contract.
- Access to AI without a paywall.
- Reduced water/energy footprint (not eliminated).
Through line: access and privacy guaranteed by architecture, not by goodwill.
Figures to be sourced at writing time.