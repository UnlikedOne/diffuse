# Roadmap

What is being worked on and what is planned, kept honest. This is a prototype, so
the roadmap is about closing real gaps rather than adding polish.

## In progress

### Strong private mode

Keep the embedding and the first layers on your own machine, so that what leaves is
a hidden state rather than your token ids and no node ever receives the prompt.
This is opt-in by design, because it requires the client to hold the head of the
model. It is the change that makes the strong privacy claim true on the default
path. See the [threat model](/privacy).

## Planned

### Measurement

- A single-machine reference run to confirm the distributed pipeline matches a
  non-distributed baseline token for token.
- The relay path cost, which adds two network legs per token and is the common case
  for peer-to-peer.
- A heterogeneous network mixing CPU and GPU nodes.

### Resilience

- Relayed nodes reattaching to another sentinel when theirs goes down, removing the
  single point of failure.
- Sending ClearSession to relayed peers so their KV cache is freed promptly.
- Admission control on the relay so capacity is not consumed without contribution.

### Reach

- Support for more model families beyond dense text transformers.
- Loading weights from sources other than the Hugging Face Hub, with a canonical
  model id separate from the weights source.
- Multiple nodes per machine, so a single machine can run more than one node or
  concurrent client.

## How to follow along

The honest current state of every known gap lives on the
[limitations](/limitations) page, and measured performance on the
[benchmarks](/benchmarks) page. Releases and changes are on
[GitHub](https://github.com/UnlikedOne/diffuse/releases).
