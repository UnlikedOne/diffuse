# What is Diffuse

Diffuse runs large language models across a peer-to-peer network of ordinary
machines. No single computer holds the whole model. Each node holds a **slice**
of the model's layers, and a request flows through the slices like a current:
every node does its part, passes transformed numbers to the next, and the answer
returns to you.

There is no central server, no account, and no API key. You install one binary,
run `diffuse chat`, and you are talking to a model that may be spread across
machines on three continents.

## Why it exists

Running a capable model normally means one of two things: renting time on
someone else's servers, where your prompts pass through a company that can log
them, or owning a machine large enough to hold the whole model, which most people
do not have.

Diffuse takes a third path. A model too large for any single machine is split
vertically, layer by layer, across many small ones. A laptop that could never
hold a 32B model can hold three of its layers, and thirty such laptops can serve
it together.

## What makes it different

- **Decentralized.** Nodes discover each other by gossip and self-organize. There
  is no coordinator to run, pay for, or trust.
- **Encrypted in transit.** Every hop between nodes is sealed end to end with
  X25519 key exchange and ChaCha20-Poly1305. Keys are bound to node identities,
  so there is no certificate authority.
- **Self-healing.** When a node disappears, the network notices, and other nodes
  step in to cover the slice that went dark.
- **Inclusive.** A node measures its own memory, takes a slice that fits, and
  contributes what it can. Machines behind NAT still serve through an encrypted
  relay.
- **Open.** The daemon is Rust, the model runner is a Python worker on PyTorch and
  Transformers, and the whole thing is AGPL-3.0.

## What it is not

Diffuse is a prototype, and it is honest about its edges. It does not hide
*that* you are talking, only works with dense text models today, and has real
operational limits. Read the [threat model](/privacy) and the
[limitations](/limitations) before trusting it with anything sensitive.

## Where to go next

- [How it works](/introduction/how-it-works) for the mental model.
- [Installation](/start/installation) to get the binary on your machine.
- [Quickstart](/start/quickstart) to send your first prompt.
