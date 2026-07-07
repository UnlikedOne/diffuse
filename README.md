<div align="center">

# Diffuse

### Your own AI. Split across the world. Watched by no one.

**Large language models, running on a peer-to-peer network of ordinary machines.
Your prompt never leaves your device in clear text.**

`no servers`   ·   `no surveillance`   ·   `no logs`

</div>

<br>

> The last few years made something clear that used to be theory:
> **the AI you talk to answers to someone, and it is not you.**

In 2025 the U.S. Department of Defense signed contracts worth up to $200 million each with Anthropic, Google, OpenAI, and xAI to bring frontier AI into national security work. The Pentagon sought access to these models for broad purposes, including uses the companies themselves had flagged as dangerous: mass domestic surveillance and autonomous weapons. Most complied. When one refused those specific lines, the administration moved to cut it off entirely.

More than a decade earlier, the Snowden disclosures had already shown the shape of it: programs that pulled user data straight from the servers of the largest tech companies. The pattern does not change. Only the model does.

**Diffuse is the refusal of that pattern.**

It is for the journalist who cannot trust the cloud. The doctor bound by confidentiality. The researcher under a regime that watches. And for anyone who simply believes that thinking should be private by default.

The network belongs to the people running it. That is the entire point.

<br>

## Install

One command. Linux x86_64.

```bash
curl -fsSL https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.sh | bash
```

This downloads the `diffuse` binary, sets up the worker, and puts everything in place.
Prefer to read before you run? The script is [`install.sh`](install.sh); read it first, then pipe it to bash.

Then just talk to the network:

```bash
diffuse chat
```

No account. No key. No server of your own. Zero configuration.

<br>
## Use it

```bash
# talk to the network
diffuse chat

# see which models are live
diffuse models

# ask a single question
diffuse query --prompt "Explain black holes in two sentences."
```

## Contribute your machine

Run a node in the foreground (you see the logs, Ctrl+C stops it):

```bash
diffuse host --model Qwen/Qwen2.5-0.5B-Instruct
```

To keep contributing after you close the terminal, run it detached with `nohup`:

```bash
nohup diffuse host --model Qwen/Qwen2.5-0.5B-Instruct > ~/.diffuse/host.log 2>&1 &
```

Your node keeps running in the background. Check on it with `tail -f ~/.diffuse/host.log`, and stop it with `pkill diffuse`.

For a node that also restarts after a reboot, see [`deploy/README.md`](deploy/README.md) (systemd).

<br>

## How it works

Machines around the world each hold a **slice** of a model's layers. Your request flows through them like a current: each node does its part, sees only encrypted numbers, and passes it on. The answer returns to you, and only you.

```
   your device                 the network                  your device
  ┌───────────┐   activations  ┌───────┐   ┌───────┐   logits   ┌───────────┐
  │ tokenize  │───(encrypted)─▶│ node  │──▶│ node  │──(enc.)──▶ │  decode   │
  │ layers 0-2│                │ 2-14  │   │ 14-24 │            │  the token│
  └───────────┘                └───────┘   └───────┘            └───────────┘
   prompt stays here        blind, encrypted middle        answer forms here
```

| | |
|---|---|
| **Runs the impossible** | Models too large for one machine run across many small ones, split vertically by layers. |
| **Organizes itself** | No master, no server. Nodes find each other by gossip, measure their own hardware, and take the slice the network needs most. When one falls, the others heal the gap. |
| **Keeps your words home** | Your device tokenizes and runs the first layers itself. What leaves is transformed math, not your prompt. |
| **Sealed in transit** | Every hop between nodes is end-to-end encrypted with X25519 and ChaCha20-Poly1305. No certificate authority. |
| **Welcomes any machine** | An old laptop gives what it can. A workstation gives more. Diffuse measures each model's real memory cost and hands out slices that fit. |

<br>

## Architecture

Three planes hold the system together:

| Plane | Role |
|-------|------|
| **Control** | liveness, discovery, capacity, and where a new node is most useful |
| **Data** | activation flow through the pipeline, with KV-cache for speed |
| **Trust** | identity and the encryption that seals every hop |

The daemon and coordination logic are written in Rust. Model execution runs in a separate Python worker on PyTorch and Transformers. See [`DESIGN.md`](DESIGN.md).

<br>

## Honest limits

Diffuse hides **what you say.** Your prompt never leaves you in clear text, and the math is encrypted between nodes.

It does **not** hide **that you are talking,** it does not mask your IP by default, and a node still sees the raw activations it is asked to compute. Turning those activations back into your words is hard, and you decide how hard by keeping more layers on your side. It is made difficult, not proven impossible.

Before you trust Diffuse with something that matters, read [`THREAT_MODEL.md`](THREAT_MODEL.md). It states plainly what is protected, from whom, and what is not. No marketing. No lies.

<br>

## Status

A working prototype. Distributed generation, self-healing, decentralized discovery, encrypted routing, hardware-aware slicing, and a live chat client are built, tested, and running on the public internet. Rough in places. Alive.

<br>

## Sources

- **[Pentagon–Anthropic dispute over autonomous weapons](https://en.wikipedia.org/wiki/Anthropic%E2%80%93United_States_Department_of_Defense_dispute)** — overview of the standoff between a frontier AI lab and the U.S. defense establishment over surveillance and autonomous-weapon use.
- **[CRS: Defense AI contracts](https://www.congress.gov/crs-product/IN12669)** — U.S. Congressional Research Service brief on the $200M-class defense contracts awarded to major AI labs.
- **[Reporting on the cutoff order](https://edition.cnn.com/2026/02/27/tech/anthropic-pentagon-deadline)** — news coverage of the administration moving to end business with a lab that refused two uses.
- **[PRISM (surveillance program)](https://en.wikipedia.org/wiki/PRISM)** — the Snowden-era program that collected user data directly from major tech companies' servers.
- **[Petals](https://github.com/bigscience-workshop/petals)** — prior art in peer-to-peer LLM inference, and an inspiration for this project.

<br>

<div align="center">

**Built for a world where private thought should not require permission.**

AGPL-3.0-or-later · free, and it stays free

</div>