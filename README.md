<div align="center">

<img src="https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square" alt="License"> <img src="https://img.shields.io/badge/status-prototype-orange?style=flat-square" alt="Status"> <img src="https://img.shields.io/badge/platform-linux%20x86__64-lightgrey?style=flat-square" alt="Platform"> <img src="https://img.shields.io/badge/built%20with-Rust%20%2B%20PyTorch-informational?style=flat-square" alt="Stack">

# Diffuse

<img src="assets/diffuse-chat.png" alt="Diffuse chat" width="720">

### Your own AI. Split across the world. Watched by no one.

Large language models running on a peer-to-peer network of ordinary machines.
Your prompt never leaves your device in clear text.

<code>no servers</code> · <code>no surveillance</code> · <code>no prompt logging</code>

[Install](#install) · [Quickstart](#quickstart) · [How it works](#how-it-works) · [Architecture](#architecture) · [Threat model](#honest-limits) · [Documentation](https://unlikedone.github.io/diffuse/)

</div>

---

> The last few years made something clear that used to be theory:
> **the AI you talk to answers to someone, and it is not you.**

In 2025 the U.S. Department of Defense signed contracts worth up to $200 million each with Anthropic, Google, OpenAI, and xAI to bring frontier AI into national security work. The Pentagon sought access for broad purposes, including uses the companies themselves had flagged as dangerous: mass domestic surveillance and autonomous weapons. Most complied. When one refused those specific lines, the administration moved to cut it off entirely.

More than a decade earlier, the Snowden disclosures had already shown the shape of it: programs that pulled user data straight from the servers of the largest tech companies. The pattern does not change. Only the model does.

**Diffuse is the refusal of that pattern.**

It is for the journalist who cannot trust the cloud. The doctor bound by confidentiality. The researcher under a regime that watches. And for anyone who simply believes that thinking should be private by default.

The network belongs to the people running it. That is the entire point.

---

## Install

One command. Nothing to build — the binary is prebuilt for your machine.

**Linux and macOS** (Intel or Apple Silicon):

```bash
curl -fsSL https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.sh | bash
```

**Windows** (PowerShell):

```powershell
irm https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.ps1 | iex
```

This downloads the `diffuse` binary, checks its published checksum, sets up the Python worker, and puts everything in place. No account, no API key, no server of your own, and no Rust toolchain.

| Platform | Binary |
|---|---|
| Linux x86_64 | `diffuse-linux-x86_64` |
| Linux aarch64 (ARM) | `diffuse-linux-aarch64` |
| macOS Apple Silicon | `diffuse-macos-aarch64` |
| macOS Intel | `diffuse-macos-x86_64` |
| Windows x86_64 | `diffuse-windows-x86_64.exe` |

Anything else builds from source with `cargo build --release`.

> **Prefer to read before you run?** The scripts are [`install.sh`](install.sh) and [`install.ps1`](install.ps1). Read one, then run it.

---

## Quickstart

```bash
# Talk to the network
diffuse chat

# See which models are live
diffuse models

# Ask a single question
diffuse query --prompt "Explain black holes in two sentences."
```

That is all. Zero configuration: `diffuse chat` joins the network, finds a model, and streams the answer back token by token.

---

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

The prompt is consumed on your machine. Only transformed activations travel the network, sealed end to end. The model never sees your words; the network never sees your prompt.

---

## Features

| | |
|---|---|
| **Runs the impossible** | Models too large for one machine run across many small ones, split vertically by layers. |
| **Organizes itself** | No master, no server. Nodes find each other by gossip, measure their own hardware, and take the slice the network needs most. When one falls, the others heal the gap. |
| **Keeps your words home** | Your device tokenizes and runs the first layers itself. What leaves is transformed math, not your prompt. |
| **Sealed in transit** | Every hop between nodes is end-to-end encrypted with X25519 and ChaCha20-Poly1305. No certificate authority; keys are bound to node identities. |
| **Reaches every contributor** | Nodes behind NAT or a firewall serve through an encrypted relay, so a home machine can contribute, not only consume. |
| **Streams in real time** | Answers appear token by token as the network generates them, not in one delayed block. |
| **Welcomes any machine** | An old laptop gives what it can; a workstation gives more. Diffuse measures each model's real memory cost and hands out slices that fit. |

---

## Contribute your machine

Run a node in the foreground (you see the logs, `Ctrl+C` stops it):

```bash
diffuse host --model Qwen/Qwen2.5-0.5B-Instruct
```

By default your node joins the existing network through the built-in sentinels. To join a specific network or your own sentinel, pass `--bootstrap`:

```bash
diffuse host --model Qwen/Qwen2.5-0.5B-Instruct --bootstrap http://your-sentinel:9440
```

### Behind a NAT or firewall?

Diffuse handles it. On startup your node checks whether it can accept inbound connections. If it cannot (home network, CGNAT, a cloud shell), it automatically serves its compute through a sentinel **relay**, so it can still contribute rather than only consume. The relayed compute stays end-to-end encrypted between client and serving node; the relay sees only ciphertext and flow metadata. See [`THREAT_MODEL.md`](THREAT_MODEL.md) for exactly what that exposes.

### Keep it running

To keep contributing after you close the terminal, run it detached:

```bash
nohup diffuse host --model Qwen/Qwen2.5-0.5B-Instruct > ~/.diffuse/host.log 2>&1 &
```

Check on it with `tail -f ~/.diffuse/host.log`, stop it with `pkill diffuse`. For a node that restarts after a reboot, see [`deploy/README.md`](deploy/README.md) (systemd).

---

## Architecture

Three planes hold the system together:

| Plane | Role |
|-------|------|
| **Control** | liveness, discovery, capacity, and where a new node is most useful |
| **Data** | activation flow through the pipeline, with KV-cache for speed |
| **Trust** | identity and the encryption that seals every hop |

The daemon, coordination, gossip, orchestration, and encrypted transport are written in **Rust**. Model execution runs in a separate **Python worker** on PyTorch and Transformers, isolated from the network behind a local boundary. For the full design, see [`DESIGN.md`](DESIGN.md).

---

## Honest limits

Diffuse hides **what you say.** Your prompt never leaves you in clear text, and the math is encrypted between nodes.

It does **not** hide **that you are talking.** It does not mask your IP by default, and a node still sees the raw activations it is asked to compute. Turning those activations back into your words is hard, and you decide how hard by keeping more layers on your side. It is made difficult, not proven impossible.

Before you trust Diffuse with something that matters, read [`THREAT_MODEL.md`](THREAT_MODEL.md). It states plainly what is protected, from whom, and what is not. No marketing. No lies.

---

## Status

A working prototype, running on the public internet. Built and tested:

- distributed generation across machines, split by layer
- self-healing when a node drops, with automatic re-replication
- decentralized discovery by signed gossip
- end-to-end encrypted routing between nodes
- hardware-aware slicing that fits each machine
- NAT relay so contributors behind firewalls can serve
- real-time token-by-token streaming
- a zero-config chat client and one-command installer

Rough in places. Alive.

---

## Sources

- **[Pentagon–Anthropic dispute over autonomous weapons](https://en.wikipedia.org/wiki/Anthropic%E2%80%93United_States_Department_of_Defense_dispute)** — the standoff between a frontier AI lab and the U.S. defense establishment over surveillance and autonomous-weapon use.
- **[CRS: Defense AI contracts](https://www.congress.gov/crs-product/IN12669)** — U.S. Congressional Research Service brief on the $200M-class defense contracts awarded to major AI labs.
- **[Reporting on the cutoff order](https://edition.cnn.com/2026/02/27/tech/anthropic-pentagon-deadline)** — coverage of the administration moving to end business with a lab that refused two uses.
- **[PRISM](https://en.wikipedia.org/wiki/PRISM)** — the Snowden-era program that collected user data directly from major tech companies' servers.
- **[Petals](https://github.com/bigscience-workshop/petals)** — prior art in peer-to-peer LLM inference, and an inspiration for this project.

---

<div align="center">

**Built for a world where private thought should not require permission.**

AGPL-3.0-or-later · free, and it stays free

</div>