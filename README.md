<div align="center">

# 🌐 Diffuse

**Your own AI. Split across the world. Watched by no one.**

*Run large language models on a network of ordinary machines, peer to peer.
Your prompt never leaves your device in clear text.*

`no servers` · `no surveillance` · `no logs`

</div>

Diffuse turns a crowd of ordinary machines into one mind. Alone, none of them can run a large model. Together, they can. Each machine holds a slice of the model's layers, and your request flows through the network as encrypted math, never as words on someone else's server.

No accounts. No company. No one holding the keys but you.

```
diffuse chat --bootstrap <sentinel>

  ›  who watches this conversation?
     No one. Your prompt was tokenized on your machine,
     ran its first layers locally, and left as encrypted
     activations. The network computed. It never saw your words.
```

## 💭 Why this exists

The last few years made something clear that used to be theory: **the AI you talk to answers to someone, and it is not you.**

In 2025 the U.S. Department of Defense signed contracts worth up to $200 million each with Anthropic, Google, OpenAI, and xAI to bring frontier AI into national security work. [[1]](https://www.congress.gov/crs-product/IN12669) The Pentagon pushed for access to these models for "all lawful purposes," including uses the companies themselves flagged: **mass domestic surveillance and fully autonomous weapons.** [[2]](https://en.wikipedia.org/wiki/Anthropic%E2%80%93United_States_Department_of_Defense_dispute) Most complied. When one refused those two lines, the administration moved to cut it off entirely. [[3]](https://edition.cnn.com/2026/02/27/tech/anthropic-pentagon-deadline)

Whatever side you take, the lesson is structural: **when your thinking runs on someone else's servers, the terms of that thinking are set by contracts, politics, and power you never see.** A model deployed on classified military networks was reportedly used in real operations. [[1]](https://www.congress.gov/crs-product/IN12669) The pipe between you and the machine is never neutral.

More than a decade earlier, the Snowden disclosures showed the same shape at a different scale: programs like PRISM collected user data directly from the servers of the largest tech companies. [[4]](https://en.wikipedia.org/wiki/PRISM) The pattern does not change. Only the model does.

Diffuse is the refusal of that pattern.

It is for the journalist who cannot trust the cloud. The doctor bound by confidentiality. The researcher under a regime that watches. And for anyone who simply believes that **thinking should be private by default.**

The network belongs to the people running it. That is the entire point.

## ✨ What it does

- 🧩 **Runs the impossible.** Models too large for any single machine run across many small ones, split vertically by layers.
- 🔄 **Organizes itself.** No master, no server. Nodes find each other through gossip, measure their own hardware, and take the slice the network needs most. When one falls, the others heal the gap.
- 🔒 **Keeps your words home.** Your device tokenizes and runs the first layers itself. What leaves is transformed math, not your prompt.
- 🛡️ **Sealed in transit.** Every hop between nodes is end to end encrypted with X25519 and ChaCha20-Poly1305. No certificate authority, no middleman.
- 💻 **Welcomes any machine.** An old laptop gives what it can. A workstation gives more. Diffuse measures each model's real memory cost and hands out slices that actually fit.

## 🚀 Start

Lend your machine to the network:

```
diffuse host --model Qwen/Qwen2.5-0.5B-Instruct --spawn-worker
```

Talk to the network:

```
diffuse chat --bootstrap <sentinel>
```

See what is alive out there:

```
diffuse models --bootstrap <sentinel>
```

## 🔬 How it works

Three planes hold the system together:

- **Control plane** decides who is alive, who holds what, and where a new node is most useful.
- **Data plane** moves activations through the pipeline and caches them for speed.
- **Trust plane** carries identity and the encryption that seals every hop.

The daemon and coordination logic are written in Rust. Model execution runs in a separate Python worker on PyTorch and Transformers. See [`DESIGN.md`](DESIGN.md).

## ⚖️ Honest limits

Diffuse hides **what you say.** Your prompt never leaves you in clear text, and the math is encrypted between nodes.

It does **not** hide **that you are talking,** it does not mask your IP by default, and a node still sees the raw activations it is asked to compute. Turning those activations back into your words is hard, and you decide how hard by keeping more layers on your side. But it is not magic, and this project will never pretend it is.

Before you trust Diffuse with something that matters, read [`THREAT_MODEL.md`](THREAT_MODEL.md). It states plainly what is protected, from whom, and what is not. No marketing. No lies.

## 🌱 Status

A working prototype. Distributed generation, self healing, decentralized discovery, encrypted routing, hardware aware slicing, and a live chat client are built and tested. It is rough in places. It is alive.

## 📜 License

AGPL-3.0-or-later. Free, and it stays free.

## 🔗 Sources

1. Congressional Research Service, *Pentagon-Anthropic Dispute over Autonomous Weapon Systems*, congress.gov/crs-product/IN12669
2. *Anthropic–United States Department of Defense dispute*, Wikipedia
3. CNN Business, *Trump administration orders military contractors and federal agencies to cease business with Anthropic*, Feb 2026
4. *PRISM (surveillance program)*, Wikipedia

<div align="center">

*Inspired by the peer-to-peer spirit of BitTorrent and by Petals.
Built for a world where private thought should not require permission.*

</div>
