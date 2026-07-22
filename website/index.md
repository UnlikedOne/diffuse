---
layout: home
title: Diffuse
titleTemplate: Your own AI, watched by no one

hero:
  name: Diffuse
  text: Your own AI. Split across the world.
  tagline: Large language models running on a peer-to-peer network of ordinary machines. Your prompt never leaves your device in clear text.
  image:
    src: /logo.svg
    alt: Diffuse
  actions:
    - theme: brand
      text: Get started
      link: /start/installation
    - theme: alt
      text: How it works
      link: /introduction/how-it-works
    - theme: alt
      text: GitHub
      link: https://github.com/UnlikedOne/diffuse

features:
  - icon: 🧠
    title: Runs the impossible
    details: Models too large for one machine run across many small ones, split vertically by layers. An old laptop gives what it can, a workstation gives more.
  - icon: 🕸️
    title: Organizes itself
    details: No master, no server. Nodes find each other by gossip, measure their own hardware, and take the slice the network needs most. When one falls, the others heal the gap.
  - icon: 🔒
    title: Keeps your words home
    details: Your device tokenizes locally and traffic is sealed end to end with X25519 and ChaCha20-Poly1305. No certificate authority, keys bound to node identities.
  - icon: 🌐
    title: Reaches every contributor
    details: Nodes behind NAT or a firewall serve through an encrypted relay, so a home machine can contribute, not only consume.
  - icon: ⚡
    title: Streams in real time
    details: Answers appear token by token as the network generates them, not in one delayed block. Works from the CLI or any OpenAI-compatible client.
  - icon: 🧩
    title: OpenAI-compatible
    details: A local server speaks the OpenAI API, so LibreChat, Open WebUI, Continue and the official SDKs work against the network with no code changes.
---

<div class="dx-tags">
  <span class="dx-tag"><strong>no</strong> servers</span>
  <span class="dx-tag"><strong>no</strong> surveillance</span>
  <span class="dx-tag"><strong>no</strong> prompt logging</span>
  <span class="dx-tag">built with <strong>Rust</strong> + <strong>PyTorch</strong></span>
  <span class="dx-tag">license <strong>AGPL-3.0</strong></span>
</div>

<div class="dx-marquee">
  <p class="dx-marquee__label">Built on and works with</p>
  <div class="dx-marquee__track">
    <span class="dx-chip">Rust</span>
    <span class="dx-chip">tokio</span>
    <span class="dx-chip">PyTorch</span>
    <span class="dx-chip">Transformers</span>
    <span class="dx-chip">Hugging Face</span>
    <span class="dx-chip">X25519</span>
    <span class="dx-chip">ChaCha20-Poly1305</span>
    <span class="dx-chip">Ed25519</span>
    <span class="dx-chip">LibreChat</span>
    <span class="dx-chip">Open WebUI</span>
    <span class="dx-chip">Continue</span>
    <span class="dx-chip">OpenAI API</span>
    <span class="dx-chip">Rust</span>
    <span class="dx-chip">tokio</span>
    <span class="dx-chip">PyTorch</span>
    <span class="dx-chip">Transformers</span>
    <span class="dx-chip">Hugging Face</span>
    <span class="dx-chip">X25519</span>
    <span class="dx-chip">ChaCha20-Poly1305</span>
    <span class="dx-chip">Ed25519</span>
    <span class="dx-chip">LibreChat</span>
    <span class="dx-chip">Open WebUI</span>
    <span class="dx-chip">Continue</span>
    <span class="dx-chip">OpenAI API</span>
  </div>
</div>

<DxNetwork />
