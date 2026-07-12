# Threat Model

This is a short, honest summary. A full formal treatment will appear in the
Diffuse white paper. Do not rely on Diffuse for high-risk use before reading it.

## What Diffuse protects

**Prompt content in transit.** The client tokenizes and runs the first layers
of the model locally. Only intermediate activations leave the machine, never the
prompt tokens in clear text.

**Inter-node confidentiality.** Computation between nodes travels over an
authenticated, end-to-end encrypted channel (X25519 key exchange, then
ChaCha20-Poly1305 AEAD). There is no certificate authority; keys are bound to
node identities.

**Announcement integrity.** Peer announcements in the gossip network are signed
(Ed25519). Forged or tampered peer records are rejected.

**Session unlinkability (partial).** Each client request uses an ephemeral
key-exchange keypair, so two requests are not trivially linkable by a shared key.

## What Diffuse does NOT protect (by default)

**Traffic metadata.** The fact that you are talking, when, how much, and the
shape of your traffic are observable. Diffuse does not run over Tor or a mixnet
by default.

**Network identity / IP.** Your IP address is visible to the nodes you connect
to. Masking it is left to the user (VPN, proxy, or their own Tor).

**Intermediate activations on nodes.** A node computes on the activations it
receives. These are not the prompt in clear, but reconstructing content from
intermediate activations is a known research area. The difficulty is tunable by
how many layers the client keeps locally. It is made hard, not proven impossible.

**Mid-pipeline plaintext.** Computation on each node happens in the clear in
memory (no trusted execution environment or homomorphic encryption).

**Relayed traffic metadata (nodes behind NAT).** A node that cannot accept
inbound connections (behind NAT or a firewall) serves compute through a sentinel
acting as a relay. The relayed compute payload stays end-to-end encrypted between
the client and the serving node, so the relay never sees content. However, the
relay does observe flow metadata for that traffic: which client talks to which
node, timing, and volume. Users who must hide this metadata should serve only
from directly reachable nodes, or place a network anonymity layer beneath
Diffuse.

**Content self-revelation.** If your prompt names you, no system can un-say it.

**Sybil and reputation attacks.** There is no anti-Sybil mechanism yet. A single
actor can run many nodes.

## Adversaries considered

Diffuse aims to protect against a **curious node operator** and a **passive
network observer** with respect to prompt content. It does **not** currently
defend against a **global passive adversary** (traffic analysis across the whole
network) or a **well-resourced actor** performing activation inversion.

## Summary

Diffuse raises the cost of surveilling what you think, within a defined
perimeter. It is not anonymity, and it is not unconditional secrecy. Know the
boundary before you cross it.