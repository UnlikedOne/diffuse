# Diffuse

Distributed peer-to-peer LLM inference where your prompt never leaves your
device.

Diffuse runs a large language model across several machines that do not trust
each other. The model is split vertically by layers, and each machine holds only
a slice. Your own machine always runs the first layers, so the raw tokens of
your prompt never leave it. Every other node sees only an encrypted hidden
state.

This is the property the whole project is built around. It is the difference
between Diffuse and other distributed inference systems, where peers can see the
prompt.

## Status

The current release is v0.2.0. It ships real token by token streaming in the
chat, latency instrumentation that separates compute from network, and
persistent connections that cut per token latency from 498 ms to 335 ms.

Serving a model from a machine with a public IP works. Serving from behind NAT
across independent networks does not yet work reliably and is in progress. This
is stated plainly rather than hidden. See [Architecture](ARCHITECTURE.md).

## Start here

Read [Privacy](PRIVACY.md) for why the prompt stays local and what that does and
does not protect. Read [Architecture](ARCHITECTURE.md) for how the system fits
together. Read [CLI](CLI.md) to join a mesh, serve a slice, or run a chat. Read
[Benchmarks](BENCHMARKS.md) for what has actually been measured.

## Principle

No marketing, no lies. Performance numbers are real, limitations are documented,
and work in progress is labeled as such.