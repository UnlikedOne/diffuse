# CLI

This page describes the diffuse command line interface. It covers joining a
mesh, serving a slice of a model, and running an interactive chat.

The examples assume the daemon binary is on your path. Build it from the
repository root with cargo if it is not.

## Overview

Diffuse is run as a daemon. The daemon joins the gossip network, optionally
serves a slice of layers, and exposes a local chat. A node needs to know at
least one sentinel to reach the rest of the mesh.

Ports used by a node are 9440 for gossip and 10440 for compute. The compute port
is always the gossip port plus 1000. Make sure both are reachable if you run a
node with a public IP.

## Global behavior

Every invocation starts the daemon, which supervises the local Python worker on
port 50051 bound to localhost. You do not start the worker yourself. The daemon
launches it when a slice needs to be served and talks to it over gRPC.

## Joining a mesh

To join the network through the live sentinel:

```
diffuse join --sentinel 204.168.151.107
```

This connects to the sentinel, exchanges signed gossip, and builds a local view
of which slices are live. Joining alone does not serve any layers. It is enough
to act as a client that runs prompts against models served by others.

## Serving a slice

To contribute a slice of layers to a model:

```
diffuse serve --model Qwen2.5-7B-Instruct --layers 8:16 --sentinel 204.168.151.107
```

The layers argument is a half open range, so 8:16 serves layers 8 through 15.
The daemon loads only those layers into the worker, announces the slice over
gossip, and starts accepting encrypted hidden states for it.

For the model to be usable end to end, every slice must be covered by at least
one live node. If a slice has no server, clients see a dead replica.

Serving from a machine with a public IP works directly. Serving from a machine
behind NAT relies on the relay, which is not yet reliable across independent
networks. See [Architecture](ARCHITECTURE.md) for the current status.

## Running a chat

To open an interactive chat against a model that is fully served on the mesh:

```
diffuse chat --model Qwen2.5-7B-Instruct --sentinel 204.168.151.107
```

The client runs the first layers of the model locally, so your prompt is never
sent in the clear. Tokens stream back one at a time as they are decoded.

The chat prints latency instrumentation that separates compute time from network
time, so you can see where time is going on your connection and hardware.

## Running your own sentinel

A sentinel is a node with a public IP that other nodes use as an entry point. To
run one, start the daemon on a public host and let others point their sentinel
flag at its address. On the live deployment this runs as a systemd service named
diffuse on the Hetzner host.

Make sure ports 9440 and 10440 are open to the internet on the sentinel host, or
nodes will not be able to join or compute through it.

## Known rough edges

The startup banner still prints version 0.1.0 even on a v0.2.0 build. The version
string is hardcoded and does not yet reflect the release tag.

Pressing Ctrl+C during generation currently tears down the whole chat rather
than interrupting only the current generation, and can leave the Python worker
process orphaned. Both are known and tracked.

Very small models such as a 0.5B model run greedy decoding with no sampling, so
they can loop on repetitive output. Sampling is planned.