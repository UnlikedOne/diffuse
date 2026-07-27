# Use cases

Where Diffuse fits, and where it does not. Being honest about both is the point.

## Where it fits

### Privacy-sensitive drafting and reasoning

You want to think out loud with a model without handing your text to a company
that can log it. Diffuse keeps your words off any central server and encrypts
every hop. For the strongest guarantee, keep the first layers local (in progress).

### Running a model your machine cannot hold alone

A 32B model does not fit on a laptop. With a handful of contributing machines,
each holding a slice, the network serves it together. You get access to a model
that no single one of your devices could run.

### Community and lab networks

A research group, a hackerspace, or a company team can run a **private network**
of their own machines with their own sentinel, sharing capacity internally and
never touching the public network. See [self-hosting](/self-hosting).

### Contributing spare compute

You have a machine that sits idle. Run `diffuse host` and it contributes a slice
to the network, helping serve models for everyone, including from behind a home
router via the relay.

### A drop-in OpenAI endpoint for local tools

You already use LibreChat, Open WebUI, or Continue. Point them at
`diffuse serve` and they work against the network with no code changes.

## Where it does not fit

### Low-latency interactive products

Generation over a distributed CPU pipeline is measured in seconds per response,
not milliseconds. Diffuse is a poor fit for anything that needs snappy, real-time
replies at scale.

### High-throughput production serving

Concurrent requests on a node are grouped into one pass, which roughly doubles
aggregate throughput, but there is no autoscaling and no SLA. This is a
prototype for privacy and decentralization, not a production inference
platform.

### Anonymity

Diffuse hides what you say, not that you are talking. If you need to hide your
identity or the fact of use, Diffuse alone does not provide that.

### Anything requiring unsupported model types

U-Net diffusion, recurrent stacks such as Mamba, and models shipping their own
Python do not work, the last by decision rather than by limitation.
Mixture-of-experts is unproven. See
[what Diffuse can run](/concepts/model-support).

## A realistic mental model

Think of Diffuse as a **volunteer, encrypted, peer-to-peer** way to run open text
models, trading speed for privacy and decentralization. It is closer in spirit to
Tor or BitTorrent than to a hosted API: slower, community-run, and built so that
no single party sees everything.
