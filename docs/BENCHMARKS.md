# Benchmarks

This page reports what has actually been measured, and separately describes the
benchmarks that are planned but not yet run. Numbers in the measured section are
real. The planned section contains no results, only the method that will be used
to produce them.

Honesty about performance is a design goal of this project. A distributed model
that runs on ordinary hardware is not going to be fast, and pretending otherwise
would be dishonest. The point of Diffuse is privacy, not speed.

## Measured: persistent connections

The one number measured so far concerns the transport, not a full model sweep.

Early builds opened a fresh connection between nodes for every token. This was
the largest single source of latency. Switching to persistent connections that
are reused across tokens reduced per token latency from 498 ms to 335 ms, a
reduction of about 33 percent.

| Configuration            | Latency per token |
|--------------------------|-------------------|
| Fresh connection / token | 498 ms            |
| Persistent connection    | 335 ms            |

This was measured on the development setup at the time, not on a controlled
multi node cluster, so treat it as a transport improvement figure rather than a
throughput benchmark for any particular model. The instrumentation that produced
it also decomposes each token into compute time and network time, and is
available live in the chat.

## Planned: 7B on CPU

A rigorous CPU benchmark is planned but has not been run. The plan is documented
here so the method is public before any results exist.

Model. Qwen2.5-7B-Instruct, a 7B parameter model, chosen as a realistic lower
bound for something people would actually want to run.

Hardware. Four CPU nodes on Hetzner CPX41 instances, each with 16 GB of RAM.

Two scenarios, run separately to isolate network cost from compute cost. The
first is dispersed across the internet, with nodes in different locations. The
second is a LAN within a single Hetzner datacenter. Comparing the two shows how
much of the latency is the network and how much is raw compute.

Metrics. Single stream latency split into compute and network. Aggregate
throughput under several concurrent requests. Prefill time for the prompt.

Expected outcome, stated honestly in advance. A 7B model on CPU will be slow, on
the order of seconds per token. That is physics, not a bug. The useful conclusion
is likely to be that CPU serving suits batch and asynchronous workloads rather
than interactive chat. The benchmark exists to quantify that, not to hide it.

Two practical notes for whoever runs this. First, check the worker dtype in
worker/diffuse_worker/slicing.py, because a 7B model in float32 needs about
28 GB, which will not fit the plan, so it must run in a smaller dtype. Second,
delete the Hetzner instances afterward, because they are billed until they are
destroyed.

## What is not benchmarked yet

There is no end to end latency figure for any full model, no throughput number
under load, and no prefill measurement. Those come from the planned run above.
Until then, the only honest performance claim is the persistent connection
improvement described at the top of this page.