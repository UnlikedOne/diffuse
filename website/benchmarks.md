# Benchmarks

This page reports what has actually been measured, on what hardware, and under
what conditions. Numbers in the measured sections are real. Where something was
not measured, it says so.

Honesty about performance is a design goal. A model split across ordinary
machines is not going to be fast, and pretending otherwise would be dishonest.
The point of Diffuse is privacy, not speed.

## Mistral 7B across three nodes

Run of 19 July 2026, the first end-to-end measurement of a real model on the
public network. Three Hetzner CPX42 instances in Helsinki, each 8 shared vCPU and
16 GB of RAM, no GPU, capped so no single machine could hold the whole model.

| Node | Location | Slice | Layers |
|------|----------|-------|--------|
| 65.108.214.7 | Helsinki | 0:14 | 14 |
| 95.217.133.207 | Helsinki | 14:28 | 14 |
| 157.180.24.80 | Helsinki | 28:32 | 4 |

Model `mistralai/Mistral-7B-Instruct-v0.3`, 32 layers, bfloat16. The client ran
behind NAT in Germany, so every token crossed the public internet twice.

### Single-stream latency

Five consecutive runs, same prompt, 80 tokens each, greedy decoding.

<DxBars
  title="Per-token latency across 5 runs"
  unit=" ms"
  :max="1000"
  :items="[
    { label: 'Run 1', value: 837 },
    { label: 'Run 2', value: 830 },
    { label: 'Run 3', value: 865 },
    { label: 'Run 4', value: 858 },
    { label: 'Run 5', value: 823 }
  ]"
  caption="Mean 843 ms per token, standard deviation about 16 ms, under two percent. The pipeline is stable and reproducible."
/>

### Where each token goes

<DxStack
  title="Per-token cost: compute versus network and cryptography"
  unit=" ms"
  :segments="[
    { label: 'Compute', value: 439, tone: 'a' },
    { label: 'Network and crypto', value: 404, tone: 'b' }
  ]"
  caption="Roughly 52 percent compute, 48 percent network and cryptography. On this hardware, moving the hidden state costs almost as much as computing it."
/>

### Concurrency

Two simultaneous requests were measured in the same session. A pipeline fills its
idle stages when more than one request is in flight, so aggregate throughput more
than doubled.

<DxBars
  title="Aggregate throughput"
  unit=" tok/s"
  :max="3"
  :items="[
    { label: '1 request', value: 1.19 },
    { label: '2 requests', value: 2.61, tone: 'b' }
  ]"
/>

<DxBars
  title="Per-token latency under load"
  unit=" ms"
  :max="1000"
  :items="[
    { label: '1 request', value: 843 },
    { label: '2 requests', value: 766, tone: 'b' }
  ]"
  caption="Treat as indicative. Extending to four concurrent clients failed because a client spawns a worker on a fixed port, so several clients on one machine collide. That is a client limitation, not a network one."
/>

## Persistent connections

An earlier transport measurement. Early builds opened a fresh connection between
nodes for every token, which was the single largest source of latency. Reusing a
persistent connection across tokens cut per-token latency by about a third.

<DxBars
  title="Per-token latency by connection strategy"
  unit=" ms"
  :max="560"
  :items="[
    { label: 'Fresh / token', value: 498 },
    { label: 'Persistent', value: 335, tone: 'b' }
  ]"
  caption="About 33 percent faster. Measured on the development setup at the time, so read it as a transport improvement rather than a model throughput benchmark."
/>

## Reproducing this

```bash
diffuse host --model mistralai/Mistral-7B-Instruct-v0.3 \
  --public-addr YOUR_IP:9440 \
  --bootstrap http://204.168.151.107:9440 \
  --overhead 0.6
```

Run that on three machines, waiting for each to report a slice before starting
the next, then query from a fourth:

```bash
RUST_LOG=info diffuse query \
  --prompt "Explain black holes in two sentences." \
  --model mistralai/Mistral-7B-Instruct-v0.3
```

The `generation:` line in the log carries the latency decomposition.

## Planned

- **A single-machine reference run** to confirm the distributed pipeline matches
  a non-distributed baseline token for token.
- **The relay path**, which adds two network legs per token and is the case that
  matters most for a peer-to-peer network where most machines sit behind a router.
- **A heterogeneous network** mixing CPU and GPU nodes, where compute drops to
  tens of milliseconds and the network becomes the dominant cost.
