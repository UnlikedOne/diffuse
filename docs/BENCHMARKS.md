# Benchmarks

This page reports what has actually been measured, on what hardware, under what
conditions, and separately describes what is planned but not yet run. Numbers in
the measured sections are real. Where something was not measured, or was measured
badly, it says so.

Honesty about performance is a design goal of this project. A model split across
ordinary machines is not going to be fast, and pretending otherwise would be
dishonest. The point of Diffuse is privacy, not speed.

## Measured: Mistral 7B across three nodes

Run of 19 July 2026. The first end to end measurement of a real model on the
public network.

### Topology

Three Hetzner CPX42 instances in Helsinki, each 8 shared vCPU and 16 GB of RAM,
no GPU. Every node was capped with `--overhead 0.6` so that no single machine
could hold the whole model, forcing a three stage pipeline.

| Node | Location | Slice | Layers |
|------|----------|-------|--------|
| 65.108.214.7 | Helsinki | 0:14 | 14 |
| 95.217.133.207 | Helsinki | 14:28 | 14 |
| 157.180.24.80 | Helsinki | 28:32 | 4 |

Model: `mistralai/Mistral-7B-Instruct-v0.3`, 32 layers, loaded in bfloat16.
Entry point: the public sentinel at 204.168.151.107. The client ran on a home
connection behind NAT in Germany, so every token crossed the public internet
twice, once outbound to the first stage and once inbound from the last.

All three replicas were routed directly. No relay was involved in this run.

### Single stream latency

Five consecutive runs, same prompt, 80 tokens each, greedy decoding.

| Run | Total | Prefill | Per token | Compute | Network and crypto |
|-----|-------|---------|-----------|---------|--------------------|
| 1 | 68643 ms | 2451 ms | 837 ms | 433 ms | 402 ms |
| 2 | 66894 ms | 1306 ms | 830 ms | 437 ms | 391 ms |
| 3 | 69835 ms | 1435 ms | 865 ms | 449 ms | 415 ms |
| 4 | 69523 ms | 1657 ms | 858 ms | 436 ms | 420 ms |
| 5 | 66359 ms | 1304 ms | 823 ms | 441 ms | 379 ms |

Mean per token: 843 ms. Standard deviation: about 16 ms, under two percent. The
pipeline is stable and the measurement is reproducible.

Roughly 52 percent of each token is compute and 48 percent is network and
cryptography. That split is the useful number: on this hardware, moving the
hidden state costs almost as much as computing it.

### Prefill

Prefill ranged from 1304 ms to 2451 ms and did not track prompt length. A four
token prompt and a nine token prompt produced the same figure within noise,
which suggests prefill here is dominated by a fixed cost, most likely session
setup and the first pass through the pipeline, rather than by the prompt itself.

### Concurrency

Two simultaneous requests were measured earlier in the same session, each
returning 766 ms per token against 843 ms for a single request. Aggregate
throughput rose from about 1.19 to 2.61 tokens per second.

This is the expected behaviour of a pipeline: with more than one request in
flight, stages that would otherwise idle while waiting for the previous stage are
given work. It supports the view that CPU serving suits batch and asynchronous
workloads better than interactive chat.

The measurement should be treated as indicative rather than solid. An attempt to
extend it to four concurrent requests failed: the client spawns a local Python
worker on a fixed port, so several clients on one machine collide. Proper
concurrency numbers need either dynamic worker ports or clients on separate
machines. That is a limitation of the client, not of the network.

### What this run does not establish

**Output quality was poor during this run.** Generation degenerated into
repetition on longer outputs. The cause has since been found and fixed: each
stage derived its sequence position from `DynamicCache.get_seq_length()`, which
reads layer index 0. Layers keep their global index when a model is sliced, so a
stage holding layers 16 to 31 wrote its keys and values at indices 16 to 31 and
left index 0 empty forever. That stage therefore restarted its rotary position
encoding from zero on every generated token: the attention saw the full history,
but each new token was positioned as if it were the first. Sequence position is
now tracked per session in the runner rather than read from the cache, and the
pipeline reproduces a single machine reference token for token. The latency
figures below were measured before that fix and remain valid, since the bug
affected correctness rather than speed.

**Partial shard download did not take effect.** The mechanism exists and works on
slices that begin at layer 0, but on middle slices the guard that checks for
unmaterialised parameters fires on the embeddings, the final norm and the output
head, which a middle stage legitimately does not need. The loader then falls back
to a full download. Every node in this run therefore fetched the entire 14.5 GB
model rather than only its own shards. This does not affect the latency numbers,
but the disk saving remains unproven.

**No relay path was exercised.** All three nodes had public addresses. Relayed
serving through a sentinel adds two extra network legs per token and was not
measured here.

### Reproducing this

```bash
diffuse host --model mistralai/Mistral-7B-Instruct-v0.3 \
  --spawn-worker \
  --public-addr YOUR_IP:9440 \
  --bootstrap http://204.168.151.107:9440 \
  --overhead 0.6
```

Run that on three machines, waiting for each to report a slice before starting
the next so that the assignment fills the gaps in order. Then query from a fourth
machine:

```bash
RUST_LOG=info diffuse query \
  --prompt "Explain black holes in two sentences." \
  --model mistralai/Mistral-7B-Instruct-v0.3
```

The `generation:` line in the log carries the latency decomposition. Total cost
of the run described above was under two euros of Hetzner time. Delete the
instances afterward, because Hetzner bills them until they are destroyed, not
until they are powered off.

## Measured: persistent connections

An earlier figure, concerning the transport rather than a full model.

Early builds opened a fresh connection between nodes for every token. This was
the largest single source of latency. Switching to persistent connections reused
across tokens reduced per token latency from 498 ms to 335 ms, about 33 percent.

| Configuration | Latency per token |
|---------------|-------------------|
| Fresh connection per token | 498 ms |
| Persistent connection | 335 ms |

This was measured on the development setup at the time, not on a controlled multi
node cluster, so treat it as a transport improvement figure rather than a
throughput benchmark for any particular model. The instrumentation that produced
it is the same one that decomposes each token into compute and network time, and
it is available live in the chat.

## Planned

Three measurements are missing, in rough order of importance.

**A reference run on one machine.** The same model, same prompt, same greedy
decoding, generated by transformers on a single machine with no Diffuse in the
path. If the output repeats there too, the repetition seen above is a property of
greedy decoding and the distributed computation is sound. If it does not, the
pipeline is degrading the computation and that has to be found before any further
performance work means anything.

**The relay path.** Every measurement so far routed directly to nodes with public
addresses. A contributor behind NAT serves through a sentinel relay, which adds
two network legs per token. That cost is currently unknown, and it is the case
that matters most for a peer to peer network where most machines sit behind a
router.

**A heterogeneous network.** Compute currently sits at about 440 ms per token on
shared vCPU. A consumer GPU would bring that closer to tens of milliseconds,
at which point the network becomes the dominant cost and the shape of the whole
system changes. Measuring a mixed network, some CPU nodes and some GPU nodes,
would show how the orchestrator handles stages of very different speeds.

An earlier plan called for a LAN benchmark inside a single datacenter, compared
against dispersed nodes, to separate network cost from compute cost. That plan is
deferred: a controlled machine park is the enterprise deployment case, and the
enterprise product does not exist yet. The public network is what Diffuse is
today, so that is what gets measured.