# Slices and the pipeline

A transformer is a stack of near-identical layers. Diffuse splits that stack into
contiguous ranges called **slices**, and every node holds one slice.

## What a slice is

A slice is a half-open range of layer indices, written `start:end`. A 24-layer
model might be covered by:

```
node A   0:8
node B   8:16
node C   16:24
```

Together these three nodes serve the whole model. A request enters at layer 0,
flows through A, then B, then C, and the last slice produces the output.

The first slice also holds the embedding table, and the last slice holds the
output head. A node that takes the front of the model therefore carries more
weight than a node in the middle, which the capacity planner accounts for.

## How a node picks its slice

When you run `diffuse host`, the node:

1. **Profiles itself.** It measures available memory and computes how many layers
   of the requested model it can hold.
2. **Looks at coverage.** It reads the current network state from gossip and finds
   where the model is thin or missing.
3. **Takes the most useful slice.** It fills a gap first, then reinforces the
   weakest slice, then adds redundancy where coverage is already complete.

This means you do not choose a slice by hand. You bring a machine, and the
network assigns it where it helps most.

## Coverage and servability

The capacity analysis walks the model from layer 0 to its true total and checks
that every range is covered by at least one live node. The real layer count comes
from the worker and travels through gossip, so a model that is only partially held
is correctly reported as **incomplete**, not mistaken for complete.

A model is **servable** only when its coverage has no gaps. `diffuse models` and
the OpenAI `/v1/models` endpoint list only servable models. A request for an
incomplete model fails cleanly and names the missing layer ranges rather than
building a route that dead-ends.

## The KV cache

Attention needs the keys and values of every previous token. Recomputing them
each step would be wasteful, so each node keeps a **KV cache** per session. The
client sends a fresh session id with every request and clears it when generation
ends, so no cache state is ever shared between requests.

## Next

- [Gossip and discovery](/concepts/gossip)
- [Replication and healing](/concepts/replication)
