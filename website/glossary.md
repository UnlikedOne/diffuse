# Glossary

The vocabulary Diffuse uses, in one place.

### Slice

A contiguous range of a model's layers, written `start:end`, held by one node. The
unit Diffuse splits a model into.

### Stage

One step in the pipeline: the node (or nodes) that run a given slice. A request
passes through the stages in order.

### Node

A machine on the network. It may be a **host** (holding a slice and serving
compute), a **client** (running chat, query, or serve), or both.

### Client

A machine that uses the network. It tokenizes locally and drives generation but
holds no model weights by default.

### Host

A node running `diffuse host`. It holds a slice, announces it over gossip, and
serves compute to clients. Hosting is how the network exists.

### Sentinel

A well-known node used for bootstrap discovery and, when needed, as an encrypted
relay for peers behind NAT.

### Gossip

The protocol by which nodes exchange their view of the network with a few peers at
a time, so knowledge of who is online spreads without any central registry.

### Replica

An additional node holding the same slice as another. Replicas share load and keep
a model servable when one holder disappears.

### Servable

A model is servable when every layer range is covered by at least one live node,
so a request can flow end to end. `diffuse models` lists only servable models.

### Robust, fragile, incomplete

The three coverage states of a model. **Robust**: fully covered and above the
target replication. **Fragile**: fully covered but some slice has a single holder.
**Incomplete**: some layer range is held by nobody, so it is not servable.

### Relay

An encrypted forwarding path through a sentinel that lets a node behind NAT serve
compute. The relay carries ciphertext only.

### Activation, hidden state

The tensor of numbers passed between stages. It is the model's internal
representation of the input at a given depth, not the text itself.

### KV cache

The stored keys and values of previous tokens, kept per session on each node so
attention does not reprocess the whole prompt every step.

### Session

One generation, identified by a unique id. Its KV cache is cleared when generation
ends, so no state is shared between requests.

### Node id

A node's Ed25519 public key, which is its identity. Peers are keyed by node id, not
by network address.
