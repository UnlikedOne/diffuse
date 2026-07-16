# Architecture

Diffuse runs a large language model across several machines that do not trust
each other, without any single machine ever seeing your prompt in the clear.

This document describes how the system is put together. It reflects the current
state of the code, including the parts that do not yet work reliably. Where
something is incomplete, it says so.

## The core idea

A transformer model is a stack of layers. A forward pass runs the hidden state
through layer 0, then layer 1, and so on until the final layer produces the next
token.

Diffuse splits that stack vertically. Each node holds a contiguous slice of
layers. A request travels through the nodes in order, like a pipeline. The first
slice always stays on your own machine.

```
your machine        node A            node B         your machine
[ layers 0..k ] -> [ layers k..m ] -> [ layers m..n ] -> [ decode ]
     prompt            blind             blind            your token
   stays here                                            comes back
```

Because your machine runs the first layers, the raw token embeddings of your
prompt never leave it. The other nodes only ever receive a hidden state, which
is a tensor of activations from somewhere in the middle of the network. This is
the property the whole project is built around. It is discussed in detail in
[Privacy](PRIVACY.md).

## Nodes and slices

A node is one process that owns one slice of layers. A slice is defined by a
start layer and an end layer. When a node starts, it loads only the weights for
its slice, so a node holding ten layers of a large model needs far less memory
than a machine trying to hold the whole thing.

Slices are assigned so that the union of all live slices covers the full model.
If a slice has no live node, the model cannot serve requests, and the client
reports that a replica is dead.

## Daemon and worker

Each node is two processes.

The daemon is written in Rust and uses tonic for gRPC. It handles discovery,
transport, encryption, relay, and the pipeline routing. It is the part that
talks to the network.

The worker is written in Python and runs the actual PyTorch forward pass for the
slice. It listens on port 50051, bound to localhost only, and is never exposed
to the network. The daemon is the only thing that talks to it.

The split exists because the numerical work belongs in PyTorch while the
networking, cryptography, and coordination belong in Rust. Keeping the worker
local-only means the machine learning runtime never has to be network hardened.

## Discovery

Nodes find each other through a gossip protocol. Each node periodically
announces what slice it serves and how to reach it. Announcements are signed
with Ed25519, so a node can verify who produced a given piece of gossip and
reject anything unsigned or tampered with.

Gossip runs on port 9440. Compute runs on port 10440, which is always the gossip
port plus 1000.

## Sentinels

Some nodes have a public IP and act as entry points for the mesh. These are
called sentinels. A new node contacts a sentinel to join the gossip network and
to test whether it is reachable from the outside.

The live public sentinel is a Hetzner host at 204.168.151.107, running as the
systemd service diffuse.

## Encryption

Traffic between nodes is encrypted end to end. Session keys are established with
X25519 and traffic is encrypted with ChaCha20-Poly1305. A node in the middle of
the pipeline decrypts only the hidden state addressed to it, runs its slice, and
re-encrypts the result for the next hop.

This is a hand rolled protocol. It is honest to say that a hand rolled protocol
carries more risk than a reviewed standard transport, and the Enterprise track
is expected to move to mTLS for exactly this reason. For the public peer to peer
network the custom scheme is what ships today.

## Request path

A single request goes through these stages.

The client runs the local layers on the prompt and produces a hidden state. It
picks a replica, meaning one live node per remaining slice, and opens a
persistent connection to the first remote node. The hidden state is encrypted
and sent. Each node decrypts, runs its slice, encrypts, and forwards to the next
node. The final node returns the last hidden state to the client. The client
runs the final layers, samples a token, and streams it to the user. The loop
then repeats for the next token, reusing the same connections.

Persistent connections matter because opening a fresh connection per token was
the single largest source of latency. Reusing them brought per token latency
from 498 ms to 335 ms. See [Benchmarks](BENCHMARKS.md).

## Routing and NAT

Reaching a node with a public IP is straightforward. Reaching a node behind NAT
is not, because it has no address the client can dial directly.

Diffuse has a relay path for this. A NATed node registers with a sentinel over a
bidirectional stream. When a client needs to reach that node, it sends the
request to the sentinel, which forwards it down the registered stream and passes
the response back. The client routes such nodes through a Relayed replica type
rather than dialing them directly.

This path works in a single machine loopback test. It does not yet work
reliably between independent networks. When a node on one network hosts a model
and a client on another network tries to reach it, the client currently fails
with errors such as all replicas dead or a failure to pre-connect to
0.0.0.0:10440. The likely causes are a node advertising 0.0.0.0 as its endpoint,
a NATed node not registering correctly with the relay, or a node_id mismatch
between the gossip table and the relay table.

Until this is fixed, serving a model from behind NAT across networks should be
considered non functional. Nodes with public IPs, and nodes on the same network
as the client, work today. The full relay design is in
[nat-relay-design.md](nat-relay-design.md).

## Known limitations

The relay does not work reliably across independent networks, as described
above. The custom transport encryption is not a reviewed standard. Small models
run greedy decoding with no sampling. Interrupting generation with Ctrl+C
currently tears down the whole chat session and can leave an orphaned Python
worker. These are tracked and none of them are hidden.