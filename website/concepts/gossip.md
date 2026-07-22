# Gossip and discovery

There is no central registry of who is on the network. Nodes learn about each
other by **gossip**: they periodically exchange their view of the world with a
few peers, and knowledge spreads.

## Bootstrapping

A new node needs one known address to start, called a **sentinel**. Diffuse ships
with built-in sentinels, and you can point at your own with `--bootstrap`. The
node contacts a sentinel, receives its list of known peers, and it is now part of
the mesh.

## What a peer record holds

Each node advertises a signed record of itself:

- node id (an Ed25519 public key)
- daemon and worker endpoints
- the model it serves and its slice range
- the model's total layer count
- a key-exchange public key
- whether it is directly reachable

The record is signed by the node's identity key, so a peer cannot forge another
node's advertisement. Records that fail signature verification are rejected.

## Keeping the view fresh

Nodes re-announce themselves periodically and exchange views with a small random
fanout of peers. A record carries a last-seen timestamp; a node that stops
announcing is **pruned** after it goes stale. This is what lets the network
notice a departure without any coordinator declaring it.

## Identity, not address

Peers are keyed by node id, not by network address. Two nodes behind the same NAT
announcing the same address stay distinct, and a node that changes address keeps
its identity. This avoids a class of bugs where nodes collide or overwrite each
other under a shared endpoint.

## Next

- [Replication and healing](/concepts/replication)
- [Trust and encryption](/concepts/trust)
