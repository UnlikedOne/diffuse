# NAT relay

Most machines on the internet cannot be dialed. A laptop on a home network, a
workstation in an office, a VM behind a corporate firewall: all of them can open
outbound connections, none of them can accept inbound ones. If Diffuse only
worked for machines with public addresses, it would be a cloud product with extra
steps.

This page describes how a node behind NAT serves a slice anyway.

Status: implemented and working across independent networks. A node behind a home
router in one country has served a slice to a client on a different network in
another, with a sentinel acting only as a rendezvous point.

## The problem

A node that holds layers 8 to 16 has to receive an encrypted hidden state from
whoever holds layers 0 to 8, run its slice, and send the result onward. That
requires being reachable. Behind NAT, the node has a private address such as
192.168.1.42 and the router presents a single public address shared with every
other device on the network. Nothing outside can open a connection to it.

The classic answers are UPnP port mapping, which is unreliable and often
disabled, and hole punching, which fails against symmetric NAT. Diffuse uses
neither. It uses connection reversal through a sentinel, which works everywhere
outbound connections work, at the cost of an extra hop.

## Detecting the situation

A node does not assume anything about its own reachability. It asks.

On startup, after binding its compute port, the node calls `CheckReachability` on
a sentinel, passing the compute endpoint it believes it has. The sentinel reads
the source IP of that call, which is the node's real public address as seen from
outside, and attempts a TCP connection back to that address on the given port.

The sentinel returns two things: whether the connection succeeded, and the
observed address. Both matter.

If the probe succeeds, the node is directly reachable and announces itself
normally. If it fails, the node knows it is behind NAT and switches to relayed
mode. Either way it now knows its public address, which is the fix for a subtle
earlier bug: a node that binds `0.0.0.0` used to announce `0.0.0.0` in gossip,
so every NAT'd node in the network collided under the same identifier and clients
derived a useless `0.0.0.0:10440` compute endpoint. Nodes now advertise the
address the sentinel actually saw.

If no external sentinel is available to probe against, the node assumes it is
reachable. That is the correct assumption for a public bootstrap node, which is
the only case where the situation arises.

## Registering with the relay

A node that discovers it is behind NAT opens a long lived bidirectional stream to
a sentinel: the `Attach` RPC.

The first message on that stream is a registration marker carrying the node's
public key. The sentinel records the mapping from node id to stream and keeps the
stream open. Because the node opened the connection outbound, the NAT allows
traffic back along it indefinitely.

From then on the sentinel can push work to a node it could never have dialed.

## Routing a request

When a client builds a route and finds that the replica it needs is marked as
unreachable, it does not try to connect. It sends a `RelayCompute` call to the
sentinel, containing the target node id and the encrypted compute request.

The sentinel looks up the node id in its registration table, wraps the request in
an envelope with a fresh request id, and pushes it down the attached stream. The
node receives it, runs its slice, and sends a reply back up the same stream
carrying the same request id. The sentinel matches the reply to the waiting
caller and returns it.

If the target is not attached, the sentinel returns `target not attached` rather
than hanging, and the client marks that replica dead and tries another.

## What the sentinel can and cannot see

The sentinel forwards ciphertext. Session keys are established between the client
and each serving node directly, using X25519, and the payload is encrypted with
ChaCha20-Poly1305. The sentinel holds no key that would let it decrypt anything
passing through it.

What it does see is metadata: which node is talking to which, how often, and how
large the payloads are. A sentinel operator can build a picture of who serves
what and when. This is real and worth stating plainly. It is the same trade
that any rendezvous server involves.

It also sees the node ids of everyone attached to it, which is public
information anyway, since node ids circulate in signed gossip.

## Leaving

When a node stops, its stream to the sentinel breaks. The sentinel notices the
broken stream, removes the registration, and evicts the peer from its registry
immediately rather than waiting for the gossip staleness timeout.

That eviction matters. Without it, there is a window of up to a minute where the
peer is gone but still advertised, and clients route to a node that no longer
exists, getting `target not attached` after already committing to a route. The
relay knows about the departure long before gossip would, so it tells the
registry.

## Cost

A relayed hop is two network legs instead of one: client to sentinel, sentinel to
node, and the same on the way back. On a pipeline with several relayed stages
this adds up.

The overhead has not been measured. Every benchmark so far routed directly to
nodes with public addresses, so the relay cost is currently unknown. Measuring it
is the next benchmark that matters, because in a real peer to peer network most
contributors are behind a router and the relayed path is the common case, not the
exception.

## Limits

The sentinel is a single point of failure for the nodes attached to it. If it
goes down, every relayed node it was serving becomes unreachable until it
reattaches elsewhere. Nothing currently reattaches automatically to a different
sentinel.

There is no admission control on the relay. A node can attach and consume
sentinel bandwidth without contributing anything. On a public network with
adversarial participants this would need addressing.

Relayed nodes are not currently sent `ClearSession` when a client finishes,
because the cleanup path only walks direct replicas. Their KV caches expire on a
ten minute timer instead.