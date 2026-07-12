# NAT Relay Design

Status: design approved, implementation pending.

This document specifies how nodes behind NAT can serve compute in Diffuse, using
a sentinel as an encrypted relay. It is the design agreed before implementation;
follow it step by step when building the feature.

## Problem

A node behind NAT can open outbound connections but cannot accept inbound ones.
Today the client (via the orchestrator) dials each pipeline node directly with
`connect_compute(endpoint)`. When the node is behind NAT this fails with
"connection refused". Such a node can consume inference but cannot serve it.

Brick 1 (already implemented) lets a node learn whether it is reachable: at
startup it asks a sentinel to dial it back on its compute port, and it records a
`reachable` flag that travels in the gossip records.

This design covers the next step: making unreachable nodes serve compute through
a relay, without ever exposing plaintext to the relay.

## Principle

An unreachable node opens a permanent outbound stream to a sentinel. The sentinel
pushes incoming compute requests down that stream and forwards the replies back.
The compute payload stays end to end encrypted between the client and the serving
node, so the sentinel only ever sees ciphertext plus flow metadata.

## Actors

- C: the client that initiates a query and orchestrates the pipeline.
- S: a sentinel with a public address, acting as relay.
- N: a node behind NAT that serves a slice.

## Flow

### Registration (at N startup, when N is unreachable)

1. N detects it is unreachable (Brick 1).
2. N opens a permanent bidirectional stream to S via `Relay.Connect`.
3. The first message N sends identifies it by `node_id`.
4. S stores `node_id -> sender` in a registration table, where `sender` is the
   sending side of the channel that pushes work toward N.
5. The stream stays open for N's lifetime, with keepalives. If it drops, N
   reconnects.

### Request (C wants to compute on N)

1. C builds its route and sees N is `reachable: false`.
2. Instead of `connect_compute(N)`, C calls `Relay.RelayCompute` on S, passing
   N's `node_id` and its usual encrypted `ComputeRequest`.
3. S looks up N's channel, generates a unique `request_id`, and pushes
   `{request_id, ComputeRequest}` into the stream toward N.
4. S registers a pending entry `request_id -> oneshot sender` and waits on it
   with a timeout.

### Compute and return

1. N receives `{request_id, ComputeRequest}` on its stream.
2. N decrypts (it holds the key, it is the E2E recipient), runs its worker,
   re-encrypts the result.
3. N sends `{request_id, ComputeResponse}` back on the same stream.
4. S finds the pending entry via `request_id` and delivers the `ComputeResponse`
   into the oneshot channel.
5. The waiting `RelayCompute` call wakes up and returns the response to C.

C now has a response computed by N, without the sentinel ever seeing plaintext
and without N ever needing to be reachable.

## Protocol (proto additions)

```proto
message RelayRegister {
  bytes node_id = 1;
}

message RelayEnvelope {
  string request_id = 1;
  ComputeRequest request = 2;
}

message RelayReply {
  string request_id = 1;
  ComputeResponse response = 2;
}

message RelayComputeRequest {
  bytes target_node_id = 1;
  ComputeRequest request = 2;
}

service Relay {
  // N opens this permanent stream: it sends a first RelayReply carrying its
  // registration, then receives RelayEnvelope items and answers with RelayReply.
  rpc Connect(stream RelayReply) returns (stream RelayEnvelope);
  // C calls this to reach a node behind NAT.
  rpc RelayCompute(RelayComputeRequest) returns (ComputeResponse);
}
```

Note on `Connect`: the registration is carried as the first `RelayReply` on the
outbound stream (with an empty `request_id` reserved for registration, and the
`node_id` conveyed there), or via a dedicated first-message convention. Decide
the exact encoding at implementation time; the simplest is a reserved
`request_id = "register"` whose payload carries the node id.

## Broker state (sentinel side)

Two shared structures behind async locks:

- `registrations: Map<node_id, mpsc::Sender<RelayEnvelope>>`
  Push work toward each connected unreachable node.
- `pending: Map<request_id, oneshot::Sender<ComputeResponse>>`
  Wake the waiting `RelayCompute` call when its reply returns.

On `RelayCompute`:
1. Generate `request_id`.
2. Create a oneshot channel, insert its sender into `pending`.
3. Look up the target in `registrations`; if absent, fail fast (client marks the
   replica dead and fails over, consistent with existing behavior).
4. Push the envelope toward N.
5. Await the oneshot with a timeout. On timeout, remove the pending entry and
   return an error.

On a `RelayReply` arriving from N:
1. Look up `pending` by `request_id`.
2. Send the response into the oneshot; remove the entry.
3. If no pending entry exists (late reply after timeout), drop it.

On N disconnect:
1. Remove N from `registrations`.
2. Any pending requests targeting N time out and fail over.

## Client routing change

In `build_from_registry`, when a peer is `reachable: false`:
- Do not `connect_compute` directly.
- Mark the replica as relayed, remembering which sentinel relays it and the
  target `node_id`.
- In the failover path, relayed replicas call `RelayCompute` on the sentinel
  instead of `RunSlice` directly.

A relayed replica should be flagged so the latency instrumentation can attribute
the extra hop, and so the UI can be honest that this path is slower.

## Design decisions (agreed)

1. Latency: each token to a relayed node travels C -> S -> N -> S -> C, i.e. two
   extra network legs per token. This is accepted and must be measured. Relayed
   replicas are flagged so instrumentation and UI can be honest about it.
2. Security invariant: encryption is between C and N, keyed from N's kx public
   key published in signed gossip. S cannot derive the key and cannot decrypt.
   The relay sees only metadata (who talks to whom, sizes, timing), never
   content. This must be stated plainly in the threat model.
3. Multiplexing: one sentinel may relay several nodes and serve several clients
   at once; `request_id` correlates. No per-sentinel node cap in v1 (note as
   future hardening).
4. Fallback: if the target is not registered (not yet connected or dropped),
   `RelayCompute` fails fast and the client fails over to another replica, reusing
   the existing failover logic.

## Threat model addition (to write)

Add to the limitations/threat model: when a serving node is behind NAT, its
compute traffic is relayed through a sentinel. The sentinel observes flow
metadata for that traffic (endpoints, timing, volume) but never plaintext, since
the payload is end to end encrypted between client and serving node. Users who
must hide this metadata should serve only from reachable nodes or place a
network anonymity layer beneath Diffuse.

## Implementation order (next session)

1. Proto: add the `Relay` service and messages; regenerate.
2. Sentinel broker: `registrations` and `pending` maps, the two RPCs, timeouts,
   disconnect cleanup.
3. Node side: when unreachable, open and maintain the `Connect` stream, handle
   envelopes, run the worker, send replies, reconnect on drop.
4. Client routing: relayed replica variant, `RelayCompute` in the failover path,
   relayed flag for instrumentation.
5. Test: unreachable node (e.g. behind home NAT or Cloud Shell) serving a slice
   through the Hetzner sentinel to a separate client.
6. Docs: threat model addition, README note on relayed serving and its latency
   cost.

## Explicitly out of scope for this feature

- UDP hole punching and direct NAT-to-NAT paths (future, likely via iroh/QUIC).
- UPnP/NAT-PMP port mapping (small future add).
- Topology-aware placement to minimize relay hops (premature until metrics show
  the need).
