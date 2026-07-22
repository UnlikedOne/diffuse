# NAT relay

Most home machines cannot accept inbound connections. They sit behind a router
doing NAT, or behind CGNAT with no public address at all. Without help, such a
machine could only consume the network, never contribute to it. The relay fixes
that.

## Reachability check

On startup a node asks a sentinel to connect back to it. If the connection
succeeds, the node is **directly reachable** and serves compute on its own port.
If it fails, the node is behind NAT and switches to relayed mode.

## Serving through a relay

A relayed node opens a persistent connection to a sentinel and registers itself.
When a client wants to use that node, it sends the compute request to the
sentinel, which forwards it down the existing connection to the node, and passes
the reply back.

Crucially, the relay carries **ciphertext only**. The end-to-end encryption is
between the client and the serving node, keyed to the serving node's identity.
The sentinel relays sealed bytes and sees flow metadata (that a request passed,
its rough size, timing) but never the activations themselves.

## What this costs

- **A dependency on the sentinel.** A relayed node is reachable only while its
  sentinel is up. If the sentinel goes down, the node does not currently reattach
  elsewhere automatically.
- **Sentinel bandwidth.** Relayed traffic flows through the sentinel, so a busy
  relay consumes its bandwidth. There is no admission control on this today.

These are recorded honestly in the [limitations](/limitations).

## Why it matters

Without the relay, contribution would be limited to machines with public
addresses, which is a small and unrepresentative slice of the world. The relay is
what lets an ordinary home computer give layers back to the network rather than
only taking from it.

## Next

- [Trust and encryption](/concepts/trust)
- [Privacy and threat model](/privacy)
