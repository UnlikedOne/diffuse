# Trust and encryption

Diffuse has no certificate authority and no central directory of who is allowed
to participate. Trust is built from identity keys and end-to-end encryption
instead.

## Node identity

Every node has an **Ed25519 keypair**. The public key is the node id. It signs
its own gossip advertisement, so no peer can forge another node's record: change a
field and the signature no longer verifies, and the record is dropped.

Because identity is a key rather than an address, a node keeps its identity when
its network address changes, and two nodes cannot collide under a shared address.

## Sealing every hop

Compute between a client and a serving node is encrypted end to end:

- **X25519** key exchange establishes a shared secret. Each node advertises a
  key-exchange public key alongside its identity, so the client can seal a
  message that only the target node can open.
- **ChaCha20-Poly1305** encrypts and authenticates the activations in transit.

The keys are bound to node identities from gossip, which is why no certificate
authority is needed. There is nothing to issue, revoke, or trust externally.

## What a relay sees

When a node is reached through a [relay](/concepts/nat-relay), the encryption is
still between the client and the serving node. The relay forwards sealed bytes. It
learns flow metadata but never the plaintext activations.

## What a serving node sees

A node in the pipeline decrypts the activations addressed to it, runs its slice,
and re-encrypts the result for the next hop. Inside that node, in plaintext, the
data is a tensor of floating-point activations.

What that reveals about your prompt depends on where in the pipeline the node
sits and how many layers run before it. This is exactly the subject of the
[privacy and threat model](/privacy) page, which is honest about what is protected
and what is not.

## Next

- [Privacy and threat model](/privacy)
- [NAT relay](/concepts/nat-relay)
