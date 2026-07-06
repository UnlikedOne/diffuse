# Security Policy

Diffuse is privacy and security software. Vulnerabilities are taken seriously.

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Report privately through GitHub's private vulnerability reporting
(the "Report a vulnerability" button under the Security tab), or by opening a
minimal issue asking for a secure contact channel without disclosing details.

Please include:

- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- affected component (daemon, worker, transport, gossip, etc.),
- any suggested remediation.

## Scope

Security-relevant areas include, but are not limited to:

- the encrypted inter-node transport (X25519 / ChaCha20-Poly1305),
- gossip authentication and peer-record signing,
- the client-side layer boundary (what leaves the client machine),
- session key handling and cache lifetime,
- the worker RPC surface.

## What to expect

This is early research software maintained by volunteers. We aim to acknowledge
reports promptly and to be transparent about fixes. We will credit reporters who
wish to be named, once a fix is available.

## Out of scope

- Attacks requiring a malicious node observing activations it is asked to
  compute: this is a documented limitation, not a vulnerability. See
  [`THREAT_MODEL.md`](THREAT_MODEL.md).
- Traffic metadata and IP exposure: documented limitations, not vulnerabilities.
