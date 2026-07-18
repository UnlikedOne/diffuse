# CLI reference

Every command Diffuse ships today, with the flags that exist and the behavior you
should expect. Anything not listed here does not exist yet.

## Install

One command, Linux x86_64:

```bash
curl -fsSL https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.sh | bash
```

This fetches the `diffuse` binary, sets up the Python worker in `~/.diffuse/worker`,
and puts everything in place. No account, no API key, no server of your own.

The script is [install.sh](https://github.com/UnlikedOne/diffuse/blob/main/install.sh)
if you would rather read it before piping it to a shell.

## How a node is put together

Every command starts a daemon. The daemon joins the gossip network, supervises a
local Python worker on port 50051 bound to localhost, and speaks to other nodes
over gRPC. You never start the worker yourself.

Two ports matter. Gossip runs on 9440. Compute runs on 10440, which is always the
gossip port plus 1000. If you run a node with a public IP, both need to be
reachable.

## diffuse chat

Interactive chat against a model served on the network.

```bash
diffuse chat
```

You pick a model from the ones currently live, and the client builds an encrypted
route to the nodes serving it. Your machine tokenizes the prompt and runs the
first layers locally, so the prompt itself is never transmitted. Tokens stream
back one at a time as they are generated.

To join through a specific sentinel rather than the built-in ones:

```bash
diffuse chat --bootstrap http://204.168.151.107:9440
```

## diffuse query

A single question, no interactive session:

```bash
diffuse query --prompt "Explain black holes in two sentences." --model <model>/<model>
```

With an explicit model and sentinel:

```bash
diffuse query \
  --prompt "Explain black holes in two sentences." \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --bootstrap http://204.168.151.107:9440
```

Useful for scripting, and for reading the latency instrumentation on a single
request without the chat UI in the way.

## diffuse models

List the models currently served by the network:

```bash
diffuse models
```

Shows each live model, the layer coverage, and how many replicas hold each slice.
A model is only usable when its slices are fully covered.

## diffuse host

Contribute your machine. The daemon measures how much of the model your hardware
can hold, takes the slice the network needs most, loads it, and starts serving.

```bash
diffuse host --model Qwen/Qwen2.5-0.5B-Instruct
```

Against a specific sentinel:

```bash
diffuse host --model Qwen/Qwen2.5-0.5B-Instruct --bootstrap http://204.168.151.107:9440
```

The model argument is a Hugging Face model id. The slice assignment is automatic:
you do not choose a layer range, the daemon picks one based on your available
memory and where the network is thin.

To keep contributing after you close the terminal:

```bash
nohup diffuse host --model Qwen/Qwen2.5-0.5B-Instruct > ~/.diffuse/host.log 2>&1 &
```

Follow it with `tail -f ~/.diffuse/host.log`, stop it with `pkill diffuse`. For a
node that survives a reboot, see `deploy/README.md` in the repository for the
systemd unit.

## Disk space

Models are downloaded to the Hugging Face cache, which grows quickly. A node
downloads the full model even when it only serves a slice of the layers, so a
7B model costs the full download regardless of the tranche assigned.

Check what is stored:

```bash
du -sh ~/.cache/huggingface/hub/models--* | sort -rh
```

Remove a model you no longer serve:

```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct
```

Failed downloads leave partial blobs behind. They occupy space and serve no
purpose:

```bash
find ~/.cache/huggingface/hub -name "*.incomplete" -delete
```

If a load fails with `No space left on device`, this cache is usually the
reason.

## What the encryption actually does

This is the part worth reading carefully, because it is the reason the project
exists.

Session keys are established with **X25519**, an elliptic-curve Diffie-Hellman
exchange. Each client generates an ephemeral keypair per session, so two requests
from the same person are not trivially linkable by a shared key. The public key of
every node is published in gossip and **signed with Ed25519**, so a client can
verify it is negotiating with the node it intends to, and forged or tampered peer
records are rejected.

Traffic is encrypted with **ChaCha20-Poly1305**, an authenticated cipher: a node
cannot silently alter a hidden state in flight without the tampering being
detected on decryption. There is no certificate authority anywhere in the design;
keys are bound to node identities directly.

The encryption is genuinely end to end between your machine and each serving node.
A node in the middle of the pipeline decrypts only the hidden state addressed to
it, runs its slice, and re-encrypts the result for the next hop. It never holds a
key that would let it read another hop's traffic.

Two honest caveats. This is a hand-rolled protocol built on sound primitives, and
a hand-rolled protocol has not had the review a standard transport has. And
encryption protects the wire, not the computation: the node running your slice
sees a plaintext tensor of activations in its own memory. What that does and does
not reveal is covered in [Privacy](PRIVACY.md).

## Reaching nodes behind NAT

A node with a public IP is dialed directly and works today. A node behind NAT has
no address a client can dial, so Diffuse routes it through a sentinel relay
instead.

That relay path works in a single-machine test but is **not yet reliable between
independent networks**. If a node on one network hosts a model and a client on
another network tries to reach it, the client currently fails with `all replicas
dead` or a failure to pre-connect to `0.0.0.0:10440`. Serving from behind NAT
across networks should be treated as non-functional until this is fixed.
Consuming from behind NAT works fine. The design and current status are in
[NAT relay design](nat-relay-design.md).

## Known rough edges

The startup banner still prints `version 0.1.0` on a v0.2.0 build. The version
string is hardcoded and does not yet track the release tag.

Pressing `Ctrl+C` during generation tears down the whole chat session instead of
interrupting only the current generation, and can leave an orphaned Python worker
process behind.

Very small models such as a 0.5B run greedy decoding with no sampling, so they can
loop on repetitive output. Sampling is planned. This is a property of the model
and the decoding strategy, not of the network carrying it.