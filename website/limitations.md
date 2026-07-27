# Limitations

Every item here is a real defect or a real gap, written down rather than left for
whoever tries the project next to discover. Diffuse is a prototype and says so.

## Affects usability

**One node per machine.** Ports 9440, 10440, 50051, and 50099 are fixed. A second
daemon or client on the same machine fails to bind, so a single machine cannot
run two concurrent queries.

**The entry node can read your prompt on the default path.** A thin client that
holds no weights sends its token ids to the node running the first slice, which
can decode them back to your prompt. Traffic is encrypted in transit, but the
entry node itself sees the prompt until the first layers run locally. A private
mode that keeps them local is in progress. See the [threat model](/privacy).

**Local model paths do not work.** `--model` is passed to the Hugging Face API,
which rejects a filesystem path. Serving weights not on the Hub is not possible
yet, and the model id also doubles as the network identifier, so a local path
means nothing to another peer.

**No download progress.** Worker output is suppressed, so a node fetching a large
model shows little for several minutes.

## Affects operations

**Some integration tests do not compile.** After several rounds of API changes,
a subset of the files under `crates/diffuse-daemon/tests/` no longer builds. The
capacity, registry, placement, and downgrade suites were revived and pass; the
routing, gossip, and worker-facing suites still need updating. There is also no CI
job that compiles or runs the tests on a pull request.

**Relayed peers are not sent ClearSession.** Session cleanup walks direct replicas
only, so a relayed node keeps its KV cache until the expiry timer removes it.

**The sentinel is a single point of failure for relayed nodes.** A node attached
to a sentinel that goes down becomes unreachable and does not reattach elsewhere
automatically. There is also no admission control on the relay.

**The OpenAI API has no authentication.** `diffuse serve` binds to `127.0.0.1` for
that reason and warns on any other address. Do not expose it on an untrusted
network without an authenticating proxy in front.

## Measured but unexplained

**Network cost is high even in one datacenter.** In a three-node run inside a
single facility, network and cryptography accounted for a large share of per-token
latency despite sub-millisecond physical latency. The overhead is serialization,
encryption, and per-hop cost rather than transit, and has not been profiled
further.

**Prefill does not scale with prompt length.** Short and slightly longer prompts
produce roughly the same prefill time. A fixed cost dominates, and which fixed
cost has not been established.

## What only works on some models

Diffuse serves any model whose computation is an ordered stack of layers that
passes one tensor between them. That covers ordinary chat models, models that
read images, audio or video, and encoder-decoders such as Whisper and MusicGen.

It does not cover U-Net diffusion, recurrent stacks such as Mamba, or anything
requiring `trust_remote_code`. Mixture-of-experts remains unproven: no such
model has been run end to end here. Image generation is unverified for the same
reason. See [what Diffuse can run](/concepts/model-support) for the rule and
[choosing a model](/guides/choosing-a-model) for the practical advice.
