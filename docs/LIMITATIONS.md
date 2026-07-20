# Known limitations

Everything on this page is a real defect or a real gap, written down rather than
discovered by whoever tries the project next. Items are grouped by how much they
affect someone actually using Diffuse.

## Affects usability

**One node per machine.** Ports 9440, 10440 and 50051 are fixed. A second daemon
on the same machine fails to bind. This also means a single client cannot issue
two concurrent queries, because each query spawns its own worker on port 50051,
and it is why the concurrency benchmark could not be extended beyond two
requests.

**Ctrl+C during chat tears down the session.** Interrupting generation ends the
whole chat rather than stopping the current response.

**No download progress.** Worker output goes to `/dev/null`, so a node fetching a
14 GB model shows nothing for several minutes. There is no way to distinguish a
slow download from a hung process without inspecting the Hugging Face cache
directory by hand.

**Version string is wrong.** The banner prints `version 0.1.0` regardless of the
actual release. It is a hardcoded string rather than `CARGO_PKG_VERSION`.

**Local model paths do not work.** `--model` is passed to the Hugging Face API,
which rejects a filesystem path. Serving weights that are not on the Hub is not
possible. Beyond the API question there is a design issue: the model id doubles
as the network identifier for which nodes are serving the same model, and a local
path means nothing to another peer. Doing this properly needs a canonical model
id separate from the weights source, plus a hash so peers can verify they are
serving identical weights.

## Affects operations

**Integration tests do not compile.** Seven files under
`crates/diffuse-daemon/tests/` fail to build after several rounds of changes:
`Peer` gained a `reachable` field, `Stage` and `Orchestrator` gained
instrumentation fields, `Replica` became an enum, `request_slice` now returns a
tuple, and `build_from_registry` takes a fifth argument. The unit tests for the
registry pass and cover the peer identity logic, but routing, repair and chaos
have no working coverage. This is worse than having no tests, because the files
give an impression of coverage that does not exist.

**Relayed peers are not sent ClearSession.** The session cleanup path walks
direct replicas only. A relayed node keeps its KV cache until the ten minute
expiry timer removes it.

**Sentinel is a single point of failure for relayed nodes.** A node attached to a
sentinel that goes down becomes unreachable and does not reattach elsewhere
automatically.

**No admission control on the relay.** Any node can attach and consume sentinel
bandwidth without contributing.

## Measured but unexplained

**Network cost is high for a single datacenter.** In the three node Helsinki run,
network and cryptography accounted for 48 percent of per token latency despite
all machines sitting in the same facility. Physical latency there is well under a
millisecond, so the ~400 ms is serialisation, encryption and per hop overhead
rather than transit. It has not been profiled further.

**Prefill does not scale with prompt length.** A four token prompt and a nine
token prompt both produce roughly 1400 ms of prefill. The fixed cost dominates,
but which fixed cost has not been established.

## Fixed, recorded for context

These were real and are no longer. They are listed because the failure modes are
instructive.

**Peers keyed by endpoint.** The peer registry used `daemon_endpoint` as its key.
Every node without `--public-addr` announced `http://0.0.0.0:9440`, so all NAT'd
nodes collided under one entry, overwrote each other, and clients derived an
unusable `0.0.0.0:10440` compute endpoint. Peers are now keyed by node id, which
is an Ed25519 public key and genuinely unique. Eight unit tests cover this.

**Nodes advertised their bind address.** Related to the above: a node binding
`0.0.0.0` announced `0.0.0.0`. Nodes now advertise the address a sentinel
observed them from.

**Orphaned Python workers.** Ctrl+C on the daemon left the worker running, which
kept the model in memory, which made the next start fail with `machine too small
to hold any slice`. The worker is now killed on exit via a guard whose Drop
implementation waits on the child process.

**Shared KV cache across queries.** Every query used the literal session id
`query-session`, so consecutive queries shared a cache on the serving nodes. The
second query would continue the first one's generation: asking about Michael
Jackson after asking for three colours returned "4. Green, 5. Yellow, 6. Purple".
Beyond correctness this was a privacy defect, since two different users would
have shared cache state. Sessions are now UUIDs, cleared explicitly when a query
ends, and expire after ten minutes otherwise.

**Sequence position was read from a cache indexed by global layer index.** Stages
other than the first never advanced their position counter, because
`DynamicCache.get_seq_length()` reads layer 0 and a sliced stage writes its keys
and values at the layer indices it inherited from the full model. A stage holding
layers 16 to 31 filled cache entries 16 to 31 and left entry 0 empty forever, so
its reported sequence length stayed at zero. Every generated token was therefore
positioned as if it were the first: attention saw the full history, but the
rotary encoding restarted from zero each step, and output degraded progressively
into repetition. This is what produced the repetitive text in the July benchmark,
which had been attributed to greedy decoding. Position is now tracked per session
in the runner rather than read from the cache. Parity against a single machine
reference is exact, token for token.

**Partial shard download fell back to a full download on most slices.** The
loader builds a skeleton on the meta device from the full config, replaces the
layer list with the slice it keeps, and downloads only the shards holding those
layers. A final guard then verified that every parameter in the model had been
materialised, which is the wrong contract: the skeleton still contains the
embeddings, the final norm and the output head, and a slice legitimately does not
need all of them. Any slice missing one of those three failed the guard and fell
back to fetching the whole model, so nodes downloaded 14.5 GB regardless of the
slice they served. Modules the slice does not use are now detached from the
skeleton after the tie repair and before the guard, which makes "no meta
parameters anywhere" a correct and strict invariant with no exception list to
maintain. The tensor selection also now downloads the embedding matrix for a
final stage only when the checkpoint has tied embeddings, since that is the only
case where the head weight has no separate entry in the file.