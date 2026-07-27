# How the network is made fast

A model split across machines pays for every token twice: once in computation,
once in the network between the pieces. This page explains what Diffuse does
about both, with the numbers that justify each choice.

Every figure here was measured, and the conditions are stated. A number without
its conditions is not worth having.

## Only the answer travels, not the whole vocabulary

The last slice of a model produces a score for every word it knows. For
Qwen2.5-0.5B that is 151,936 numbers per token. Sending them all is easy and
wasteful: the client reads one row and throws the rest away.

Diffuse sends the top candidates instead.

| Per token, last hop | Before | After |
|---|---|---|
| Qwen2.5-0.5B | 608 KB | **130 B** |
| Mistral-Small-24B, 8 nodes | 652 KB total | **70 KB total** |
| MusicGen-small, 4 audio streams | 32.8 KB | **74 B** |

At 100 Mbit/s that is roughly 45 ms per token given back on the small model,
and it costs nothing: ties are broken the same way `argmax` breaks them, so the
answer is byte-for-byte what it was.

The same logic applies to the prefill. Feeding a 500-token prompt used to ship
262 MB of scores that nobody read, because the model only needs the last row.
Now it ships the last row.

A model that answers on several streams at once gets one shortlist per stream
when it decodes greedily. MusicGen writes four audio codebooks per step and
samples rather than taking the best, so it gets something shorter still: the
draw happens on the last slice, where the whole distribution is, and four token
ids come back. Seventy-four bytes in place of thirty-two kilobytes.

## Activations at half the width

Hidden states travel in bfloat16 rather than float32 — the same precision the
model computes in, so nothing is lost by sending it. That halves every hop.

A node that predates this is fed float32, because it would otherwise read half
a tensor and crash. Nodes announce what they speak in their own encrypted
answer, which is the one statement about themselves that a relay cannot strip
and a stranger cannot forge.

## Activations go node to node, not back through you

The obvious way to run a pipeline is for the client to call each node in turn.
It is also the slow way: with eight nodes, a token makes sixteen trips over
*your* connection.

Diffuse hands the route to the first node instead. Each node computes its
slice, encrypts the result for the next one, and passes it along; the answer
comes back down the chain. Only two of those trips touch your link.

Measured on two nodes with a 25 ms client link, twelve tokens:

| | Time |
|---|---|
| Client relays every hop | 4059 ms |
| Nodes pass to each other | **2707 ms** |

::: warning This gain comes entirely from asymmetry
The number of trips does not change. What changes is that most of them move
between nodes, which are usually close to each other, instead of over a home
connection. **If your client sits in the same datacenter as the nodes, chaining
gains you nothing.** If you benchmark from a machine next to your servers, you
are measuring the case where this optimisation does not apply.
:::

Encryption stays hop by hop, which changes nothing about what a node can see: a
node already handles the plaintext activations it computes on.

## Several conversations in one pass

A node serving eight people was running their tokens one after another. Grouping
them into a single pass through the layers amortises the cost of reading the
weights, which is what actually dominates on CPU.

Eight concurrent sessions, measured through the encrypted compute plane:

| | Aggregate | Median latency |
|---|---|---|
| One at a time | 23.1 tokens/s | 344 ms |
| Grouped | **44.4 tokens/s** | **179 ms** |

A single stream pays nothing for this. The scheduler never waits for a batch to
fill: it takes what is queued and goes. Requests pile up naturally while a
batch computes, so grouping happens under load and not otherwise.

Sessions of different lengths are padded to a common length with a mask. With
equal lengths the result is bit-identical to running them separately; with
padding it differs by a few units in the last place of bfloat16, from a longer
accumulation, not from anything being wrong.

## What was not optimised, and why you should know

**Guidance and sampling are done where the numbers are.** A model that asks for
classifier-free guidance runs both branches, and the last slice combines them
before anything is shortened — it is the only place holding both. Sampling
happens there too, over the whole distribution: drawing from a shortlist is not
the same draw, because truncating moves the mass onto the strongest candidates
and the answer comes out harder than the model meant it. Four token ids travel
instead of four vocabularies.

**Round-trip latency is untouched.** Chaining moves trips off your link; it does
not remove them. If your nodes are spread across continents, the round trips
between them are your floor, and no amount of compression will move it. Measure
before assuming bandwidth is your problem: after the changes above, a 24B model
on eight nodes sends 70 KB per token, which is under 6 ms at 100 Mbit/s. If a
token takes 120 ms of network, that is latency, not volume.

## Measuring your own network

`bench/` holds the harness used for these numbers. Run it before and after any
change you make, on the topology you actually deploy.

One caveat worth repeating: the bundled benchmark runs the client **on the
sentinel**, inside the datacenter. That measures the LAN case. For what your
users will feel, run the client from somewhere else.
