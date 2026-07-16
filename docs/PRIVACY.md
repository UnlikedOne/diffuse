# Privacy

The reason Diffuse exists is that your prompt never leaves your machine in the
clear. This page explains exactly what that means, why it holds, and just as
importantly what it does not protect against. The goal is an honest threat
model, not a marketing claim.

## The claim

When you run a prompt through Diffuse, your own machine runs the first layers of
the model. The raw token embeddings of your prompt are consumed locally and
never transmitted. What leaves your machine is a hidden state, which is a tensor
of activations from a middle layer of the network, encrypted end to end for the
next node.

Other systems that split a model across peers typically send the prompt, or its
plain embeddings, to another machine. In those systems a peer can read what you
asked. In Diffuse a peer cannot, because it never receives the prompt.

## Why the prompt stays local

A transformer processes text by first turning tokens into embeddings, then
passing them through a stack of layers. Diffuse keeps the embedding step and the
first layers on your machine. By the time any data crosses the network, it is
already several layers deep into the model and no longer the text you typed.

The remote nodes each hold a middle slice of layers. They receive a hidden
state, run their slice, and pass the result on. They never hold the first layers,
so they never have the information needed to read the prompt directly.

## What a remote node actually sees

A node in the middle of the pipeline sees an encrypted hidden state addressed to
it, which it decrypts, processes, and re-encrypts for the next hop. In plaintext
form, inside that one node, it is a tensor of floating point activations. It is
not your text, and it is not your token ids.

A node also necessarily learns some metadata: that a request is passing through,
its rough size, and timing. It learns which model is being served, because it
loaded a slice of that model. It does not learn who you are from the computation
itself, though network level identifiers like your IP may be visible to the node
you connect to directly, exactly as with any network service.

## What this does not protect against

This is the part that most privacy claims leave out. Diffuse should not be sold
as more than it is.

A hidden state is not random noise. It is a deterministic function of your
prompt. The research literature on activation inversion and embedding inversion
shows that, under some conditions, an adversary who controls a node and knows the
model weights can attempt to reconstruct information about the input from
intermediate activations. Keeping the first layers local raises the bar,
because the earliest and most directly invertible representations never leave
your machine, but it does not reduce the risk to zero. A motivated adversary who
runs a node, has the weights, and invests effort may recover partial information
about your prompt. This is an open problem, not a solved one.

Diffuse does not hide that you are using Diffuse, does not hide which model you
are running, and does not hide the timing or size of your requests from the nodes
that carry them. It is not an anonymity system and does not route over Tor or
anything similar.

The transport encryption between nodes is a hand rolled scheme built on X25519
and ChaCha20-Poly1305. Those are sound primitives, but a hand rolled protocol
around them has not had the review that a standard transport has. The Enterprise
track is expected to move to mTLS partly for this reason.

## Honest summary

Diffuse gives you a real and unusual property: the machines that help run the
model never receive your prompt in the clear, unlike other distributed inference
systems. That is worth something concrete.

It does not give you cryptographic proof that your prompt cannot be partially
reconstructed by a hostile node willing to attack the intermediate activations,
and it does not anonymize you or hide your usage. If your threat model requires
those guarantees, Diffuse in its current form does not meet them, and it will
say so rather than pretend otherwise.