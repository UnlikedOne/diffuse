# Privacy and threat model

This page is an honest threat model, not a marketing claim. It explains what
Diffuse protects, what it does not, and where the line sits today.

## What is protected

- **The wire.** Every hop between a client and a serving node is sealed end to end
  with X25519 and ChaCha20-Poly1305. A network eavesdropper, and a relay
  forwarding for a NAT'd node, sees ciphertext and flow metadata only, never the
  data.
- **Centralization.** There is no company in the middle that receives all of your
  prompts and could log them. Trust is spread across independent nodes rather than
  concentrated in one operator.
- **Identity.** Keys are bound to node identities from signed gossip, so there is
  no certificate authority to trust and peers cannot forge one another.

## What a serving node sees

This is the part most distributed-inference projects gloss over, so it is stated
plainly here.

A node in the **middle** of the pipeline decrypts an activation tensor addressed
to it, runs its slice, and re-encrypts the result. In plaintext, inside that one
node, the data is a tensor of floating-point activations several layers deep into
the model. Turning that back into your words is hard.

The node holding the **first slice** is different. It embeds your tokens, which
means it receives your token ids. On the default path, those token ids are your
prompt, and they can be decoded back to your exact text. Transport encryption
protects them from the network, but the entry node itself sees them.

So today, on the default path, the honest statement is: **your prompt is encrypted
in transit and no single party sees all of your traffic, but the node that runs
the first slice can read your prompt.**

## Strong privacy: keeping the first layers local

The way to remove trust in the entry node is to run the embedding and the first
few layers on your own machine, so that what leaves is a hidden state rather than
your token ids. No node then receives the prompt.

This is opt-in by design, because it requires your machine to download and hold
the head of the model (the embedding table plus a couple of layers), which is a
real cost. A private mode that does exactly this is in progress. When enabled, the
strong guarantee holds: the raw prompt and token ids never leave your device, and
the entry node sees only an intermediate activation.

::: tip The tradeoff
A thin client that holds no weights cannot, by construction, hide the prompt from
the entry node without heavyweight cryptography. Strong privacy therefore costs a
download. Diffuse keeps the thin client as the default and makes local first
layers an explicit choice.
:::

## What is not protected

- **That you are talking.** Diffuse does not mask your IP by default. The node you
  connect to directly can see your address, exactly as with any network service.
- **Metadata.** A node learns that a request passed through, its rough size, and
  timing. It learns which model is being served, because it loaded a slice of it.
- **Traffic analysis.** Timing and size patterns are not obfuscated.

## In one line

Diffuse hides **what you say** on the wire and spreads trust across many nodes. It
does not hide **that you are talking**, and on the default path the entry node can
read your prompt until you keep the first layers local. Everything here is made
difficult, not proven impossible.
