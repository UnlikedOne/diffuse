# FAQ

Straight answers to the questions people actually ask.

## General

### Is Diffuse free?

Yes. It is open source under AGPL-3.0. There is no account, no API key, and no
paid tier. You run a binary and you are on the network.

### Do I need a GPU?

No. The worker runs on CPU by default and works on ordinary machines, including
laptops and a Raspberry Pi. A GPU makes a node faster but is not required.

### Is it fast?

No, and it does not pretend to be. A model split across ordinary machines over the
public internet is slower than a single datacenter GPU. On the measured three-node
run, generation ran at roughly 1.2 tokens per second. The point of Diffuse is
privacy and decentralization, not speed. See the [benchmarks](/benchmarks).

### Which models work?

Dense text transformers, such as most instruction-tuned chat models. Multimodal,
mixture-of-experts, and state-space models are unsupported or unproven. See
[choosing a model](/guides/choosing-a-model).

## Privacy

### Can a node read my prompt?

On the default path, the node that runs the first slice receives your token ids,
which are your prompt. Traffic is encrypted in transit, so the network cannot read
it, but the entry node can. A private mode that keeps the first layers on your
machine is in progress. Read the honest [threat model](/privacy) before trusting
it with anything sensitive.

### Does Diffuse hide that I am using it?

No. It hides what you say on the wire, not that you are talking. Your IP is visible
to the node you connect to, exactly as with any network service.

### Are my prompts logged anywhere?

There is no central server, so nothing is logged centrally. A malicious entry node
could log what it receives, which is why strong privacy needs local first layers.

## Running a node

### What does hosting cost me?

Bandwidth, some memory, and CPU while a request flows through your slice. You
choose how much by the size of the slice you take, which the planner fits to your
machine.

### Can I run a node behind my home router?

Yes. If your node cannot accept inbound connections, it serves through an
encrypted [relay](/concepts/nat-relay) automatically.

### Will hosting download the whole model?

No, only the shards that contain your slice, though shard boundaries mean you may
download one or two shards. See [choosing a model](/guides/choosing-a-model).

### How do I keep a node running after I close the terminal?

```bash
nohup diffuse host --model Qwen/Qwen2.5-0.5B-Instruct > ~/.diffuse/host.log 2>&1 &
```

## Using it

### Can I use it with my existing chat app?

Yes. Run `diffuse serve` and point any OpenAI-compatible client (LibreChat, Open
WebUI, Continue, the SDKs) at `http://localhost:8080/v1`. See the
[server guide](/guides/server).

### Can I run my own private network?

Yes. Point your nodes and clients at your own sentinel with `--bootstrap`. See
[self-hosting a network](/self-hosting).

### Why did my query fail with "model is incomplete"?

Some layer range of that model is held by nobody, so it cannot be served end to
end. The error names the missing slices. Wait for them to come online or host one
yourself.

## Troubleshooting

### The command hangs or the worker will not start.

The Python worker imports torch before opening its port. Diffuse now retries for
up to 30 seconds and prints a clear error naming the cause. See
[troubleshooting](/troubleshooting).

### A second run fails to bind a port.

Only one node or client can use the fixed ports per machine. Stop the previous
process. See [troubleshooting](/troubleshooting).
