# Choosing a model to host

Not every model works with Diffuse, and the ones that do have very different
hardware requirements. This page tells you what to check before running
`diffuse host`, so you find out in seconds rather than after a 60 GB download.

## What works

Diffuse splits a model by layers: each node holds a contiguous run of decoder
layers and passes hidden states to the next node. That works when a model is a
plain stack of transformer layers that runs strictly front to back.

**Dense text models work.** Anything with a flat list of decoder layers, a single
embedding table at the front and a language modelling head at the back. Most
instruction tuned chat models fall into this category.

**Multimodal models do not work yet.** A model with separate vision or audio
towers keeps its layer count inside a nested `text_config` rather than at the top
of its config, and the loader does not look there. Attempting one currently fails
with an error like `'Gemma4Config' object has no attribute 'n_layer'`. The vision
and audio weights would also be downloaded and never used.

**Mixture of experts models are unproven.** A MoE layer holds many expert
sub-networks and a router that selects between them. If the experts belong to the
layer, slicing may work; if routing spans layers, it will not. No MoE model has
been verified end to end. They are also far heavier per layer, which usually puts
them out of reach of an ordinary machine anyway.

**State space models do not work.** Architectures in the Mamba family carry a
recurrent state between layers rather than a hidden state tensor. The pipeline
only forwards a tensor, so the state is lost at every boundary.

If the loader cannot find the layer list at all, you will see `cannot locate a
list of N transformer layers`. That means the model's structure does not match
what the slicer expects, and no amount of configuration will help.

## Checking a model before you commit

The capacity planner reads only metadata, downloads no weights, and answers in a
few seconds:

```bash
cd ~/.diffuse/worker
./.venv/bin/python -c "
from diffuse_worker.capacity import plan_capacity
p = plan_capacity('MODEL_ID', 0.3, load_dtype='bfloat16')
print(p['max_layers'], 'layers of', p['total_layers'])
print(round(p['avg_layer_bytes'] / 1e9, 2), 'GB per layer')
"
```

Three numbers matter.

`total_layers` is how many layers the model has. It tells you how many
contributors the model needs before it becomes servable.

`max_layers` is how many your machine can hold. Zero means the model is out of
reach: either the layers are too large, or the machine is too small.

`avg_layer_bytes` is the size of one layer. Below about 2 GB, an ordinary
16 GB machine can hold a useful slice. Above 8 GB it cannot hold even one.

## Disk, which is usually the real limit

Weights are published as safetensors shards, and a shard can only be downloaded
whole. If your layers span two shards, you download both, even if you need a
fraction of each.

A concrete example. DeepSeek-R1-Distill-Qwen-32B is about 65 GB across 8 shards.
A node taking layers 0 to 6 downloaded a single 8.3 GB shard: one eighth of the
model for seven of its sixty-four layers. A different slice, landing across two
shards, would have cost twice that.

So the rule of thumb is: budget at least one shard, and possibly two. Divide the
model's total size by its shard count to get the likely floor. A model published
as two enormous shards is a bad choice for a small machine even if its layers are
individually small.

Check what you have before starting:

```bash
df -h ~ | tail -1
du -sh ~/.cache/huggingface/hub/models--* 2>/dev/null | sort -rh
```

Old models accumulate. Remove ones you no longer serve.

## Gated models

Some models require accepting a licence on their Hugging Face page. Without a
token you will get a 403 during planning or download. Log in once:

```bash
hf auth login
```

Even for open models, an authenticated session gets better download rate limits,
which matters when several nodes fetch at the same time.

## Overhead

`--overhead` is the fraction of available memory Diffuse leaves alone. The
default is 0.3, meaning 70 percent of free memory may be used for weights and
the rest is kept for the KV cache, activations, PyTorch itself and the operating
system.

Raise it to be more conservative, or to deliberately hold a smaller slice:

```bash
diffuse host --model MODEL_ID --overhead 0.6
```

A higher overhead means fewer layers, which means more contributors are needed to
cover the model, which means a longer pipeline and more network hops per token.
Lower it only if you know the machine has memory to spare and is doing nothing
else.

## Being the first holder

When you are the first node to host a model, you take a slice starting at layer
0 and the network reports the model as present. It is not yet usable.

A model is only servable when every layer is covered. If you hold 7 layers of a
64 layer model, a client cannot run a query: 57 layers have nowhere to execute.
Other contributors have to join and fill the gaps, and each of them will take the
next uncovered range automatically.

The status shown next to a model reflects replication, not completeness, so a
model that is nowhere near covered can still appear alongside a reassuring bar.
Read the slice list underneath: it shows which ranges exist. Compare the highest
end layer against the model's real layer count from the capacity planner.

## Behind NAT

If your machine cannot accept inbound connections, which is the normal situation
on a home network, Diffuse detects it on startup and routes your compute through
a sentinel relay. You will see:

```
⚠ behind NAT, seen from outside as 203.0.113.42
✓ relaying compute through sentinel
```

Nothing is required from you. The relay adds two network legs per token, so a
relayed node is slower than one with a public address, but it works and the
sentinel cannot read what passes through it.