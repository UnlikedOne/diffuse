# Choosing a model

Not every model works with Diffuse, and the ones that do have very different
hardware costs. This page tells you what to check before running `diffuse host`,
so you find out in seconds rather than after a large download.

## What works

Diffuse splits a model by layers: each node holds a contiguous run of decoder
layers and passes hidden states to the next. That works when a model is a plain
stack of transformer layers that runs strictly front to back.

- **Dense text models work.** A flat list of decoder layers, one embedding table
  at the front, a language-modelling head at the back. Most instruction-tuned chat
  models fall here.
- **Models that read media work.** A vision or audio tower is an endpoint, not a
  stack to split: it rides with the slice holding the embeddings, and on a client
  that means your own machine. SmolVLM, Qwen2-VL and Voxtral have all been run
  end to end.
- **Encoder-decoders work.** Whisper and MusicGen have two stacks; the decoder is
  what gets split, and the encoder's output travels once per session. The encoder
  runs where the question is asked.

## What does not

- **U-Net diffusion does not work.** Stable Diffusion and its family connect
  across resolutions, so what passes between two points is several tensors at
  different scales rather than one. It does not fit a chain.
- **State space models do not work.** Mamba-style architectures carry a recurrent
  state between layers. The pipeline only forwards a tensor, so the state is lost
  at every boundary. This is an engineering gap rather than a barrier.
- **Mixture-of-experts is unproven.** Slicing may work if experts belong to a
  layer and fail if routing spans layers. None has been verified end to end, and
  they are heavy per layer anyway.
- **`trust_remote_code` is refused.** A checkpoint that ships its own Python
  would run on volunteer machines. That is a security decision, not a limitation.

If the loader cannot find the layer list, you will see `cannot locate a list of N
transformer layers`. That means the structure does not match what the slicer
expects, and no configuration will help.

## Check before you commit

The capacity planner reads only metadata, downloads no weights, and answers in
seconds:

```bash
cd ~/.diffuse/worker
./.venv/bin/python -c "
from diffuse_worker.capacity import plan_capacity
p = plan_capacity('MODEL_ID', 0.3, load_dtype='bfloat16')
print(p['max_layers'], 'layers of', p['total_layers'])
print(round(p['avg_layer_bytes'] / 1e9, 2), 'GB per layer')
"
```

Three numbers matter:

- **`total_layers`** is how many layers the model has, and therefore how many
  contributors it needs before it becomes servable.
- **`max_layers`** is how many your machine can hold. Zero means the model is out
  of reach.
- **`avg_layer_bytes`** is the size of one layer. Below about 2 GB an ordinary
  16 GB machine can hold a useful slice; above 8 GB it cannot hold even one.

## Disk is usually the real limit

Weights ship as safetensors shards, and a shard can only be downloaded whole. If
your layers span two shards, you download both even if you need a fraction of
each.

For example, a 32B model of about 65 GB across 8 shards: a node taking layers 0
to 6 downloaded a single 8.3 GB shard, one eighth of the model for seven of its
sixty-four layers. A slice landing across two shards would cost twice that.

Rule of thumb: budget at least one shard, possibly two. Divide the model's total
size by its shard count to get the likely floor. A model published as two enormous
shards is a poor choice for a small machine even if its layers are individually
small.

## A safe starting point

`Qwen/Qwen2.5-0.5B-Instruct` is small, dense, and known to work end to end. It is
the model used throughout this documentation and a good first target for hosting
or querying.

## Next

- [Host a node](/guides/host)
- [Slices and the pipeline](/concepts/pipeline)
