# Troubleshooting

Common failures and how to read them.

## The worker does not start

Symptoms: a command hangs at startup, or fails with a message that the local
tokenizer worker did not become ready.

The Python worker imports torch and transformers before it opens its port, which
can take longer than expected on a loaded machine or a cold disk cache. Diffuse
probes the worker with retries for up to 30 seconds and then reports a clear
error naming the port. If you see that error:

- Confirm the worker environment is intact:
  ```bash
  ~/.diffuse/worker/.venv/bin/python -c "import diffuse_worker; print('ok')"
  ```
- If that import fails, reinstall the worker (rerun the installer, or rebuild the
  venv from the [installation](/start/installation) steps).
- A `protobuf` version error means the generated stubs are out of step with the
  installed runtime. Regenerate them:
  ```bash
  cd ~/.diffuse/worker && bash scripts/gen_proto.sh
  ```

## Port already in use

Symptoms: the second run of a command fails to bind.

Diffuse uses fixed ports (9440, 10440, 50051, and 50099 for the client
tokenizer). Only one node or client can use them per machine at a time. A previous
process stopped with `Ctrl+C` normally cleans up its worker; if one was killed
harder, find and stop it:

```bash
pgrep -af diffuse_worker
pkill -f diffuse_worker
```

## Model not found on the network

`diffuse query` or a chat completion returns that the model is not present. No
node serves it. Check what is live:

```bash
diffuse models
```

Pick a model from that list, or [host](/guides/host) the one you want so it
becomes servable.

## Model is incomplete (503)

The model is present but some layer range is held by nobody, so it cannot be
served end to end. The error names the missing slices, for example
`no peer serves layers 6:64 (of 64 total)`. Either wait for those slices to come
online or host one yourself. See [replication](/concepts/replication).

## Behind NAT and not reachable

If your `host` node cannot accept inbound connections, it serves through a
sentinel [relay](/concepts/nat-relay) automatically. If it cannot reach a
sentinel, it cannot be reached at all. Confirm your `--bootstrap` sentinel is up
and that outbound connections to it are allowed.

## Command not found after install

`~/.local/bin` may not be on your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Add that line to your shell profile to make it permanent.

## Getting more detail

Raise the log level:

```bash
RUST_LOG=info diffuse chat
```

## The installed worker is not the one you edited

Symptoms: you changed the Python worker, rebuilt, and the change has no effect.
Or a model that should be supported fails with an error naming something you
already fixed, such as `'Idefics3Config' object has no attribute 'n_layer'`.

Diffuse looks for the worker in this order:

1. `$DIFFUSE_WORKER_DIR`
2. `~/.diffuse/worker` — where the installer puts it
3. `./worker` — the copy in a checkout

**The installed copy wins over your checkout.** If you installed Diffuse and
then cloned the repository to work on it, `diffuse host` runs the installed
Python against your freshly built binary, and the two disagree.

Point it at your checkout while you work:

```bash
export DIFFUSE_WORKER_DIR=/path/to/diffuse/worker
```

Or reinstall so the two match. Before deploying anything, reinstall: a new
binary driving an old worker is the most confusing failure mode there is,
because everything looks right.

::: tip How to tell which one is running
```bash
diffuse host --model Qwen/Qwen2.5-0.5B-Instruct 2>&1 | head -3
```
The worker logs its own path when it starts. If it is not the directory you are
editing, that is your answer.
:::

## A worker on the port outlives the command that started it

Symptoms: you restart a worker after a code change, and the old behaviour
persists.

A worker that fails to bind because the port is taken exits quietly, and the
daemon connects to whatever is already there — which may be an older process.
Check and clear before restarting:

```bash
pkill -9 -f "m diffuse_worker"
ss -ltn | grep 5005      # should print nothing
```

## Out of disk space while downloading a model

Symptoms: `No space left on device` partway through a download, sometimes
leaving the environment in a broken state if it interrupted a package install.

Two things surprise people:

**Some repositories ship the weights twice.** Mistral publishes both a
`consolidated.safetensors` and the sharded Hugging Face files. Downloading the
lot costs double. Fetch only what you need:

```python
snapshot_download("mistralai/Voxtral-Mini-3B-2507",
                  allow_patterns=["*.json", "model-*.safetensors"],
                  ignore_patterns=["consolidated*"])
```

**The transfer cache needs headroom.** Budget roughly twice a model's size while
it downloads, then reclaim it:

```bash
rm -rf ~/.cache/huggingface/xet     # transfer cache, safe to remove
rm -rf ~/.cache/pip                 # safe to remove
cargo clean --profile dev           # rebuildable
```

Your downloaded models live in `~/.cache/huggingface/hub`. Removing one only
costs you the time to fetch it again.

## torchvision imports but nothing works

Symptoms: `operator torchvision::nms does not exist`, or an image processor that
refuses to load.

`torchvision` is compiled against a specific `torch`. Installing the latest one
next to an older torch produces a package that imports and then fails on first
use. Install the matching version without letting pip touch torch:

```bash
.venv/bin/pip install --no-deps "torchvision==0.27.1"   # for torch 2.12
.venv/bin/python -c "import torchvision; from torchvision.transforms import functional"
```

If that second line runs, the pair is compatible.

## A Mistral model has no chat template

Symptoms: `Cannot use chat template functions because tokenizer.chat_template is
not set`.

Mistral repositories ship a `tekken.json` rather than a `tokenizer_config.json`.
Their processor needs `mistral-common` to build a template:

```bash
.venv/bin/pip install mistral-common
```

## The node claims a slice that does not exist

Symptoms: `end_layer 32 exceeds total 28`.

The capacity planner counted an encoder's blocks as decoder layers. This is
fixed for the models in [model support](/concepts/model-support); if you hit it
on a new architecture, it means the encoder is named in a way the rule does not
recognise. Report the model id — the fix is one line, and it belongs in the
rule rather than in a list.

## An answer that is right for the wrong reason

Not an error, but worth knowing before you trust a result.

Small models often cannot read what you think you are asking about. Asked which
direction a shape moves in a video, a 2B vision model answered "right" for a
shape moving right — and gave the same answer for the mirrored clip. The model
running whole, with no Diffuse involved, did the same.

Before concluding that Diffuse changed an answer, run the model whole on one
machine and compare. That is the only comparison that means anything, and it is
how every claim in these docs was checked.
