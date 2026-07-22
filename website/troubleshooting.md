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
