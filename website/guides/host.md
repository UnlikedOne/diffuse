# Host a node

Hosting is how the network exists. When you run `diffuse host`, your machine holds
a slice of a model and serves compute to clients. This is contribution, not
consumption.

```bash
diffuse host --model Qwen/Qwen2.5-0.5B-Instruct
```

Your node profiles its hardware, takes the slice the network needs most, announces
it over gossip, and starts serving. It runs in the foreground so you can watch the
logs. `Ctrl+C` stops it cleanly and shuts down its worker.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--model <id>` | required | the model to serve a slice of |
| `--worker <url>` | `http://127.0.0.1:50051` | the worker endpoint |
| `--listen <addr>` | `0.0.0.0:9440` | gossip listen address |
| `--bootstrap <urls>` | built-in sentinels | sentinels to join through |
| `--overhead <f>` | `0.3` | memory fraction held back as headroom |
| `--public-addr <addr>` | auto | the address to advertise to peers |

## Behind a NAT or firewall

Diffuse handles it. On startup your node checks whether it can accept inbound
connections. If it cannot, it automatically serves its compute through a sentinel
[relay](/concepts/nat-relay), so it still contributes rather than only consuming.
The relayed compute stays encrypted end to end; the relay sees only ciphertext.

## Keep it running

To keep contributing after you close the terminal, run it detached:

```bash
nohup diffuse host --model Qwen/Qwen2.5-0.5B-Instruct > ~/.diffuse/host.log 2>&1 &
```

Watch it with `tail -f ~/.diffuse/host.log`, stop it with `pkill diffuse`.

## How much you need

You do not need to hold a whole model. A node holding three layers of a 64-layer
model is a useful contribution. Diffuse measures each model's real memory cost and
hands you a slice that fits. A small machine takes a small slice; that is the
design, not a limitation. See [choosing a model](/guides/choosing-a-model).

## Next

- [Choosing a model](/guides/choosing-a-model)
- [Replication and healing](/concepts/replication)
