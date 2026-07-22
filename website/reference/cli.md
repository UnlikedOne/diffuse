# CLI reference

Every Diffuse command, its flags, and defaults.

```bash
diffuse <command> [options]
diffuse --version
diffuse --help
```

## `chat`

Interactive chat with a model on the network.

| Flag | Default | Meaning |
|------|---------|---------|
| `--bootstrap <urls>` | built-in sentinels | sentinels to discover the network |
| `--memory` | off | keep conversation history across turns |

## `query`

Ask one question, print the answer, exit.

| Flag | Default | Meaning |
|------|---------|---------|
| `--model <id>` | required | model to query |
| `--prompt <text>` | required | prompt to send |
| `--bootstrap <urls>` | built-in sentinels | sentinels to discover the network |
| `--max-tokens <n>` | 80 | maximum tokens to generate |

## `models`

List the models currently hosted on the network.

| Flag | Default | Meaning |
|------|---------|---------|
| `--bootstrap <urls>` | built-in sentinels | sentinels to discover the network |

## `serve`

Run a local OpenAI-compatible HTTP server in front of the network.

| Flag | Default | Meaning |
|------|---------|---------|
| `--port <n>` | 8080 | port to listen on |
| `--host <addr>` | 127.0.0.1 | bind address, loopback only by default |
| `--model <id>` | none | default model when a request omits one |
| `--bootstrap <urls>` | built-in sentinels | sentinels to discover the network |

## `host`

Join the network: profile, pick a slice, load it, announce, and serve.

| Flag | Default | Meaning |
|------|---------|---------|
| `--model <id>` | required | model to serve a slice of |
| `--worker <url>` | `http://127.0.0.1:50051` | worker endpoint |
| `--listen <addr>` | `0.0.0.0:9440` | gossip listen address |
| `--bootstrap <urls>` | built-in sentinels | sentinels to join through |
| `--overhead <f>` | 0.3 | memory fraction held back as headroom |
| `--public-addr <addr>` | auto | address to advertise to peers |

## `plan`

Analyze this machine and show which slice it would host, without joining.

| Flag | Default | Meaning |
|------|---------|---------|
| `--model <id>` | required | model to analyze against |
| `--worker <url>` | `http://127.0.0.1:50051` | worker endpoint |
| `--overhead <f>` | 0.3 | memory fraction held back as headroom |
| `--bootstrap <urls>` | none | sentinels for a network view |

## Environment variables

| Variable | Meaning |
|----------|---------|
| `DIFFUSE_WORKER_DIR` | path to the Python worker (set by the installer) |
| `DIFFUSE_WORKER_PORT` | port the worker binds |
| `RUST_LOG` | log verbosity, for example `RUST_LOG=info` |
