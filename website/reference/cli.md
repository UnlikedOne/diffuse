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
| `--image <path>` | none | image to send with the prompt, repeatable |
| `--audio <path>` | none | audio clip to send, repeatable |
| `--video <path>` | none | video to send, repeatable |
| `--media <path>` | none | any attachment; kind read from the extension |
| `--bootstrap <urls>` | built-in sentinels | sentinels to discover the network |
| `--max-tokens <n>` | 80 | maximum tokens to generate |

Attachments are consumed on your machine; only activations leave. When the model
answers with something other than text, the file is written to the current
directory and its path printed. See [images, audio and video](/guides/multimodal).

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
| `--model <id>` | optional | model to serve a slice of; omit to open the [marketplace](/guides/marketplace) |
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


## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `HF_TOKEN` | none | Hugging Face token; unlocks gated models, stays on your machine |
| `DIFFUSE_WORKER_DIR` | installed copy, then `./worker` | which worker to run; see [troubleshooting](/troubleshooting#the-installed-worker-is-not-the-one-you-edited) |
| `DIFFUSE_WORKER_PORT` | `50051` | port the worker listens on |
| `DIFFUSE_WORKER_DEVICE` | auto | `cpu`, `cuda`, or `mps`; detected when unset |
| `DIFFUSE_WORKER_MAX_BATCH` | `8` | concurrent requests grouped into one pass; `1` disables |
| `DIFFUSE_WORKER_CONCURRENCY` | `16` | how many requests the worker serves at once |
| `DIFFUSE_WORKER_THREADS` | torch default | CPU threads for inference |
| `DIFFUSE_WORKER_CACHE` | Hugging Face default | where model weights are stored |
| `DIFFUSE_ASCII` | unset | plain symbols instead of Unicode, for limited terminals |
| `RUST_LOG` | `warn` | `info` shows routing and timing, `debug` shows every hop |
