# Installation

Diffuse has two parts: the `diffuse` binary (Rust) and a Python worker that runs
the model on PyTorch. The installer sets up both.

A prebuilt binary is published for **Linux x86_64**. Every other platform builds
from source, which is a single `cargo build` plus the worker environment. All
paths are covered below.

## Prerequisites

| Requirement | Why |
|-------------|-----|
| `python3` 3.10 or newer | runs the model worker |
| `git` | fetches the worker source |
| `curl` | downloads the binary (prebuilt path) |
| Rust toolchain | only when building from source |
| ~4 GB free disk | PyTorch and a small model |

The worker downloads PyTorch on first setup, which can take a few minutes.

## Quick install (Linux x86_64)

One command. It downloads the binary, installs the worker, and sets up its
Python environment.

```bash
curl -fsSL https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.sh | bash
```

::: tip Read before you run
The script is [`install.sh`](https://github.com/UnlikedOne/diffuse/blob/main/install.sh).
Read it, then pipe it to bash.
:::

It installs to:

- `~/.local/bin/diffuse` — the binary
- `~/.diffuse/worker` — the Python worker and its virtual environment

If `~/.local/bin` is not on your `PATH`, add it:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Linux distributions

Install the prerequisites, then run the quick installer above.

::: code-group

```bash [Debian / Ubuntu]
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git curl build-essential
curl -fsSL https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.sh | bash
```

```bash [Fedora / RHEL]
sudo dnf install -y python3 python3-pip git curl gcc gcc-c++ make
curl -fsSL https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.sh | bash
```

```bash [Arch]
sudo pacman -S --needed python git curl base-devel
curl -fsSL https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.sh | bash
```

:::

## macOS

There is no prebuilt macOS binary yet, so build from source. Works on both Intel
and Apple Silicon.

```bash
brew install rust python git
git clone https://github.com/UnlikedOne/diffuse.git
cd diffuse
cargo build --release
```

Then set up the worker:

```bash
python3 -m venv worker/.venv
source worker/.venv/bin/activate
pip install --upgrade pip
pip install torch transformers safetensors grpcio grpcio-tools protobuf numpy psutil huggingface_hub
pip install -e worker
( cd worker && bash scripts/gen_proto.sh )
deactivate
```

Point Diffuse at the worker and binary:

```bash
export DIFFUSE_WORKER_DIR="$PWD/worker"
export PATH="$PWD/target/release:$PATH"
diffuse chat
```

::: tip Apple Silicon
Install the standard `torch` wheel. Diffuse runs the worker on CPU by default,
which is portable across architectures.
:::

## Windows (PowerShell)

Build from source with the PowerShell helper. Install
[Rust](https://rustup.rs), [Python 3](https://www.python.org/downloads/), and
[Git](https://git-scm.com/download/win) first.

```powershell
irm https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.ps1 | iex
```

Or do it by hand:

```powershell
git clone https://github.com/UnlikedOne/diffuse.git
cd diffuse
cargo build --release

python -m venv worker\.venv
worker\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch transformers safetensors grpcio grpcio-tools protobuf numpy psutil huggingface_hub
pip install -e worker
cd worker; bash scripts\gen_proto.sh; cd ..
deactivate

$env:DIFFUSE_WORKER_DIR = "$PWD\worker"
$env:PATH = "$PWD\target\release;$env:PATH"
diffuse chat
```

## ARM and Raspberry Pi

Diffuse builds natively on ARM64 (Apple Silicon, Ampere, Raspberry Pi 4/5, and
the like). Build from source as above. Two notes:

- **PyTorch on ARM.** The standard `pip install torch` provides ARM64 CPU wheels.
  On a Raspberry Pi running a 64-bit OS this works out of the box; a 32-bit OS is
  not supported.
- **Memory.** A Raspberry Pi holds only a few layers. That is fine, that is the
  point. Run it as a `host` contributing a small slice rather than trying to
  serve a model alone.

```bash
sudo apt install -y python3 python3-venv python3-pip git curl build-essential
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://github.com/UnlikedOne/diffuse.git && cd diffuse
cargo build --release
```

## Build from source (any platform)

The universal path, summarized:

```bash
git clone https://github.com/UnlikedOne/diffuse.git
cd diffuse

cargo build --release

python3 -m venv worker/.venv
source worker/.venv/bin/activate
pip install --upgrade pip
pip install torch transformers safetensors grpcio grpcio-tools protobuf numpy psutil huggingface_hub
pip install -e worker
( cd worker && bash scripts/gen_proto.sh )
deactivate

export DIFFUSE_WORKER_DIR="$PWD/worker"
export PATH="$PWD/target/release:$PATH"
```

`DIFFUSE_WORKER_DIR` tells the binary where the worker lives. The installer sets
this up for you; from a source build you set it yourself.

## Verify

```bash
diffuse --version
diffuse models
```

`diffuse --version` prints the build version derived from the Git tag.
`diffuse models` lists what the network is serving right now.

## Next

- [Quickstart](/start/quickstart) to send your first prompt.
- [Troubleshooting](/troubleshooting) if the worker will not start.
