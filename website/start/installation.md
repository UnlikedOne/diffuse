# Installation

Diffuse has two parts: the `diffuse` binary (Rust) and a Python worker that runs
the model on PyTorch. The installer sets up both.

Prebuilt binaries are published for **Linux** (x86_64 and ARM64), **macOS**
(Apple Silicon and Intel) and **Windows** (x86_64). None of them need a Rust
toolchain. Anything else builds from source, which is a single `cargo build`
plus the worker environment; all paths are covered below.

| Platform | Binary in the release |
|---|---|
| Linux x86_64 | `diffuse-linux-x86_64` |
| Linux aarch64 | `diffuse-linux-aarch64` |
| macOS Apple Silicon | `diffuse-macos-aarch64` |
| macOS Intel | `diffuse-macos-x86_64` |
| Windows x86_64 | `diffuse-windows-x86_64.exe` |

Each is published with a `.sha256` next to it, and the installers check it.

## Prerequisites

| Requirement | Why |
|-------------|-----|
| `python3` 3.10 or newer | runs the model worker |
| `git` | fetches the worker source |
| `curl` | downloads the binary (prebuilt path) |
| Rust toolchain | only on platforms with no prebuilt binary |
| ~4 GB free disk | PyTorch and a small model |

The worker downloads PyTorch on first setup, which can take a few minutes.

## Quick install (Linux and macOS)

One command. It picks the binary for your platform, verifies its checksum,
installs the worker, and sets up its Python environment.

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

The same command as Linux. It picks the Apple Silicon or the Intel binary from
`uname -m`, so there is nothing to choose.

```bash
brew install python git
curl -fsSL https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.sh | bash
```

::: tip Gatekeeper
The binary is not notarised, so macOS quarantines it on download. The installer
clears that flag for you. If you fetch the binary by hand instead, run
`xattr -d com.apple.quarantine ./diffuse` once.
:::

## Windows (PowerShell)

Install [Python 3](https://www.python.org/downloads/) and
[Git](https://git-scm.com/download/win), then run one line. No Rust toolchain:
the `.exe` is prebuilt.

```powershell
irm https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.ps1 | iex
```

It installs to `%USERPROFILE%\.local\bin\diffuse.exe`, puts the worker in
`%USERPROFILE%\.diffuse\worker`, sets `DIFFUSE_WORKER_DIR`, and adds the binary
directory to your user `PATH`. Open a new terminal afterwards so the shell picks
those up.

## ARM and Raspberry Pi

ARM64 has a prebuilt binary, so the quick installer works on Ampere, Graviton, a
Raspberry Pi 4/5 running a 64-bit OS, and anything else aarch64. Two notes:

- **PyTorch on ARM.** The standard `pip install torch` provides ARM64 CPU wheels.
  On a Raspberry Pi running a 64-bit OS this works out of the box; a 32-bit OS is
  not supported.
- **Memory.** A Raspberry Pi holds only a few layers. That is fine, that is the
  point. Run it as a `host` contributing a small slice rather than trying to
  serve a model alone.

```bash
sudo apt install -y python3 python3-venv python3-pip git curl
curl -fsSL https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.sh | bash
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
pip install -e "worker[multimodal,diffusion]"
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
