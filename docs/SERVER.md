# Running a node on a server

This is the procedure that works, written down so it does not have to be
rediscovered. It is what the public sentinel runs and what every benchmark node
has used.

## Requirements

Linux x86_64. Ubuntu 24.04 is what has been tested. At least 4 GB of RAM for a
small model, 16 GB to hold a useful slice of a 7B model. Disk space matters more
than people expect: a node currently downloads the full model even when it serves
only part of it, so budget the full model size plus room to spare.

Ports 9440 and 10440 must be open inbound if you want the node to serve directly.
The compute port is always the gossip port plus 1000. A node that cannot accept
inbound connections will still work, routed through a sentinel relay, but with
extra latency.

## Install

```bash
apt-get update
apt-get install -y python3-venv python3-dev git curl
curl -fsSL https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
```

The installer fetches the binary from the latest release, clones the worker
sources, builds a virtualenv, installs PyTorch for CPU, and generates the
protobuf stubs. The stub generation step is easy to miss when installing by hand
and its absence produces a worker that fails to import `data_pb2`.

Verify:

```bash
diffuse --help
ls ~/.diffuse/worker/diffuse_worker/data_pb2.py
```

The version string printed by `diffuse --version` is currently hardcoded and does
not track releases, so it is not a useful check. To confirm you have the current
binary:

```bash
curl -sL https://github.com/UnlikedOne/diffuse/releases/latest/download/diffuse-linux-x86_64 -o /tmp/check
sha256sum /tmp/check ~/.local/bin/diffuse
```

Matching hashes mean the installed binary is the current release.

## Building from source instead

If you want a specific commit rather than the latest release:

```bash
apt-get install -y build-essential pkg-config libssl-dev protobuf-compiler python3-venv git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

git clone https://github.com/UnlikedOne/diffuse.git ~/diffuse
cd ~/diffuse
cargo build --release

python3 -m venv worker/.venv
source worker/.venv/bin/activate
pip install --upgrade pip
pip install "setuptools<82"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers safetensors grpcio grpcio-tools protobuf numpy psutil huggingface_hub
pip install -e worker
( cd worker && bash scripts/gen_proto.sh )
deactivate
```

Building from source puts the worker in `~/diffuse/worker` rather than
`~/.diffuse/worker`, so the daemon needs to be told where it is:

```bash
export DIFFUSE_WORKER_DIR=$HOME/diffuse/worker
```

Forgetting this produces `Error: No such file or directory` when the daemon tries
to spawn the worker.

## Running

In the foreground, to see what is happening:

```bash
diffuse host --model Qwen/Qwen2.5-0.5B-Instruct \
  --spawn-worker \
  --public-addr YOUR_PUBLIC_IP:9440
```

`--public-addr` matters on a server with a public IP. Without it the node relies
on the sentinel probe to discover its address, which works, but stating it
explicitly is clearer and avoids a round trip.

Detached, surviving a closed terminal:

```bash
nohup diffuse host --model Qwen/Qwen2.5-0.5B-Instruct --spawn-worker \
  --public-addr YOUR_PUBLIC_IP:9440 > ~/host.log 2>&1 &
```

## Running under systemd

For a node that restarts on boot and on failure. This is what the public sentinel
uses.

```bash
cat > /etc/systemd/system/diffuse.service << 'UNIT'
[Unit]
Description=Diffuse node
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/diffuse
Environment=DIFFUSE_WORKER_DIR=/root/diffuse/worker
Environment=PATH=/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=RUST_LOG=info
ExecStart=/root/diffuse/target/release/diffuse host --model Qwen/Qwen2.5-0.5B-Instruct --worker http://127.0.0.1:50051 --listen 0.0.0.0:9440 --public-addr YOUR_PUBLIC_IP:9440 --spawn-worker
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable diffuse
systemctl start diffuse
journalctl -u diffuse -f
```

Adjust the paths if you installed with the installer rather than from source:
the binary is at `~/.local/bin/diffuse` and the worker at `~/.diffuse/worker`.

`RUST_LOG=info` is worth setting. Without it the reachability probe, relay
attachment and peer pruning are all silent, which makes diagnosis guesswork.

## Gated models

Models that require accepting a licence need a Hugging Face token. The worker
reads `HF_TOKEN` from its environment, and inherits the daemon's environment, so:

```bash
export HF_TOKEN=hf_your_token
```

Under systemd, add `Environment=HF_TOKEN=hf_your_token` to the unit.

An unauthenticated node still works for open models but is rate limited by
Hugging Face, which matters when several nodes download at once.

## Updating

With the installer:

```bash
curl -fsSL https://raw.githubusercontent.com/UnlikedOne/diffuse/main/install.sh | bash
```

From source:

```bash
cd ~/diffuse
git fetch origin && git reset --hard origin/main
source $HOME/.cargo/env
cargo build --release
systemctl restart diffuse
```

Note that the worker is Python and the daemon is Rust. A change to the worker
takes effect on restart with no rebuild. A change to the daemon needs
`cargo build --release`. If you edit worker sources in a checkout while the
daemon uses the installed copy at `~/.diffuse/worker`, the edit does nothing
until you copy it across and kill the running worker:

```bash
cp worker/diffuse_worker/*.py ~/.diffuse/worker/diffuse_worker/
pkill -f diffuse_worker
```

## Disk

Models accumulate in the Hugging Face cache. Check what is there:

```bash
du -sh ~/.cache/huggingface/hub/models--* | sort -rh
```

Remove one you no longer serve:

```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct
```

Failed downloads leave partial blobs that occupy space and serve no purpose:

```bash
find ~/.cache/huggingface/hub -name "*.incomplete" -delete
```

A load that fails with `No space left on device` is almost always this.

## Diagnosing a node that will not start

Worker logs go to `/dev/null` when the daemon spawns it, which hides Python
errors. To see them, run the worker yourself in one terminal:

```bash
cd ~/.diffuse/worker
DIFFUSE_WORKER_PORT=50051 ./.venv/bin/python -m diffuse_worker
```

and start the daemon without `--spawn-worker` in another. The Python traceback
will be visible.

Common failures and what they mean:

`worker never came up: transport error` usually means the worker died during
load, most often out of memory. Check `dmesg -T | grep -i oom`.

`No such file or directory` when spawning the worker means `DIFFUSE_WORKER_DIR`
points somewhere wrong, or the worker was never installed.

`cannot import name 'data_pb2'` means the protobuf stubs were not generated. Run
`bash scripts/gen_proto.sh` from inside the worker directory with its virtualenv
active.

`machine too small to hold any slice` after a restart often means an orphaned
worker from the previous run is still holding the model in memory. Check with
`ps aux | grep diffuse_worker` and kill it.