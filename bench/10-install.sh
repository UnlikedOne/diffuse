#!/usr/bin/env bash
set -euo pipefail

# Install diffuse on every benchmark node, following the exact procedure that
# already works on the public sentinel: clone into /root/diffuse, cargo build,
# venv, editable install, then gen_proto.sh from the repository.
# Idempotent: safe to rerun on machines that are half configured.

KEY="${KEY:-$HOME/.ssh/diffuse_oracle}"
SSH_OPTS="-n -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

mapfile -t hosts < <(hcloud server list -l bench=diffuse -o columns=ipv4 | tail -n +2 | tr -d ' ')

echo "installing on ${#hosts[@]} hosts"
echo

for ip in "${hosts[@]}"; do
  echo "=== $ip ==="
  ssh $SSH_OPTS root@"$ip" bash -s <<'REMOTE'
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export HOME=/root
export PATH=/root/.cargo/bin:$PATH

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null
  export PATH=/root/.cargo/bin:$PATH
fi

apt-get update -qq
apt-get install -y -qq build-essential pkg-config libssl-dev protobuf-compiler \
  python3-venv python3-dev git curl >/dev/null

if [ ! -d /root/diffuse/.git ]; then
  rm -rf /root/diffuse
  git clone --quiet https://github.com/UnlikedOne/diffuse.git /root/diffuse
else
  cd /root/diffuse && git fetch --quiet origin && git reset --quiet --hard origin/main
fi

cd /root/diffuse
cargo build --release 2>&1 | tail -2

WORKER_DIR=/root/diffuse/worker
if [ ! -x "${WORKER_DIR}/.venv/bin/python" ]; then
  python3 -m venv "${WORKER_DIR}/.venv"
fi

source "${WORKER_DIR}/.venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet "setuptools<82"
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet transformers safetensors grpcio grpcio-tools protobuf numpy psutil huggingface_hub
pip install --quiet -e "$WORKER_DIR"
( cd "$WORKER_DIR" && bash scripts/gen_proto.sh )
deactivate

test -x /root/diffuse/target/release/diffuse
test -f "${WORKER_DIR}/diffuse_worker/data_pb2.py"
test -f "${WORKER_DIR}/diffuse_worker/data_pb2_grpc.py"
echo "INSTALL OK $(hostname)"
REMOTE
done

echo
echo "all hosts installed"
