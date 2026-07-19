#!/usr/bin/env bash
set -euo pipefail

# Diffuse benchmark: provision eight CPX41 nodes in one Hetzner location.
# One sentinel that serves no layers, seven compute nodes.
# Every machine is billed until it is DELETED, not until it is powered off.

LOCATION="${LOCATION:-nbg1}"
TYPE="${TYPE:-cpx41}"
IMAGE="${IMAGE:-ubuntu-24.04}"
SSH_KEY="${SSH_KEY:-diffuse-sentinel}"
COUNT="${COUNT:-7}"
PREFIX="${PREFIX:-diffuse-bench}"

cloud_init=$(cat <<'CLOUDINIT'
#cloud-config
package_update: true
packages:
  - build-essential
  - pkg-config
  - libssl-dev
  - protobuf-compiler
  - python3-venv
  - python3-dev
  - git
  - curl
runcmd:
  - curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  - git clone https://github.com/UnlikedOne/diffuse.git /root/diffuse
  - bash -lc 'source $HOME/.cargo/env && cd /root/diffuse && cargo build --release'
  - python3 -m venv /root/diffuse/worker/.venv
  - /root/diffuse/worker/.venv/bin/pip install --upgrade pip
  - /root/diffuse/worker/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
  - /root/diffuse/worker/.venv/bin/pip install transformers safetensors huggingface_hub psutil grpcio grpcio-tools protobuf
  - touch /root/READY
CLOUDINIT
)

echo "creating sentinel"
hcloud server create \
  --name "${PREFIX}-sentinel" \
  --type "$TYPE" \
  --image "$IMAGE" \
  --location "$LOCATION" \
  --ssh-key "$SSH_KEY" \
  --user-data-from-file <(echo "$cloud_init") \
  --label role=sentinel \
  --label bench=diffuse

for i in $(seq -w 1 "$COUNT"); do
  echo "creating node $i"
  hcloud server create \
    --name "${PREFIX}-${i}" \
    --type "$TYPE" \
    --image "$IMAGE" \
    --location "$LOCATION" \
    --ssh-key "$SSH_KEY" \
    --user-data-from-file <(echo "$cloud_init") \
    --label role=compute \
    --label bench=diffuse
done

echo
echo "provisioned. addresses:"
hcloud server list -l bench=diffuse -o columns=name,ipv4,status
echo
echo "cloud-init still needs several minutes to build the binary."
echo "check readiness with: ./02-wait-ready.sh"
