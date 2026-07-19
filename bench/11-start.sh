#!/usr/bin/env bash
set -euo pipefail

# Start every node as a systemd service, mirroring the unit that already runs
# on the public sentinel. systemd owns the process, so nothing dies when the
# ssh session closes and nothing has to be detached by hand.

KEY="${KEY:-$HOME/.ssh/diffuse_oracle}"
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
OVERHEAD="${OVERHEAD:-0.6}"
SENTINEL_OVERHEAD="${SENTINEL_OVERHEAD:-0.95}"
HF_TOKEN="${HF_TOKEN:-}"
SSH_OPTS="-n -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

sentinel_ip=$(hcloud server list -l role=sentinel -o columns=ipv4 | tail -n +2 | tr -d ' ')
mapfile -t nodes < <(hcloud server list -l role=compute -o columns=ipv4 | tail -n +2 | tr -d ' ')

write_unit() {
  local ip="$1" overhead="$2" bootstrap="$3"
  local bootstrap_arg=""
  if [ -n "$bootstrap" ]; then
    bootstrap_arg="--bootstrap ${bootstrap}"
  fi

  ssh $SSH_OPTS root@"$ip" bash -s <<REMOTE
set -euo pipefail
cat > /etc/systemd/system/diffuse.service <<'UNIT'
[Unit]
Description=Diffuse benchmark node
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/diffuse
Environment=DIFFUSE_WORKER_DIR=/root/diffuse/worker
Environment=PATH=/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=RUST_LOG=info
Environment=HF_TOKEN=${HF_TOKEN}
ExecStart=/root/diffuse/target/release/diffuse host --model ${MODEL} --worker http://127.0.0.1:50051 --listen 0.0.0.0:9440 --public-addr ${ip}:9440 --overhead ${overhead} ${bootstrap_arg} --spawn-worker
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --quiet diffuse
systemctl restart diffuse
sleep 3
systemctl is-active diffuse
REMOTE
}

echo "sentinel ${sentinel_ip} (overhead ${SENTINEL_OVERHEAD}, no bootstrap)"
write_unit "$sentinel_ip" "$SENTINEL_OVERHEAD" ""

echo "waiting for the sentinel to accept gossip"
sleep 25

for ip in "${nodes[@]}"; do
  echo "node ${ip} (overhead ${OVERHEAD})"
  write_unit "$ip" "$OVERHEAD" "http://${sentinel_ip}:9440"
  sleep 5
done

echo
echo "cluster starting, weights are downloading"
echo "follow one node with:"
echo "  ssh -i ${KEY} root@${nodes[0]} journalctl -u diffuse -f"
