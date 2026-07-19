#!/usr/bin/env bash
set -euo pipefail

KEY="${KEY:-$HOME/.ssh/diffuse_oracle}"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

sentinel_ip=$(hcloud server list -l role=sentinel -o columns=ipv4 | tail -n +2 | tr -d ' ')

echo "network state as seen by the sentinel"
echo
ssh $SSH_OPTS root@"$sentinel_ip" bash -lc "'
  export PATH=\$HOME/.cargo/bin:\$PATH
  cd /root/diffuse
  ./target/release/diffuse models --bootstrap http://${sentinel_ip}:9440
'"

echo
echo "per node slice assignment"
mapfile -t nodes < <(hcloud server list -l role=compute -o columns=ipv4 | tail -n +2 | tr -d ' ')
for ip in "${nodes[@]}"; do
  printf "  %-16s " "$ip"
  ssh $SSH_OPTS root@"$ip" "grep -oE 'taking slice [0-9]+:[0-9]+' /root/node.log | tail -1" 2>/dev/null || echo "not started"
done
