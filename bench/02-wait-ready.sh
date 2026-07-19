#!/usr/bin/env bash
set -euo pipefail

# Wait until cloud-init has finished building diffuse on every node.

KEY="${KEY:-$HOME/.ssh/diffuse_oracle}"
PREFIX="${PREFIX:-diffuse-bench}"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5"

mapfile -t hosts < <(hcloud server list -l bench=diffuse -o columns=ipv4 | tail -n +2)

echo "waiting for ${#hosts[@]} nodes"
for ip in "${hosts[@]}"; do
  printf "  %-16s " "$ip"
  for _ in $(seq 1 120); do
    if ssh $SSH_OPTS root@"$ip" "test -f /root/READY" 2>/dev/null; then
      echo "ready"
      break
    fi
    sleep 15
    printf "."
  done
done

echo
echo "all nodes reporting ready"
