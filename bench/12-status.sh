#!/usr/bin/env bash
set -euo pipefail

KEY="${KEY:-$HOME/.ssh/diffuse_oracle}"
SSH_OPTS="-n -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

sentinel_ip=$(hcloud server list -l role=sentinel -o columns=ipv4 | tail -n +2 | tr -d ' ')

echo "per host state"
mapfile -t hosts < <(hcloud server list -l bench=diffuse -o columns=name,ipv4 | tail -n +2)
for line in "${hosts[@]}"; do
  name=$(echo "$line" | awk '{print $1}')
  ip=$(echo "$line" | awk '{print $2}')
  printf "  %-24s %-16s " "$name" "$ip"
  ssh $SSH_OPTS root@"$ip" \
    "systemctl is-active diffuse | tr -d '\n'; echo -n '  '; \
     journalctl -u diffuse --no-pager 2>/dev/null | grep -oE 'taking slice [0-9]+:[0-9]+' | tail -1 | tr -d '\n'; \
     journalctl -u diffuse --no-pager 2>/dev/null | grep -oE 'Error.*' | tail -1" 2>/dev/null || echo "unreachable"
done

echo
echo "network as seen by the sentinel"
ssh $SSH_OPTS root@"$sentinel_ip" \
  "cd /root/diffuse && ./target/release/diffuse models --bootstrap http://${sentinel_ip}:9440" 2>/dev/null \
  || echo "  sentinel not answering yet"
