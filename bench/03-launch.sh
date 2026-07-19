#!/usr/bin/env bash
set -euo pipefail

# Start the sentinel, then every compute node against it.
# The cluster is isolated: nodes bootstrap only on the benchmark sentinel,
# never on the public one, so no outside peer can pollute the measurements.

KEY="${KEY:-$HOME/.ssh/diffuse_oracle}"
MODEL="${MODEL:-mistralai/Mistral-Small-24B-Instruct-2501}"
OVERHEAD="${OVERHEAD:-0.3}"
HF_TOKEN="${HF_TOKEN:-}"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

sentinel_ip=$(hcloud server list -l role=sentinel -o columns=ipv4 | tail -n +2 | tr -d ' ')
mapfile -t nodes < <(hcloud server list -l role=compute -o columns=ipv4 | tail -n +2 | tr -d ' ')

echo "sentinel: $sentinel_ip"
echo "nodes:    ${#nodes[@]}"
echo "model:    $MODEL"
echo

remote_env="export HF_TOKEN='${HF_TOKEN}'; export PATH=\$HOME/.cargo/bin:\$PATH;"

echo "starting sentinel"
ssh $SSH_OPTS root@"$sentinel_ip" bash -lc "'
  ${remote_env}
  cd /root/diffuse
  pkill -f diffuse || true
  nohup ./target/release/diffuse host \
    --model ${MODEL} \
    --spawn-worker \
    --public-addr ${sentinel_ip}:9440 \
    --overhead 0.95 \
    > /root/sentinel.log 2>&1 &
'"

sleep 20

for ip in "${nodes[@]}"; do
  echo "starting node $ip"
  ssh $SSH_OPTS root@"$ip" bash -lc "'
    ${remote_env}
    cd /root/diffuse
    pkill -f diffuse || true
    nohup ./target/release/diffuse host \
      --model ${MODEL} \
      --spawn-worker \
      --public-addr ${ip}:9440 \
      --overhead ${OVERHEAD} \
      --bootstrap http://${sentinel_ip}:9440 \
      > /root/node.log 2>&1 &
  '"
  sleep 5
done

echo
echo "nodes launching. weights download takes a few minutes."
echo "watch coverage with: ./04-status.sh"
