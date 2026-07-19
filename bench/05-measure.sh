#!/usr/bin/env bash
set -euo pipefail

# Measurement pass. Runs from the sentinel so the client sits inside the
# same datacenter as the pipeline, which is the LAN scenario.
# Results land in results/ on this machine.

KEY="${KEY:-$HOME/.ssh/diffuse_oracle}"
MODEL="${MODEL:-mistralai/Mistral-Small-24B-Instruct-2501}"
RUNS="${RUNS:-5}"
CONCURRENT="${CONCURRENT:-4}"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

sentinel_ip=$(hcloud server list -l role=sentinel -o columns=ipv4 | tail -n +2 | tr -d ' ')
mkdir -p results
stamp=$(date +%Y%m%d-%H%M%S)
out="results/${stamp}"
mkdir -p "$out"

prompts=(
  "Explain what a transformer model is in two sentences."
  "What is the capital of France?"
  "Write one sentence about distributed systems."
  "Summarise the concept of latency."
  "Name three European countries."
)

echo "single stream latency, ${RUNS} runs"
for i in $(seq 1 "$RUNS"); do
  p="${prompts[$(( (i - 1) % ${#prompts[@]} ))]}"
  echo "  run $i"
  ssh $SSH_OPTS root@"$sentinel_ip" bash -lc "'
    export PATH=\$HOME/.cargo/bin:\$PATH
    cd /root/diffuse
    RUST_LOG=info ./target/release/diffuse query \
      --prompt \"${p}\" \
      --model ${MODEL} \
      --bootstrap http://${sentinel_ip}:9440 2>&1
  '" > "${out}/single-${i}.log" 2>&1 || true
  grep -oE "generation:.*" "${out}/single-${i}.log" || echo "    no generation line"
done

echo
echo "aggregate throughput, ${CONCURRENT} concurrent requests"
ssh $SSH_OPTS root@"$sentinel_ip" bash -lc "'
  export PATH=\$HOME/.cargo/bin:\$PATH
  cd /root/diffuse
  for c in \$(seq 1 ${CONCURRENT}); do
    RUST_LOG=info ./target/release/diffuse query \
      --prompt \"Explain distributed inference in one sentence.\" \
      --model ${MODEL} \
      --bootstrap http://${sentinel_ip}:9440 > /root/conc-\${c}.log 2>&1 &
  done
  wait
  cat /root/conc-*.log
'" > "${out}/concurrent.log" 2>&1 || true

grep -oE "generation:.*" "${out}/concurrent.log" || echo "  no generation lines"

echo
echo "topology at measurement time"
./04-status.sh > "${out}/topology.txt" 2>&1 || true

echo
echo "results in ${out}"
