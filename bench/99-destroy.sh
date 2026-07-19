#!/usr/bin/env bash
set -euo pipefail

# Hetzner bills a server until it is DELETED. Powering it off changes nothing.
# Run this the moment the measurements are done.

echo "the following servers will be permanently deleted:"
hcloud server list -l bench=diffuse -o columns=name,ipv4,status
echo

read -r -p "type DELETE to confirm: " answer
if [ "$answer" != "DELETE" ]; then
  echo "aborted, nothing was deleted"
  exit 1
fi

mapfile -t names < <(hcloud server list -l bench=diffuse -o columns=name | tail -n +2 | tr -d ' ')
for n in "${names[@]}"; do
  echo "deleting $n"
  hcloud server delete "$n"
done

echo
echo "remaining servers on the account:"
hcloud server list
