#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROTO_DIR="$ROOT/proto"
OUT_DIR="$ROOT/worker/diffuse_worker"

python -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR/data.proto"

sed -i 's/^import data_pb2 as data__pb2/from diffuse_worker import data_pb2 as data__pb2/' \
  "$OUT_DIR/data_pb2_grpc.py"

echo "proto generated into $OUT_DIR"