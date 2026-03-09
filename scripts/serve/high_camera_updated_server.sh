#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/dreamzero/checkpoints/dreamzero_high_camera_updated_run1_1k_steps}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FA2}"
ENABLE_DIT_CACHE="${ENABLE_DIT_CACHE:-false}"
DYNAMO_RECOMPILE_LIMIT="${DYNAMO_RECOMPILE_LIMIT:-800}"
DYNAMO_CACHE_SIZE_LIMIT="${DYNAMO_CACHE_SIZE_LIMIT:-1000}"

cd /root/dreamzero

export ATTENTION_BACKEND
export ENABLE_DIT_CACHE

DIT_CACHE_ARGS=()
if [[ "${ENABLE_DIT_CACHE,,}" == "true" ]]; then
  DIT_CACHE_ARGS+=(--enable-dit-cache)
fi

torchrun --standalone --nproc_per_node 1 scripts/serve/high_camera_updated_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --model-path "$MODEL_PATH" \
  --attention-backend "$ATTENTION_BACKEND" \
  --dynamo-recompile-limit "$DYNAMO_RECOMPILE_LIMIT" \
  --dynamo-cache-size-limit "$DYNAMO_CACHE_SIZE_LIMIT" \
  "${DIT_CACHE_ARGS[@]}"
