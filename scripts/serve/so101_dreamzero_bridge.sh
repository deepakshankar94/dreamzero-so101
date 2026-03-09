#!/usr/bin/env bash
set -euo pipefail

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-so101}"
OVERHEAD_CAMERA="${OVERHEAD_CAMERA:-/dev/video2}"
ARM_CAMERA="${ARM_CAMERA:-/dev/video0}"
OVERHEAD_CAMERA_KEY="${OVERHEAD_CAMERA_KEY:-overhead}"
ARM_CAMERA_KEY="${ARM_CAMERA_KEY:-grey_arm}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
TASK="${TASK:-pick white cuboid and place on blue notepad}"
ACTIONS_PER_INFER="${ACTIONS_PER_INFER:-1}"
ACTION_SLEEP_S="${ACTION_SLEEP_S:-0.10}"


python scripts/serve/so101_dreamzero_bridge.py \
  --server-host "$SERVER_HOST" \
  --server-port "$SERVER_PORT" \
  --robot-port "$ROBOT_PORT" \
  --robot-id "$ROBOT_ID" \
  --overhead-camera "$OVERHEAD_CAMERA" \
  --arm-camera "$ARM_CAMERA" \
  --overhead-camera-key "$OVERHEAD_CAMERA_KEY" \
  --arm-camera-key "$ARM_CAMERA_KEY" \
  --camera-width "$CAMERA_WIDTH" \
  --camera-height "$CAMERA_HEIGHT" \
  --camera-fps "$CAMERA_FPS" \
  --task "$TASK" \
  --actions-per-infer "$ACTIONS_PER_INFER" \
  --action-sleep-s "$ACTION_SLEEP_S"
