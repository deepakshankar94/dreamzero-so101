# Train On SO101 README

This document explains how to download the SO101 dataset, prepare DreamZero metadata, train the `high_camera_updated` embodiment, and serve the resulting checkpoint.

## Assumptions

- Repo root: `/root/dreamzero`
- Dataset ID: `deepakshankar94/high-camera-updated`
- Default training run name: `dreamzero_high_camera_updated_run1_1k_steps`

## Step 1: Activate The Environment

```bash
cd /root/dreamzero
source .venv/bin/activate
```

## Step 2: Download The Dataset

Download the dataset into the repo-local data folder:

```bash
hf download deepakshankar94/high-camera-updated \
  --repo-type dataset \
  --local-dir /root/dreamzero/data/high-camera-updated
```

If you already have the dataset locally, skip this step.

## Step 3: Convert The Dataset To DreamZero Metadata

Generate the GEAR/DreamZero metadata in-place:

```bash
python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path /root/dreamzero/data/high-camera-updated \
  --embodiment-tag high_camera_updated \
  --force
```

Verify the metadata exists:

```bash
ls /root/dreamzero/data/high-camera-updated/meta
```

You should see files such as:

- `modality.json`
- `embodiment.json`
- `episodes.jsonl`
- `tasks.jsonl`
- `relative_stats_dreamzero.json`

## Step 4: Download Base Checkpoints

Download the DreamZero-AgiBot post-training checkpoint:

```bash
hf download GEAR-Dreams/DreamZero-AgiBot \
  --repo-type model \
  --local-dir /root/dreamzero/checkpoints/DreamZero-AgiBot
```

The training script will auto-download the Wan and tokenizer weights if they are missing, but you can also download them ahead of time:

```bash
hf download Wan-AI/Wan2.1-I2V-14B-480P \
  --repo-type model \
  --local-dir /root/dreamzero/checkpoints/Wan2.1-I2V-14B-480P

hf download google/umt5-xxl \
  --repo-type model \
  --local-dir /root/dreamzero/checkpoints/umt5-xxl
```

## Step 5: Train

### Multi-GPU Default

The default training script now writes to:

```bash
/root/dreamzero/checkpoints/dreamzero_high_camera_updated_run1_1k_steps
```

and runs for `1000` steps by default.

Run:

```bash
bash /root/dreamzero/scripts/train/high_camera_updated_training.sh
```

Optional overrides:

```bash
NUM_GPUS=1 \
HIGH_CAMERA_UPDATED_DATA_ROOT=/root/dreamzero/data/high-camera-updated \
OUTPUT_DIR=/root/dreamzero/checkpoints/dreamzero_high_camera_updated_run1_1k_steps \
bash /root/dreamzero/scripts/train/high_camera_updated_training.sh
```

### Single-GPU Variant

If you want the explicit 1-GPU wrapper:

```bash
bash /root/dreamzero/scripts/train/high_camera_updated_run1.sh
```

This also writes to:

```bash
/root/dreamzero/checkpoints/dreamzero_high_camera_updated_run1_1k_steps
```

and now runs for `1000` steps.

## Step 6: Serve The Trained Model

After training finishes, start the websocket server:

```bash
MODEL_PATH=/root/dreamzero/checkpoints/dreamzero_high_camera_updated_run1_1k_steps \
ATTENTION_BACKEND=FA2 \
ENABLE_DIT_CACHE=true \
bash /root/dreamzero/scripts/serve/high_camera_updated_server.sh
```

## Step 7: Smoke Test The Server

From another terminal:

```bash
python /root/dreamzero/scripts/serve/high_camera_updated_client_example.py \
  --host 127.0.0.1 \
  --port 8000
```

Expected result:

- `action.joint_position`
- `action.gripper_position`

## Step 8: Connect The Local SO101 Robot

Use the robot bridge from the robot computer:

```bash
SERVER_HOST=<gpu-host-or-127.0.0.1> \
SERVER_PORT=8000 \
ROBOT_PORT=/dev/ttyACM0 \
ROBOT_ID=so101 \
OVERHEAD_CAMERA=/dev/video0 \
ARM_CAMERA=/dev/video6 \
TASK="pick white cuboid and place on blue notepad" \
ACTIONS_PER_INFER=1 \
bash /root/dreamzero/scripts/serve/so101_dreamzero_bridge.sh
```

Keep `ACTIONS_PER_INFER=1` for the first live tests so the robot replans after every executed step.
