# Running On Local Robot README

This document explains how to run the trained `high_camera_updated` DreamZero model on a GPU server and connect it to a local SO101 robot computer.

## Files Used

- Server launcher: `scripts/serve/high_camera_updated_server.sh`
- Server implementation: `scripts/serve/high_camera_updated_server.py`
- SO101 bridge launcher: `scripts/serve/so101_dreamzero_bridge.sh`
- SO101 bridge implementation: `scripts/serve/so101_dreamzero_bridge.py`
- Simple websocket client test: `scripts/serve/high_camera_updated_client_example.py`

## Assumptions

- The trained checkpoint is at:

  ```bash
  /root/dreamzero/checkpoints/dreamzero_high_camera_updated_run1
  ```

- The GPU server has the DreamZero repo and environment available.
- The robot computer also has the repo available and has `lerobot` installed.
- The SO101 setup uses:
  - one overhead RGB camera
  - one arm RGB camera
  - 5 arm joints plus 1 gripper

## Step 1: Start The Model Server On The GPU Box

On the GPU server:

```bash
cd /root/dreamzero
source .venv/bin/activate

MODEL_PATH=/root/dreamzero/checkpoints/dreamzero_high_camera_updated_run1 \
ATTENTION_BACKEND=FA2 \
ENABLE_DIT_CACHE=true \
bash /root/dreamzero/scripts/serve/high_camera_updated_server.sh
```

Leave this terminal running.

By default the server listens on:

```bash
0.0.0.0:8000
```

## Step 2: Check Connectivity From The Robot Computer

On the robot computer:

```bash
cd /root/dreamzero
source .venv/bin/activate
```

If the robot computer cannot directly reach the GPU server, open an SSH tunnel in a separate terminal:

```bash
ssh -L 8000:localhost:8000 <user>@<gpu-host>
```

If you use a tunnel, your local client host is:

```bash
127.0.0.1
```

If you do not use a tunnel, use the actual GPU host/IP directly.

## Step 3: Smoke Test The Websocket Server Before Touching The Robot

From the robot computer:

```bash
python /root/dreamzero/scripts/serve/high_camera_updated_client_example.py \
  --host 127.0.0.1 \
  --port 8000
```

If you are not tunneling, replace `127.0.0.1` with the real GPU host/IP.

You should see returned action keys like:

- `action.joint_position`
- `action.gripper_position`

## Step 4: Find Your Robot And Camera Devices

On the robot computer:

```bash
ls /dev/ttyACM*
ls /dev/video*
```

You need:

- one serial port for the SO101 follower arm
- one device path for the overhead camera
- one device path for the arm camera

## Step 5: Run The SO101 Bridge

On the robot computer:

```bash
SERVER_HOST=127.0.0.1 \
SERVER_PORT=8000 \
ROBOT_PORT=/dev/ttyACM0 \
ROBOT_ID=so101 \
OVERHEAD_CAMERA=/dev/video2 \
ARM_CAMERA=/dev/video0 \
TASK="pick white cuboid and place on blue notepad" \
ACTIONS_PER_INFER=1 \
bash /root/dreamzero/scripts/serve/so101_dreamzero_bridge.sh
```

If you are not tunneling, replace:

```bash
SERVER_HOST=127.0.0.1
```

with:

```bash
SERVER_HOST=<gpu-host>
```

## Important Runtime Notes

- Start with:

  ```bash
  ACTIONS_PER_INFER=1
  ```

  This executes only the first predicted action from each returned chunk.

- The server returns action chunks of horizon 24. The bridge intentionally only executes the first action by default because that is safer for live testing.

- The bridge assumes the local camera names are:
  - `overhead`
  - `grey_arm`

  If your LeRobot camera keys differ, set:

  ```bash
  OVERHEAD_CAMERA_KEY=<your_overhead_key>
  ARM_CAMERA_KEY=<your_arm_key>
  ```

- If your robot computer does not have enough VRAM to run DreamZero locally, keep the model on the GPU server and only run the SO101 bridge locally.

## Common Problems

### 1. `import lerobot` fails on the robot computer

The bridge script depends on LeRobot on the robot machine. Verify:

```bash
python -c "import lerobot; print('ok')"
```

### 2. The websocket smoke test fails

Check:

- the server process is still running on the GPU box
- port `8000` is reachable
- the SSH tunnel is open if you are using one

### 3. The bridge says it cannot infer motor key order

The script tries:

- `robot._motors_ft.keys()`
- default SO101 motor names
- any observed `.pos` keys

If your local setup exposes different state keys, adjust `scripts/serve/so101_dreamzero_bridge.py`.

### 4. Camera images are wrong or swapped

Swap:

```bash
OVERHEAD_CAMERA=/dev/videoX
ARM_CAMERA=/dev/videoY
```

or adjust:

```bash
OVERHEAD_CAMERA_KEY=...
ARM_CAMERA_KEY=...
```

### 5. The robot moves too aggressively

Keep:

```bash
ACTIONS_PER_INFER=1
```

for first tests and stop immediately if behavior is not correct.

## Minimal Verification Checklist

1. Server starts on the GPU box.
2. `high_camera_updated_client_example.py` returns action keys from the robot computer.
3. SO101 bridge connects to the robot and cameras.
4. The robot executes one action at a time without crashing.
