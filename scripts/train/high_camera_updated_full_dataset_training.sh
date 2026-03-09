#!/bin/bash
# DreamZero multi-GPU training script for the high_camera_updated embodiment.
#
# Usage:
#   bash scripts/train/high_camera_updated_full_dataset_training.sh
#
# Notes:
#   - DATASET_SHARD_SAMPLING_RATE controls how much of each cached shard is sampled.
#       1.0 = sample the full shard (best for full-dataset coverage)
#       0.1 = sample 10% of each shard (faster, but needs more training steps for similar coverage)
#   - The sharded loader is stochastic, so MAX_STEPS is a rough raw-data-equivalent pass count,
#     not a strict no-replacement epoch.

export HYDRA_FULL_ERROR=1

HIGH_CAMERA_UPDATED_DATA_ROOT=${HIGH_CAMERA_UPDATED_DATA_ROOT:-"./data/high-camera-updated"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_high_camera_updated_full_dataset"}

if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
DATASET_SHARD_SAMPLING_RATE=${DATASET_SHARD_SAMPLING_RATE:-1.0}
NUM_STEPS_PER_SHARD=${NUM_STEPS_PER_SHARD:-1000}
SAVE_STEPS=${SAVE_STEPS:-1000}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}

if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -d "$HIGH_CAMERA_UPDATED_DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $HIGH_CAMERA_UPDATED_DATA_ROOT"
    exit 1
fi

if [ ! -f "$HIGH_CAMERA_UPDATED_DATA_ROOT/meta/modality.json" ] || [ ! -f "$HIGH_CAMERA_UPDATED_DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: DreamZero metadata missing in $HIGH_CAMERA_UPDATED_DATA_ROOT/meta"
    echo "Run scripts/data/convert_lerobot_to_gear.py for this dataset before training."
    exit 1
fi

TOTAL_SAMPLES=$(DATASET_ROOT="$HIGH_CAMERA_UPDATED_DATA_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DATASET_ROOT"])
info_path = root / "meta" / "info.json"
if info_path.exists():
    with info_path.open() as f:
        info = json.load(f)
    print(int(info["total_frames"]))
else:
    episodes_path = root / "meta" / "episodes.jsonl"
    total = 0
    with episodes_path.open() as f:
        for line in f:
            total += int(json.loads(line)["length"])
    print(total)
PY
)

GLOBAL_BATCH_SIZE=$(( NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE ))
RAW_PASS_STEPS=$(( (TOTAL_SAMPLES + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE ))
MAX_STEPS=${MAX_STEPS:-$RAW_PASS_STEPS}

echo "Dataset samples: $TOTAL_SAMPLES"
echo "NUM_GPUS: $NUM_GPUS"
echo "Per-device batch size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "dataset_shard_sampling_rate: $DATASET_SHARD_SAMPLING_RATE"
echo "num_steps_per_shard: $NUM_STEPS_PER_SHARD"
echo "Approximate steps for one raw-data-equivalent pass: $RAW_PASS_STEPS"
echo "Training max_steps: $MAX_STEPS"

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/high_camera_updated_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    dataset_shard_sampling_rate=$DATASET_SHARD_SAMPLING_RATE \
    train_dataset.dataset_kwargs.num_steps_per_shard=$NUM_STEPS_PER_SHARD \
    high_camera_updated_data_root=$HIGH_CAMERA_UPDATED_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=./checkpoints/DreamZero-AgiBot \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
