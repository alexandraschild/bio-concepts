#!/bin/bash
set -euo pipefail

source .venv/bin/activate
source repro_scripts/paths.sh

# Step 1: the activation caches every later step reads. Nothing has to have run before this.

# Gorillas, through the Re-ID fine-tuned and the stock DINOv3-Base (both concepts read these)
bash slurm/submit.sh collect --image-dir $GORILLA_IMAGES --mapping $MAPPING --model-path $MODEL_PATH --timm-model vit_base_patch16_dinov3 --all-layers --output $ACTIVATION_FILE --activation-dtype bfloat16
bash slurm/submit.sh collect --image-dir $GORILLA_IMAGES --mapping $MAPPING --model-path $PRETRAINED_MODEL_PATH --timm-model vit_base_patch16_dinov3 --all-layers --output $PRETRAINED_ACTIVATION_FILE --activation-dtype bfloat16

# Jaguars and polar bears, through the stock DINOv3-Base (no Re-ID fine-tune exists for
# them), on CPU to match the paper's numbers exactly
bash slurm/submit.sh collect-jaguar --data-dir $JAGUAR_DATA --model-path $PRETRAINED_MODEL_PATH --timm-model vit_base_patch16_dinov3 --all-layers --num-workers 4 --device cpu --output $JAGUAR_ACTIVATION_FILE
bash slurm/submit.sh collect-polarbear --data-dir $POLARBEAR_DATA --model-path $PRETRAINED_MODEL_PATH --timm-model vit_base_patch16_dinov3 --all-layers --num-workers 4 --device cpu --output $POLARBEAR_ACTIVATION_FILE
