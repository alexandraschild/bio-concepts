#!/bin/bash
set -euo pipefail

source .venv/bin/activate
source repro_scripts/paths.sh

# CLIP ablation, outside the five steps: the sex and age LR probes of step 2 on OpenAI's stock
# CLIP ViT-B/16. The figures also read the DINOv3 runs of steps 2-3. Runs through without Slurm.

# Token mean of every block (~35 min on 8 CPUs); the weights come from the timm hub
python -m src.data.collect_clip --image-dir $GORILLA_IMAGES --mapping $MAPPING --output $CLIP_ACTIVATION_FILE --num-workers 2

CLIP_TRAIN="--input $CLIP_ACTIVATION_FILE --all-layers --split-file $SPLIT_FILE --eval-splits validation test"
python -m src.probes.train --run $SEX_LR_CLIP $CLIP_TRAIN --preset sex --probe logistic_regression --c 1.0
python -m src.probes.train --run $AGE_LR_CLIP $CLIP_TRAIN --preset age --probe logistic_regression --c 1.0

python -m src.probes.evaluate holdout --run $SEX_LR_CLIP
python -m src.probes.evaluate holdout --run $AGE_LR_CLIP

# One refit per held-out gorilla at layer 8, as for Figs 6 and 9
python -m src.probes.evaluate id-cv --run $SEX_LR_CLIP --activations $CLIP_ACTIVATION_FILE --preset sex --layers 8 --split-file $SPLIT_FILE --n-jobs 8
python -m src.probes.evaluate id-cv --run $AGE_LR_CLIP --activations $CLIP_ACTIVATION_FILE --preset age --layers 8 --split-file $SPLIT_FILE --n-jobs 8

# Layer-wise AUROC against both DINOv3 backbones, and the per-layer numbers
mkdir -p $FIGURES $TABLES
python -m src.report.probes layers --evals $SEX_LR_CLIP/eval/holdout $SEX_LR_PRETRAINED/eval/holdout $SEX_LR/eval/holdout --label "CLIP ViT-B/16" "DINOv3 (pre-trained)" "DINOv3 (fine-tuned)" --out $FIGURES --name clip_layerwise_auroc_sex
python -m src.report.probes layers --evals $AGE_LR_CLIP/eval/holdout $AGE_LR_PRETRAINED/eval/holdout $AGE_LR/eval/holdout --label "CLIP ViT-B/16" "DINOv3 (pre-trained)" "DINOv3 (fine-tuned)" --out $FIGURES --name clip_layerwise_auroc_age
python -m src.report.probes table --evals $SEX_LR_CLIP/eval/holdout $AGE_LR_CLIP/eval/holdout --layer all --metrics auroc --unit image identity --with-train --out $TABLES --name auroc_clip
