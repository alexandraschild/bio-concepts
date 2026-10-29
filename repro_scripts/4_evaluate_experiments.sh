#!/bin/bash
set -euo pipefail

source .venv/bin/activate
source repro_scripts/paths.sh

# Step 4: the metrics the figures are drawn from. Run once the jobs of step 3 have finished.

# =================
# Attribution metrics
# =================

# Age-sex and identity composition of the k90 set (LR) and the support vectors (SVM)
bash slurm/submit.sh eval-attribution composition --run $SEX_LR --method representer --weights count --top-frac 90
bash slurm/submit.sh eval-attribution composition --run $SEX_SVM --method representer --weights count

# Cumulative attribution mass over the top x% of training images
for RUN in $SEX_LR_C10 $SEX_LR_C01 $SEX_LR_C0001 $SEX_SVM_C10 $SEX_SVM_C01 $SEX_SVM_C0001; do
    bash slurm/submit.sh eval-attribution curve --run $RUN --method representer
done

# Held-out AUROC change from re-training without each identity
bash slurm/submit.sh eval-attribution loo --run $SEX_LR --method exact-loo

# =================
# Steering metrics
# =================

# Five random directions per concept, with the same subset, doses and signs as the
# layer-9 sweep. Steering block 9 only runs blocks 10-11, so a CPU suffices.
for RUN in $SEX_LR $AGE_LR; do
    python -m src.steering.steer_controls --reference $RUN/steering/embeddings/steered_layer9.pt
done

# Flip rate and off-manifold share per layer, K=5
for RUN in $SEX_LR $AGE_LR; do
    for LAYER in 0 1 2 3 4 5 6 7 8 9 10 11; do
        python -m src.steering.evaluate_reid --run $RUN --layer $LAYER
    done
    for CONTROL in $RUN/steering/controls/random_s*; do
        python -m src.steering.evaluate_reid --run $CONTROL --layer 9
    done
done
