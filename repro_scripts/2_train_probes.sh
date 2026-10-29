#!/bin/bash
set -euo pipefail

source .venv/bin/activate
source repro_scripts/paths.sh

# Step 2: every probe the paper reports. Run once the collect jobs of step 1 have finished.

TRAIN="--input $ACTIVATION_FILE --all-layers --split-file $SPLIT_FILE --eval-splits validation test"

# LR, SVM and difference-in-means probes per concept (Fig 2)
bash slurm/submit.sh train --run $SEX_LR  $TRAIN --preset sex --probe logistic_regression --c 1.0
bash slurm/submit.sh train --run $SEX_SVM $TRAIN --preset sex --probe dualxda_svm --c 1.0
bash slurm/submit.sh train --run $SEX_DIM $TRAIN --preset sex --probe difference_in_means
bash slurm/submit.sh train --run $AGE_LR  $TRAIN --preset age --probe logistic_regression --c 1.0
bash slurm/submit.sh train --run $AGE_SVM $TRAIN --preset age --probe dualxda_svm --c 1.0
bash slurm/submit.sh train --run $AGE_DIM $TRAIN --preset age --probe difference_in_means

# One image per identity, 32 draws (Fig 3, and the single-image age result in the text)
bash slurm/submit.sh train --run $SEX_SINGLE $TRAIN --preset sex --probe logistic_regression --c 1.0 --subsample identity --subsample-draws 32
bash slurm/submit.sh train --run $AGE_SINGLE $TRAIN --preset age --probe logistic_regression --c 1.0 --subsample identity --subsample-draws 32

# The same LR probes on the stock backbone (Figs 4, 14 and 15)
PRETRAINED="--input $PRETRAINED_ACTIVATION_FILE --all-layers --split-file $SPLIT_FILE --eval-splits validation test"
bash slurm/submit.sh train --run $SEX_LR_PRETRAINED $PRETRAINED --preset sex --probe logistic_regression --c 1.0
bash slurm/submit.sh train --run $AGE_LR_PRETRAINED $PRETRAINED --preset age --probe logistic_regression --c 1.0

# Regularization sweep behind the cumulative-mass curves (Fig 12)
bash slurm/submit.sh train --run $SEX_LR_C10    $TRAIN --preset sex --probe logistic_regression --c 10.0
bash slurm/submit.sh train --run $SEX_LR_C01    $TRAIN --preset sex --probe logistic_regression --c 0.1
bash slurm/submit.sh train --run $SEX_LR_C0001  $TRAIN --preset sex --probe logistic_regression --c 0.001
bash slurm/submit.sh train --run $SEX_SVM_C10   $TRAIN --preset sex --probe dualxda_svm --c 10.0
bash slurm/submit.sh train --run $SEX_SVM_C01   $TRAIN --preset sex --probe dualxda_svm --c 0.1
bash slurm/submit.sh train --run $SEX_SVM_C0001 $TRAIN --preset sex --probe dualxda_svm --c 0.001
