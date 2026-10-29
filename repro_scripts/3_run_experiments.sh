#!/bin/bash
set -euo pipefail

source .venv/bin/activate
source repro_scripts/paths.sh

# Step 3: the experiments, all reading the probes. Run once the train jobs of step 2 have finished.

# =================
# Probe evaluation
# =================

# Held-out metrics of every probe the figures read
for RUN in $SEX_LR $SEX_SVM $SEX_DIM $SEX_LR_PRETRAINED $AGE_LR $AGE_SVM $AGE_DIM $AGE_LR_PRETRAINED; do
    bash slurm/submit.sh eval-probes holdout --run $RUN
done

# The 32 single-image draws, one cache each
bash slurm/submit.sh eval-probes holdout --probes $SEX_SINGLE/probes/draw* --out $SEX_SINGLE/eval/holdout
bash slurm/submit.sh eval-probes holdout --probes $AGE_SINGLE/probes/draw* --out $AGE_SINGLE/eval/holdout

# Transfer: each LR probe scored on the other backbone's activations
bash slurm/submit.sh eval-probes holdout --run $SEX_LR --activations $PRETRAINED_ACTIVATION_FILE
bash slurm/submit.sh eval-probes holdout --run $SEX_LR_PRETRAINED --activations $ACTIVATION_FILE
bash slurm/submit.sh eval-probes holdout --run $AGE_LR --activations $PRETRAINED_ACTIVATION_FILE
bash slurm/submit.sh eval-probes holdout --run $AGE_LR_PRETRAINED --activations $ACTIVATION_FILE

# One refit per held-out gorilla at layer 8 (Figs 6 and 9)
bash slurm/submit.sh eval-probes id-cv --run $SEX_LR --activations $ACTIVATION_FILE --preset sex --layers 8 --split-file $SPLIT_FILE
bash slurm/submit.sh eval-probes id-cv --run $AGE_LR --activations $ACTIVATION_FILE --preset age --layers 8 --split-file $SPLIT_FILE

# Jaguars (one identity held out) and polar bears (one zoo held out; one bear held out
# for the confounded comparison in Appendix B.2), every layer (Fig 11)
SPECIES="--preset sex --all-layers --standardize --max-iter 5000 --n-jobs 8"
bash slurm/submit.sh eval-probes id-cv --activations $JAGUAR_ACTIVATION_FILE $SPECIES --out $JAGUAR_DIR/id_cv
bash slurm/submit.sh eval-probes id-cv --activations $POLARBEAR_ACTIVATION_FILE $SPECIES --fold-level zoo --out $POLARBEAR_DIR/id_cv_zoo
bash slurm/submit.sh eval-probes id-cv --activations $POLARBEAR_ACTIVATION_FILE $SPECIES --out $POLARBEAR_DIR/id_cv_identity

# =================
# Activation steering
# =================

# Steered 256-d embeddings of the LR directions, one cache per layer (Figs 5 and 8)
bash slurm/submit.sh steer --run $SEX_LR --alpha 1 2 4 8 16 32 64
bash slurm/submit.sh steer --run $AGE_LR --alpha 1 2 4 8 16 32 64

# =================
# Data attribution
# =================

# Representer decompositions: the k90 set (LR) and the support vectors (SVM) for sex
# (Figs 7 and 18), and the age SVM behind the heatmap of Fig 10
bash slurm/submit.sh attribute --run $SEX_LR --method representer
bash slurm/submit.sh attribute --run $SEX_SVM --method representer
bash slurm/submit.sh attribute --run $AGE_SVM --method representer --layers blocks.9

# The regularization sweep, at the one layer the cumulative-mass curves are drawn from (Fig 12)
for RUN in $SEX_LR_C10 $SEX_LR_C01 $SEX_LR_C0001 $SEX_SVM_C10 $SEX_SVM_C01 $SEX_SVM_C0001; do
    bash slurm/submit.sh attribute --run $RUN --method representer --layers blocks.7
done

# Exact leave-one-identity-out refits of the sex LR probe at layer 9
bash slurm/submit.sh attribute --run $SEX_LR --method exact-loo --layers blocks.9
