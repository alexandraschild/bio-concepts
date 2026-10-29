#!/bin/bash
set -euo pipefail

source .venv/bin/activate
source repro_scripts/paths.sh

# Step 5: every figure of the paper drawn by this code, plus the numbers the text quotes.
# Run once the jobs of step 4 have finished. Figs 13, 16 and 17 were drawn outside this
# repository; their numbers are in the AUROC tables, the steering eval JSONs and the
# composition CSVs written here.

mkdir -p $FIGURES $TABLES

# Fig 1: DualXDA heatmaps for an adult-female test image (sex SVM)
python -m src.attribution.xda --run $SEX_SVM --layer blocks.9 --test-image PL02_R066_20220911_147_378_830031 --out $FIGURES/fig_1_xda_sex

# Fig 2: layer-wise AUROC of the three probe types, train curves dashed
python -m src.report.probes layers --evals $SEX_LR/eval/holdout $SEX_SVM/eval/holdout $SEX_DIM/eval/holdout --label "LR (C=1)" "SVM (C=1)" "DiM" --with-train --out $FIGURES --name fig_2a_layerwise_auroc_sex
python -m src.report.probes layers --evals $AGE_LR/eval/holdout $AGE_SVM/eval/holdout $AGE_DIM/eval/holdout --label "LR (C=1)" "SVM (C=1)" "DiM" --with-train --out $FIGURES --name fig_2b_layerwise_auroc_age

# Fig 3: the full probe against the single-image-per-identity band
python -m src.report.probes layers --evals $SEX_LR/eval/holdout --label "LR (C=1)" --band $SEX_SINGLE/eval/holdout --band-label "Single image LR (C=1, 32 draws)" --out $FIGURES --name fig_3_single_image_sex

# Fig 4: cross-model transfer, each direction paired with its in-domain reference
python -m src.report.probes layers --evals $SEX_LR/eval/transfer_dinov3_base_pretrained $SEX_LR_PRETRAINED/eval/transfer_dinov3_base_finetuned $SEX_LR/eval/holdout $SEX_LR_PRETRAINED/eval/holdout --label "ft->pre" "pre->ft" "in-domain ft" "in-domain pre" --hue 0 1 0 1 --dash solid solid dashed dashed --out $FIGURES --name fig_4_transfer_sex

# Fig 5: sex steering -- flips against off-manifold over depth at alpha=16, and over
# alpha at layer 9 against the random control
python -m src.report.steering layers --evals $SEX_LR --alpha 16 --out $FIGURES --name fig_5a_steering_by_layer_sex
python -m src.report.steering dose --evals $SEX_LR --control $SEX_LR/steering/controls/random_s* --layer 9 --label "sex direction" --out $FIGURES --name fig_5b_steering_by_alpha_sex

# Fig 6: mean ID-CV logit per identity, by age-sex class
python -m src.report.probes logits --evals $SEX_LR/eval/id_cv --layer blocks.8 --classes Silverback Blackback "Adult Female" --figsize 5.4 15 --out $FIGURES --name fig_6_id_cv_logits_sex

# Fig 7: silverback and blackback enrichment in the k90 set and the support vectors
python -m src.report.attribution composition --evals $SEX_LR/attributions/representer $SEX_SVM/attributions/representer --label "LR (C=1)" "SVM (C=1)" --age-sex Silverback Blackback --out $FIGURES --name fig_7_enrichment_sex

# Fig 8: Fig 5 for age
python -m src.report.steering layers --evals $AGE_LR --alpha 16 --out $FIGURES --name fig_8a_steering_by_layer_age
python -m src.report.steering dose --evals $AGE_LR --control $AGE_LR/steering/controls/random_s* --layer 9 --label "age direction" --out $FIGURES --name fig_8b_steering_by_alpha_age

# Fig 9: Fig 6 for age
python -m src.report.probes logits --evals $AGE_LR/eval/id_cv --layer blocks.8 --classes "${AGE_CLASSES[@]}" --figsize 5.4 15 --out $FIGURES --name fig_9_id_cv_logits_age

# Fig 10: DualXDA heatmaps for an infant test image (age SVM)
python -m src.attribution.xda --run $AGE_SVM --layer blocks.9 --test-image NN62_R465_20220905_377_948_134405 --out $FIGURES/fig_10_xda_age

# Fig 11: sex probes on jaguars (one identity held out) and polar bears (one zoo held out)
python -m src.report.probes layers --evals $JAGUAR_DIR/id_cv $JAGUAR_DIR/id_cv --unit identity image --label "Per identity" "Per image" --out $FIGURES --name fig_11a_jaguar
python -m src.report.probes layers --evals $POLARBEAR_DIR/id_cv_zoo $POLARBEAR_DIR/id_cv_zoo --unit identity image --label "Per identity" "Per image" --out $FIGURES --name fig_11b_polarbear

# Fig 12: cumulative attribution mass at layer 7 over the regularization sweep
python -m src.report.attribution curve --evals $SEX_LR_C10/attributions/representer $SEX_LR_C01/attributions/representer $SEX_LR_C0001/attributions/representer $SEX_SVM_C10/attributions/representer $SEX_SVM_C01/attributions/representer $SEX_SVM_C0001/attributions/representer --label "LR (C=10)" "LR (C=0.1)" "LR (C=0.001)" "SVM (C=10.0)" "SVM (C=0.1)" "SVM (C=0.001)" --layer blocks.7 --xmax 0.10 --out $FIGURES --name fig_12_cumulative_mass

# Figs 14 and 15: fine-tuned against pre-trained backbone, in AUROC and in direction
python -m src.report.probes layers --evals $SEX_LR/eval/holdout $SEX_LR_PRETRAINED/eval/holdout --label "LR (fine-tuned, C=1)" "LR (pre-trained, C=1)" --out $FIGURES --name fig_14_finetuned_vs_pretrained
python -m src.report.probes cosine --probes $SEX_LR $SEX_LR_PRETRAINED --no-legend --out $FIGURES --name fig_15_cosine_finetuned_pretrained

# Fig 18: share of each identity's images among the support vectors, layer 9
python -m src.report.attribution top --evals $SEX_SVM/attributions/representer --layer blocks.9 --value selection_rate --out $FIGURES --name fig_18_id_selection_rate

# =================
# Numbers the text quotes without a figure of its own
# =================

# Image- and identity-level AUROC per layer, train and held-out (incl. the 0.97 identity AUROC)
python -m src.report.probes table --evals $SEX_LR/eval/holdout $SEX_SVM/eval/holdout $SEX_DIM/eval/holdout --layer all --metrics auroc --unit image identity --with-train --out $TABLES --name auroc_sex
python -m src.report.probes table --evals $AGE_LR/eval/holdout $AGE_SVM/eval/holdout $AGE_DIM/eval/holdout --layer all --metrics auroc --unit image identity --with-train --out $TABLES --name auroc_age

# Age: single-image probes, and transfer between the backbones (Sec. 5.1, 5.2)
python -m src.report.probes layers --evals $AGE_LR/eval/holdout --label "LR (C=1)" --band $AGE_SINGLE/eval/holdout --band-label "Single image LR (C=1, 32 draws)" --out $FIGURES --name text_single_image_age
python -m src.report.probes layers --evals $AGE_LR/eval/transfer_dinov3_base_pretrained $AGE_LR_PRETRAINED/eval/transfer_dinov3_base_finetuned $AGE_LR/eval/holdout $AGE_LR_PRETRAINED/eval/holdout --label "ft->pre" "pre->ft" "in-domain ft" "in-domain pre" --hue 0 1 0 1 --dash solid solid dashed dashed --out $FIGURES --name text_transfer_age

# LR and SVM directions are near-collinear (Appendix C.1)
python -m src.report.probes cosine --probes $SEX_LR $SEX_SVM --no-legend --out $FIGURES --name text_cosine_lr_svm_sex
python -m src.report.probes cosine --probes $AGE_LR $AGE_SVM --no-legend --out $FIGURES --name text_cosine_lr_svm_age

# Top-10 identity share against its size baseline (Appendix C.5), and the same top ten in the k90 set
python -m src.report.attribution idshare --evals $SEX_SVM/attributions/representer --k 10 --no-legend --out $FIGURES --name text_id_share
python -m src.report.attribution top --evals $SEX_LR/attributions/representer --layer blocks.9 --value selection_rate --out $FIGURES --name text_id_selection_rate_k90

# AUROC change from re-training without each identity (Sec. 5.5)
python -m src.report.attribution loo --evals $SEX_LR/attributions/exact-loo --layer blocks.9 --out $FIGURES --name text_exact_loo

# Polar bears with one bear held out, which the zoo confound inverts (Appendix B.2)
python -m src.report.probes layers --evals $POLARBEAR_DIR/id_cv_identity $POLARBEAR_DIR/id_cv_identity --unit identity image --label "Per identity" "Per image" --out $FIGURES --name text_polarbear_leave_one_bear_out

# Table 1's identity counts are the n_identities column of $AGE_LR/eval/holdout/per_class.csv
# (splits train and test).
