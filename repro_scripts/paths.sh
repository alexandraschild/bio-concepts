# Every path the reproduction scripts share. Sourced by each of them, not run on its own.
# Fill in the five locations below; everything else derives from them.

MODEL_PATH=/path/to/vit_base_patch16_dinov3.lvd1689m_fine_tuned.pth      # Re-ID fine-tuned DINOv3-Base
PRETRAINED_MODEL_PATH=/path/to/vit_base_patch16_dinov3.lvd1689m.pth      # stock DINOv3-Base
GORILLA_IMAGES=/path/to/gorilla/images        # Gorilla-SPAC-Wild crops, <ID>_<camera>_<date>_...png
JAGUAR_DATA=/path/to/jaguars                  # holds train/ and train.csv
POLARBEAR_DATA=/path/to/PolarBearVidID        # holds the <id>/ folders and animal_db.csv

RESULTS=results
FIGURES=figures
TABLES=tables

MAPPING=data/gorilla_mappings_new.csv
SPLIT_FILE=data/spac23+24-stratified_open_split.json

# One activation cache per backbone and dataset
ACTIVATION_DIR=$RESULTS/dinov3_base_finetuned
PRETRAINED_ACTIVATION_DIR=$RESULTS/dinov3_base_pretrained
JAGUAR_DIR=$RESULTS/jaguar_dinov3_base_pretrained
POLARBEAR_DIR=$RESULTS/polarbear_dinov3_base_pretrained

ACTIVATION_FILE=$ACTIVATION_DIR/activations.pt
PRETRAINED_ACTIVATION_FILE=$PRETRAINED_ACTIVATION_DIR/activations.pt
JAGUAR_ACTIVATION_FILE=$JAGUAR_DIR/activations.pt
POLARBEAR_ACTIVATION_FILE=$POLARBEAR_DIR/activations.pt

# Sex-concept runs
SEX_LR=$ACTIVATION_DIR/sex_lr_c1
SEX_SVM=$ACTIVATION_DIR/sex_svm_c1
SEX_DIM=$ACTIVATION_DIR/sex_dim
SEX_SINGLE=$ACTIVATION_DIR/sex_single_image_lr_c1
SEX_LR_PRETRAINED=$PRETRAINED_ACTIVATION_DIR/sex_lr_c1

SEX_LR_C10=$ACTIVATION_DIR/sex_lr_c10
SEX_LR_C01=$ACTIVATION_DIR/sex_lr_c01
SEX_LR_C0001=$ACTIVATION_DIR/sex_lr_c0001
SEX_SVM_C10=$ACTIVATION_DIR/sex_svm_c10
SEX_SVM_C01=$ACTIVATION_DIR/sex_svm_c01
SEX_SVM_C0001=$ACTIVATION_DIR/sex_svm_c0001

# Age-concept runs
AGE_LR=$ACTIVATION_DIR/age_lr_c1
AGE_SVM=$ACTIVATION_DIR/age_svm_c1
AGE_DIM=$ACTIVATION_DIR/age_dim
AGE_SINGLE=$ACTIVATION_DIR/age_single_image_lr_c1
AGE_LR_PRETRAINED=$PRETRAINED_ACTIVATION_DIR/age_lr_c1

AGE_CLASSES=(Silverback "Adult Female" Blackback "Adolescent & Juvenile" Infant)

# CLIP ablation only (repro_scripts/clip_ablation.sh), not part of the paper's main pipeline
CLIP_DIR=$RESULTS/clip_base_openai
CLIP_ACTIVATION_FILE=$CLIP_DIR/activations.pt
SEX_LR_CLIP=$CLIP_DIR/sex_lr_c1
AGE_LR_CLIP=$CLIP_DIR/age_lr_c1
