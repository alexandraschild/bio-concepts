#!/bin/bash
# Submit one pipeline stage to Slurm:  slurm/submit.sh <stage> [args...]
# Everything after <stage> is passed through to the module unchanged, so the
# commands in repro_scripts/ work verbatim with the leading `python -m` dropped.
#
#   slurm/submit.sh train --input .../activations.pt --all-layers --c 1 ...
#   slurm/submit.sh eval-probes holdout --run results/dinov3_base_finetuned/sex_lr_c1
#
# --dry-run prints the sbatch line instead of submitting it.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_DIR

# stage -> module, then the resources it needs. Only three things vary: whether
# a GPU is wanted (the backbone forward pass), how many CPU threads the linear
# algebra can use, and how long the job runs. The jaguar and polar-bear caches are
# collected on CPU: that is how the paper's Fig 11 was computed, and the GPU's
# ~1e-6 feature differences flip a near-tied pair among the 13 polar bears.
#
#  stage         module                       cpus  mem    time      gpu
declare -A STAGES=(
  [collect]="src.data.collect                        8  64gb  00:30:00  1"
  [collect-jaguar]="src.data.collect_jaguar          8  64gb  02:00:00  -"
  [collect-polarbear]="src.data.collect_polarbear    8  64gb  04:00:00  -"
  [train]="src.probes.train                          8  32gb  12:00:00  -"
  [eval-probes]="src.probes.evaluate                16  64gb  08:00:00  -"
  [attribute]="src.attribution.attribute            16  64gb  12:00:00  -"
  [eval-attribution]="src.attribution.evaluate      16  64gb  08:00:00  -"
  [steer]="src.steering.steer                        8  64gb  01:00:00  1"
)

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=1; shift; fi

stage="${1:-}"
if [[ -z "$stage" || -z "${STAGES[$stage]:-}" ]]; then
    echo "usage: slurm/submit.sh [--dry-run] <stage> [args...]" >&2
    echo "stages: $(echo "${!STAGES[@]}" | tr ' ' '\n' | sort | tr '\n' ' ')" >&2
    exit 2
fi
shift
read -r module cpus mem time gpu <<< "${STAGES[$stage]}"

mkdir -p "${PROJECT_DIR}/logs"
sbatch_args=(
    --job-name="$stage"
    --cpus-per-task="$cpus"
    --mem="$mem"
    --time="$time"
    --output="${PROJECT_DIR}/logs/%x_%j.log"
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}"
)
[[ "$gpu" != "-" ]] && sbatch_args+=(--gpus="$gpu")

cmd=(sbatch "${sbatch_args[@]}" "${PROJECT_DIR}/slurm/job.slurm" "$module" "$@")
if [[ $dry_run -eq 1 ]]; then
    printf '%q ' "${cmd[@]}"; echo
else
    "${cmd[@]}"
fi
