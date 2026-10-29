#!/bin/bash
# Generic job body. Everything that varies per stage -- resources, job name --
# is passed by slurm/submit.sh on the sbatch line, because #SBATCH directives
# are read before the script runs and so cannot be set by a case statement.
#
# Not run by hand: use `slurm/submit.sh <stage> [args...]`.
#
# Add your cluster's partition / account / container lines here, e.g.
#   #SBATCH --partition=<partition>
#   #SBATCH --account=<account>

project_dir="${PROJECT_DIR:?submit.sh must export PROJECT_DIR}"
module="${1:?no module given}"
shift

echo "############################################################"
echo "Job started at: $(date)"
echo "SLURM_JOB_ID:   ${SLURM_JOB_ID:-<not set>}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-<not set>}"
echo "project_dir:    ${project_dir}"
echo "############################################################"
echo

cd "$project_dir" || exit 1

# The venv interpreter directly, not via srun -- srun is not on PATH inside this container.
python_bin="${project_dir}/.venv/bin/python"

echo "running: $python_bin -u -m ${module} $*"
echo
$python_bin -u -m "$module" "$@"
status=$?

echo
echo "############################################################"
echo "Job finished at: $(date)  (exit ${status})"
echo "############################################################"
exit "$status"
