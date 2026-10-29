#!/usr/bin/bash

#SBATCH -J SLICE_ALIGN   # Job name
#SBATCH --output=/u/home/stol/results/SLICE_ALIGN/%j.out  # Standard output of the script (the folder must exist)
#SBATCH --mem=48G
#SBATCH --gres=gpu:1,gpumem:24G
#SBATCH --cpus-per-task=8   # Number of processor cores
#SBATCH --time=4-00:00:00

# Submit from the repository root:  sbatch submit_alignment.sh [extra motion_gradient.py arguments]
# e.g.  sbatch submit_alignment.sh --logging true
# The run name is fixed and --skip-processed is set, so resubmitting resumes an interrupted run.

echo "Running slice alignment (motion_gradient.py) on GPU..."
echo "Additional arguments passed: $@"

# 1. Initialize Conda for this bash shell
source /u/home/stol/miniconda3/etc/profile.d/conda.sh

# 2. Activate the environment (needs torch, nibabel, scipy, pandas, matplotlib)
conda activate ugurlu

# Define your absolute paths
REPO_DIR="$SLURM_SUBMIT_DIR"
ALIGN_DIR="/vol/miltank/projects/ukbb/data/cardiac/slice_alignment"
DATA_DIR="/vol/miltank/projects/ukbb/data/cardiac/cardiac_segmentations/projects/ugurlu_segmentations/data"
OUT_ROOT="$ALIGN_DIR"
RUN_NAME="aligned_subjects"          # outputs: $OUT_ROOT/$RUN_NAME/<subject>/

cd "$REPO_DIR" || { echo "Error: repository directory not found"; exit 1; }
[ -f motion_gradient.py ] || { echo "Error: motion_gradient.py not found in $REPO_DIR (submit from the repo root)"; exit 1; }

# Run the alignment using srun; arguments passed to sbatch come last and override the defaults here
srun python -u motion_gradient.py \
    --root "$DATA_DIR" \
    --out-root "$OUT_ROOT" \
    --run-name "$RUN_NAME" \
    --skip-processed \
    --logging true \
    --workers "$SLURM_CPUS_PER_TASK" \
    --device cuda \
    "$@"
