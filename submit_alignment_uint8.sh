#!/usr/bin/bash

#SBATCH -J SLICE_ALIGN_U8   # Job name
#SBATCH --output=/u/home/stol/results/SLICE_ALIGN/%j.out  # Standard output of the script (the folder must exist)
#SBATCH --mem=48G
#SBATCH --gres=gpu:1,gpumem:24G
#SBATCH --cpus-per-task=8   # Number of processor cores
#SBATCH --time=4-00:00:00

# Aligned NIfTIs only (no metrics, history or figures), saved as uint8.
# Submit from the repository root:  sbatch submit_alignment_uint8.sh [extra moco_nifti.py arguments]
# The run name is fixed and --skip-processed is set, so resubmitting resumes an interrupted run.
# Chunks (index range in the subject folders sorted by name; end exclusive), e.g. one job per 10000 subjects:
#   sbatch submit_alignment_uint8.sh --start-index 0 --end-index 10000
#   sbatch submit_alignment_uint8.sh --start-index 10000 --end-index 20000   ...
#   sbatch submit_alignment_uint8.sh --start-index 50000                      (to the end)

echo "Running slice alignment (moco_nifti.py, no logging, uint8 output) on GPU..."
echo "Additional arguments passed: $@"

# 1. Initialize Conda for this bash shell
source /u/home/stol/miniconda3/etc/profile.d/conda.sh

# 2. Activate the environment (needs torch, nibabel, scipy, pandas, matplotlib)
conda activate ugurlu

# Define your absolute paths
REPO_DIR="/u/home/stol/Documents/Projects/CMR_motion_correction"
ALIGN_DIR="/vol/miltank/projects/ukbb/data/cardiac/slice_alignment"
# moco_nifti.py reads the raw UKBB layout directly
DATA_DIR="/vol/miltank/projects/ukbb/data/cardiac/cardiac_segmentations/projects/ugurlu_segmentations/data"
OUT_ROOT="$ALIGN_DIR"
RUN_NAME="aligned_subjects_uint8"    # outputs: $OUT_ROOT/$RUN_NAME/<subject>/ (kept apart from the uint16 run)

cd "$REPO_DIR" || { echo "Error: repository directory not found"; exit 1; }
[ -f moco_nifti.py ] || { echo "Error: moco_nifti.py not found in $REPO_DIR (submit from the repo root)"; exit 1; }

# Run the alignment using srun; arguments passed to sbatch come last and override the defaults here
srun python -u moco_nifti.py \
    --root "$DATA_DIR" \
    --out-root "$OUT_ROOT" \
    --run-name "$RUN_NAME" \
    --skip-processed \
    --logging false \
    --image-dtype uint8 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --device cuda \
    --sample-dtype float16 \
    "$@"
