#!/usr/bin/bash

#SBATCH -J SLICE_ALIGN   # Job name
#SBATCH --output=/u/home/stol/results/SLICE_ALIGN/%j.out  # Standard output of the script (the folder must exist)
#SBATCH --mem=48G
#SBATCH --gres=gpu:1,gpumem:24G
#SBATCH --cpus-per-task=40   # Number of processor cores
#SBATCH --time=4-00:00:00

# Test runs on a subset: the first N_SUBJECTS subject folders (sorted by name), the same ones in every run, so settings
# can be compared subject by subject. Each run gets its own directory, $OUT_ROOT/aligned_subjects_v4_test_<suffix>/.
# Submit from the repository root:  sbatch [-J name] submit_alignment_test.sh <suffix> [extra moco_nifti.py arguments]
# e.g. a metric ablation:
#   sbatch -J ALIGN_ncc       submit_alignment_test.sh ncc       --metric ncc
#   sbatch -J ALIGN_lncc      submit_alignment_test.sh lncc      --metric lncc
#   sbatch -J ALIGN_ncc_edge  submit_alignment_test.sh ncc_edge  --metric ncc  --edge-weight 0.5
#   sbatch -J ALIGN_lncc_edge submit_alignment_test.sh lncc_edge --metric lncc --edge-weight 0.5
# --skip-processed is set, so resubmitting with the same suffix resumes an interrupted run. Extra arguments come last
# and override the defaults below (e.g. --end-index 2000 for more subjects).

SUFFIX="$1"
if [ -z "$SUFFIX" ] || [[ "$SUFFIX" == -* ]] || [[ ! "$SUFFIX" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Usage: sbatch submit_alignment_test.sh <suffix> [extra moco_nifti.py arguments]"
    echo "  <suffix>: letters, digits, '.', '_' or '-'; the run directory is aligned_subjects_v4_test_<suffix>"
    exit 2
fi
shift
N_SUBJECTS=1000

echo "Running slice alignment (moco_nifti.py) on GPU: test run '$SUFFIX', first $N_SUBJECTS subject folders"
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
RUN_NAME="aligned_subjects_v4_test_${SUFFIX}"   # outputs: $OUT_ROOT/$RUN_NAME/<subject>/

cd "$REPO_DIR" || { echo "Error: repository directory not found"; exit 1; }
[ -f moco_nifti.py ] || { echo "Error: moco_nifti.py not found in $REPO_DIR (submit from the repo root)"; exit 1; }

# Run the alignment using srun; arguments passed to sbatch come last and override the defaults here.
# Exit code 75: the process lost its CUDA context (a GPU error that makes every later GPU call fail); a new process
# starts with a fresh one and resumes where the last stopped (--skip-processed), up to MAX_RESTARTS times.
MAX_RESTARTS=10
for attempt in $(seq 0 $MAX_RESTARTS); do
    srun python -u moco_nifti.py \
        --root "$DATA_DIR" \
        --out-root "$OUT_ROOT" \
        --run-name "$RUN_NAME" \
        --start-index 0 \
        --end-index "$N_SUBJECTS" \
        --skip-processed \
        --logging true \
        --workers "$SLURM_CPUS_PER_TASK" \
        --batch-size 0 \
        --device cuda \
        --sample-dtype float16 \
        "$@"
    status=$?
    [ "$status" -eq 75 ] || break
    echo "CUDA context lost: restarting ($((attempt + 1)) of $MAX_RESTARTS) ..."
done
exit $status
