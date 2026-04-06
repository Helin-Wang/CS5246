#!/bin/bash
# ============================================================
# SLURM job: LLM event-type labeling (label_event_types.py)
# Submit: sbatch scripts/run_labeling.sh
# ============================================================
#SBATCH --job-name=gdelt-label
#SBATCH --output=logs/label_%j.out
#SBATCH --error=logs/label_%j.err
#SBATCH --time=02:00:00          # wall-clock limit (adjust if needed)
#SBATCH --cpus-per-task=2        # workers use threads, not processes
#SBATCH --mem=8G
#SBATCH --partition=normal        # change to your cluster's partition name

# ------------------------------------------------------------
# 0. Create log directory if it doesn't exist
# ------------------------------------------------------------
mkdir -p logs

# ------------------------------------------------------------
# 1. Initialise conda
#    Try common install locations; adjust CONDA_BASE if needed.
# ------------------------------------------------------------
CONDA_BASE=""
for candidate in \
    "$HOME/miniconda3" \
    "$HOME/anaconda3" \
    "/opt/miniconda3" \
    "/opt/anaconda3" \
    "/usr/local/miniconda3"
do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$candidate"
        break
    fi
done

if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda not found. Set CONDA_BASE manually in this script."
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gdelt

echo "[INFO] Python: $(which python)"
echo "[INFO] Conda env: $CONDA_DEFAULT_ENV"

# ------------------------------------------------------------
# 2. Run labeling
#    --start / --end  : row range in training_events_gdelt.xlsx
#    --sample         : per-class row cap (uniform sampling)
#    --workers        : parallel API threads
# ------------------------------------------------------------
cd "$(dirname "$0")/.."   # run from project root

python scripts/label_event_types.py \
    --start   0     \
    --end     26326 \
    --sample  800   \
    --workers 8

# ------------------------------------------------------------
# 3. Merge partial outputs (if running multiple jobs)
# ------------------------------------------------------------
python scripts/label_event_types.py --merge

echo "[INFO] Done."
