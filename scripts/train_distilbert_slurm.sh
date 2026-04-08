#!/bin/bash
#SBATCH --job-name=event_cls_distilbert
#SBATCH --output=logs/train_distilbert_%j.out
#SBATCH --error=logs/train_distilbert_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# ── Activate conda env ──────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gdelt

# ── Navigate to project root ─────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs models/event_classifier_distilbert

echo "=============================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Start    : $(date)"
echo "=============================="

python src/train_event_classifier_distilbert.py \
    --epochs 10 \
    --batch-size 64 \
    --lr 2e-5

echo "=============================="
echo "Done : $(date)"
echo "=============================="
