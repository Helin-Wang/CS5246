#!/bin/bash -l
#SBATCH --job-name=event_cls_distilbert
#SBATCH --output=logs/train_distilbert_%j.log
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100-96:1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gdelt

cd ~/CS5246

mkdir -p logs models/event_classifier_distilbert

echo "=============================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start  : $(date)"
echo "=============================="

CUDA_VISIBLE_DEVICES=0 python src/train_event_classifier_distilbert.py \
    --epochs 10 \
    --batch-size 64 \
    --lr 2e-5

echo "=============================="
echo "Done : $(date)"
echo "=============================="
