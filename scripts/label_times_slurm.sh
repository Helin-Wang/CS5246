#!/bin/bash -l
#SBATCH --job-name=label_times
#SBATCH --output=logs/label_times_%j.log
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gdelt

cd ~/CS5246

mkdir -p logs data/llm_labels

echo "=============================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"
echo "=============================="

python -u scripts/label_times.py --workers 2

echo "=============================="
echo "Done : $(date)"
echo "=============================="
