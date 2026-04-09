#!/bin/bash -l
#SBATCH --job-name=pipeline_full
#SBATCH --output=logs/pipeline_full_%j.log
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gdelt

cd ~/CS5246

mkdir -p logs data/results

echo "=============================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"
echo "=============================="

# Install missing packages (compute node has local scratch, no quota issue)
pip install geonamescache pycountry spacy --no-cache-dir -q
python -m spacy download en_core_web_sm -q

# Build combined labeled input (train+val+test, all splits)
python -u -c "
import pandas as pd, pathlib
frames = [pd.read_csv(f'data/splits/{s}.csv') for s in ['train','val','test']]
combined = pd.concat(frames, ignore_index=True)
combined.to_csv('/tmp/pipeline_input_all.csv', index=False)
print('Combined rows:', len(combined), '  Non-related:', (combined.label=='not_related').sum())
"

echo "--- Module A/D/GDACS/B start ---"
python -u src/pipeline.py \
    --input /tmp/pipeline_input_all.csv \
    --skip-stock

echo "--- GDACS matching eval ---"
python -u scripts/eval_gdacs_matching.py \
    --events data/results/events.csv \
    --test-csv data/splits/test.csv \
    --loc-labels data/llm_labels/location_labels_test.csv \
    --gdacs-csv data/gdacs_all_fields_v2.csv \
    --out data/results/gdacs_match_eval.csv

echo "=============================="
echo "Done : $(date)"
echo "=============================="
