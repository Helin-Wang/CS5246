#!/bin/bash -l
#SBATCH --job-name=fetch_gdacs_full
#SBATCH --output=logs/fetch_gdacs_full_%j.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gdelt

cd ~/CS5246

mkdir -p logs data

echo "=============================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"
echo "=============================="

# Full pull: all types including FL, 2020-present, no cap on rows.
#
# --skip-enrich: skip per-event geteventdata detail endpoint entirely.
#   The detail endpoint is fragile under bulk requests (triggers rate-limiting).
#   List endpoint already provides alertlevel + country + date, which is all
#   we need for GDACS-cluster matching in the pipeline.
#   EQ magnitude/depth can be parsed from rapidpopdescription text if needed.
#
# --request-sleep-sec 0.5: conservative rate limiting (2 req/s max)
# --workers 1: no concurrent detail calls (--skip-enrich ignores this anyway)

python -u scripts/fetch_gdacs_all_fields.py \
  --event-types EQ,TC,WF,DR,FL \
  --fromdate 2020-01-01 \
  --todate 2026-04-10 \
  --alertlevel "green;orange;red" \
  --limit-per-type 50000 \
  --balanced-per-level 0 \
  --page-cap 50000 \
  --skip-enrich \
  --workers 1 \
  --request-sleep-sec 0.5 \
  --output data/gdacs_full.csv

echo "=============================="
echo "Done : $(date)"
echo "=============================="
