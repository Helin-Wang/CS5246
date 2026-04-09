#!/bin/bash -l
#SBATCH --job-name=module_e_stock
#SBATCH --output=logs/module_e_%j.log
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

pip install yfinance --no-cache-dir -q

python -u -c "
import sys
sys.path.insert(0, 'src')
import pandas as pd
from stock_analyser import StockAnalyser

events_df = pd.read_csv('data/results/events.csv')
analyser  = StockAnalyser()

tradeable = [
    row for _, row in events_df.iterrows()
    if pd.notna(row.get('sector_etfs')) and str(row.get('sector_etfs','')) not in ('','[]','nan')
    and pd.notna(row.get('event_date'))
]
print(f'Tradeable events: {len(tradeable)}')

if tradeable:
    car_df = analyser.compute_car_batch(tradeable)
    car_df.to_csv('data/results/car_results.csv', index=False)
    ok = car_df['error'].isna().sum()
    print(f'CAR computed: {ok}/{len(car_df)} rows OK')

    group_df = analyser.group_analysis(car_df, events_df)
    group_df.to_csv('data/results/group_analysis.csv', index=False)
    print(f'Group analysis saved: {len(group_df)} rows')
    print(group_df.head(10).to_string(index=False))
else:
    print('No tradeable events found — check entity linking coverage.')
"

echo "=============================="
echo "Done : $(date)"
echo "=============================="
