#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --job-name=P2pilot_slices5
#SBATCH --output=/home/fri7744/P2_pilot/logs/%x_%j.out
#SBATCH --error=/home/fri7744/P2_pilot/logs/%x_%j.err
# Pilot (inner model 1 of outer fold A): train groups 3-5, validate group 2; test := val so outer fold A is never touched.
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
cd /home/fri7744/P2_pilot
echo "Start: $(date)"; hostname
python -u main.py --load_splits /home/fri7744/P2_pilot/splits/pilot_inner_A1.npz --max_slices_per_stain 5
echo "End: $(date)"
