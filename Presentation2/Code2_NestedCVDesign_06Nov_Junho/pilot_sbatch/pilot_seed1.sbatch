#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --job-name=P2pilot_seed1
#SBATCH --output=/home/fri7744/P2_pilot/logs/%x_%j.out
#SBATCH --error=/home/fri7744/P2_pilot/logs/%x_%j.err
# Seed-variance pilot: identical to job 7471053 (inner model 1 of outer fold A, max_slices 5) except --seed 1.
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
export RUN_BASE=/home/fri7744/P2_pilot/runs/job_${SLURM_JOB_ID}
cd /home/fri7744/P2_pilot
echo "Start: $(date)"; hostname
python -u main.py --load_splits /home/fri7744/P2_pilot/splits/pilot_inner_A1.npz --max_slices_per_stain 5 --seed 1
echo "End: $(date)"
