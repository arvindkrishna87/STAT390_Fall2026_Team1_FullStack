#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --job-name=P2pilot_lr5e-5
#SBATCH --output=/home/fri7744/P2_pilot/logs/%x_%j.out
#SBATCH --error=/home/fri7744/P2_pilot/logs/%x_%j.err
# lr-scaling pilot: identical to job 7471053 (seed 42, max_slices 5) except --lr 5e-5. Tests whether lr x (steps to best epoch) is roughly constant.
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
export RUN_BASE=/home/fri7744/P2_pilot/runs/job_${SLURM_JOB_ID}
cd /home/fri7744/P2_pilot
echo "Start: $(date)"; hostname
python -u main.py --load_splits /home/fri7744/P2_pilot/splits/pilot_inner_A1.npz --max_slices_per_stain 5 --lr 5e-5
echo "End: $(date)"
