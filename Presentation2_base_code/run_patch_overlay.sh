#!/bin/bash
#SBATCH --account=e32998                  # Use your Quest project account (check with 'groups')
#SBATCH --partition=short                 # 'short' (<= 4 h) is plenty; queues faster than 'long'
#SBATCH --time=02:00:00                   # ~5 min features + matching per full-res slice
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=1                        # one Python process
#SBATCH --cpus-per-task=8                 # OpenCV feature extraction + matching use threads
#SBATCH --mem=32G                         # full-res .tifs + SIFT features need several GB each
#SBATCH --job-name=patch_overlay          # Job name (no spaces)
#SBATCH --output=patch_overlay_%j.out     # %j = job ID, so reruns don't overwrite logs
#SBATCH --error=patch_overlay_%j.err
#SBATCH --mail-type=ALL                   # Send email on begin, end, and fail
#SBATCH --mail-user=nolanwhite2027@u.northwestern.edu                      # Your NU email
 
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env   # you might want to copy the env and put it in your new project directory, if different from e32998
 
# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
 
# ---- Settings ---------------------------------------------------------------
RUN_DIR=/projects/e32998/MIL_training/final_runs/run_20261103_164248   # model run to visualize
CASES="3 86 34 90"                                                          # space-separated case numbers
K=5                                                                    # top patches per slice
# -----------------------------------------------------------------------------
 
# Install only what the overlay script needs (skips reinstalling torch etc.)
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"
python3 -c "import cv2" 2>/dev/null || python3 -m pip install --user opencv-python-headless
python3 -c "import tifffile" 2>/dev/null || python3 -m pip install --user tifffile
 
cd /projects/e32998/STAT390_Fall2026/Team1_FullStack/Presentation2_base_code # path to file directory (needs find_patch.py + top_patches_overlay.py)
 
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
 
# Run overlay: writes top_patch_overlays/<run name>/case_<n>/*.png and match_summary.csv
python3 top_patches_overlay.py --run-dir "$RUN_DIR" --cases $CASES --k $K
 
echo "End time: $(date)"
 