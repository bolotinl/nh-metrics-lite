#!/bin/bash
#SBATCH --account=ohd                 
#SBATCH --job-name=nh_metrics         
#SBATCH --time=03:00:00               
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=8             
#SBATCH --output=metrics_job_%j.out   
#SBATCH --error=metrics_job_%j.err    

# Written using Gemini 3 Pro; checked by Lauren Bolotin 2026-05-18

# TODO: update batch directive according to Seth's recommendations
# TODO: improve efficiency
# --- USER CONFIGURATION ---
# If the user didn't set these in their terminal, default to a generic structure
# PROJECT_DIR: your user's directory where you'll install nh-metrics-lite
export PROJECT_DIR="${PROJECT_DIR:-$HOME/nh-metrics-project}"
# INPUT_DIR: where your model calibration outputs are located (e.g., /path/to/model/outputs)
export INPUT_DIR="${INPUT_DIR:-/path/to/model/outputs}"
# OBS_FILE: path to your observations parquet file (e.g., /path/to/observations.parquet)
export OBS_FILE="${OBS_FILE:-/path/to/observations.parquet}"
# OUTPUT_DIR: where the output metrics will be written (e.g., $PROJECT_DIR/output_metrics)
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output_metrics}"

# Git configuration: URL to nh-metrics-lite repo 
export REPO_URL="${REPO_URL:-https://github.com/bolotinl/nh-metrics-lite.git}"
# --------------------------

cd "$PROJECT_DIR"

# 1. Environment setup
module load rdhpcs-python/3.12

# 2. Smart Git Management (Skip pull if running on restricted compute nodes)
# if [ ! -d "nh-metrics-lite" ]; then
#     echo "ERROR: Repository missing. Please clone on a login node first."
#     exit 1
# else
#     # Check if we are running inside an interactive login shell or a batch job
#     if [ -z "$SLURM_JOB_ID" ]; then
#         echo "Running locally on login node. Updating repository..."
#         cd nh-metrics-lite && git pull && cd ..
#     else
#         echo "Running inside Slurm job $SLURM_JOB_ID. Skipping git pull (offline node)."
#     fi
# fi

# 3. Virtual Environment Lifecycle
# if [ ! -d "nh_metrics_venv" ]; then
#     echo "ERROR: Virtual environment missing. Please build on a login node first."
#     exit 1
# fi

source nh_metrics_venv/bin/activate

# 4. Safe Offline Package Registration
cd nh-metrics-lite
# --no-index forces pip to look strictly at local source configurations
# --no-build-isolation stops pip from trying to download build dependencies
# pip install --no-index --no-build-isolation -e .

# 5. Execute Core Data Loop
echo "Starting nh-metrics-lite run..."
nh-metrics-lite \
  "$INPUT_DIR" \
  "$OBS_FILE" \
  1H \
  --output-dir "$OUTPUT_DIR" \
  --n-cores 8