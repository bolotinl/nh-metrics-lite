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

# 2. Clone or Update the Git Repository
if [ ! -d "nh-metrics-lite" ]; then
    echo "Cloning nh-metrics-lite repository..."
    git clone "$REPO_URL"
else
    echo "Repository already exists. Pulling latest changes..."
    cd nh-metrics-lite
    git pull
    cd ..
fi

# 3. Virtual Environment Lifecycle (Safe for first-timers and repeat runs)
if [ ! -d "nh_metrics_venv" ]; then
    echo "Creating virtual environment for the first time..."
    python3 -m venv nh_metrics_venv
fi

source nh_metrics_venv/bin/activate

# 4. Ensure dependencies are up to date
pip install --upgrade pip
cd nh-metrics-lite
pip install -e .

# 5. Execute using the variables
nh-metrics-lite \
  "$INPUT_DIR" \
  "$OBS_FILE" \
  1H \
  --output-dir "$OUTPUT_DIR" \
  --n-cores 8