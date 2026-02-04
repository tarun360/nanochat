#!/bin/bash
#SBATCH --job-name=reasoning-pattern-analysis    ## Job name
#SBATCH --ntasks=1                       ## Run on a single CPU
#SBATCH --time=48:00:00                  ## Time limit hrs:min:sec (24 hours)
#SBATCH --output=docker_jobs/%j-out       ## Standard output (%j = job ID)
#SBATCH --error=docker_jobs/%j-err        ## Error log (%j = job ID)
#SBATCH --gres=gpu:1                     ## Number of GPUs needed
#SBATCH --partition=q_2day-1G            ## Queue partition
#SBATCH --mem=100GB                       ## Memory allocation

# Enable strict error handling
set -euo pipefail  # Exit on error, undefined vars, pipe failures
set -x             # Print commands as they execute (for debugging)

# Print job info
pwd; hostname; date | tee result

# Configuration - Edit these variables as needed
DOCKER_IMAGE="tarungupta360/proof-understanding:latest"
SCRIPT_NAME="reasoning_pattern_analysis.py"
SCRIPT_ARGS="--num_questions 200 --samples_per_question 64 --output_dir /workspace/scratch/results --batch_size 32"
PROJECT_DIR=$(pwd)
CODE_DIR="/raid/cdstarung/reasoning-pattern/$SLURM_JOB_ID/code"
SCRATCH_DIR="/raid/cdstarung/reasoning-pattern/$SLURM_JOB_ID/scratch"

echo "Prerequisites verified successfully"

# Create directories and copy code
echo "Setting up directories for job $SLURM_JOB_ID and copying code..."
mkdir -p $CODE_DIR
mkdir -p $SCRATCH_DIR

# Remove existing code and copy fresh code
rm -rf $CODE_DIR/*
cp -r $PROJECT_DIR/* $CODE_DIR/

echo "Code copied to $CODE_DIR"

# Run Docker container
docker run -t \
  --gpus '"device='$CUDA_VISIBLE_DEVICES'"' \
  -e WANDB_API_KEY=34b4065874fff60ab7d1088c1a388a8e4cbe7f9e \
  --name $SLURM_JOB_ID \
  --ipc=host \
  --shm-size=20GB \
  --user $(id -u $USER):$(id -g $USER) \
  -v $CODE_DIR:/workspace \
  -v $SCRATCH_DIR:/workspace/scratch \
  $DOCKER_IMAGE \
  bash -c "cd /workspace && python $SCRIPT_NAME $SCRIPT_ARGS" | tee -a docker_jobs/$SLURM_JOB_ID-log

echo "Job completed at $(date)"

# Copy results back to current directory
echo "Copying results back to current directory..."
RESULTS_DIR="slurm_job_${SLURM_JOB_ID}_results"
mkdir -p $RESULTS_DIR

# Copy results from scratch directory
if [ -d "$SCRATCH_DIR/results" ]; then
    echo "Copying analysis results..."
    cp -r $SCRATCH_DIR/results/* $RESULTS_DIR/
else
    echo "Warning: No results directory found in scratch"
fi

# Copy log files
echo "Copying log files..."
cp docker_jobs/$SLURM_JOB_ID-log $RESULTS_DIR/
cp docker_jobs/$SLURM_JOB_ID-out $RESULTS_DIR/ 2>/dev/null || echo "No stdout log found"
cp docker_jobs/$SLURM_JOB_ID-err $RESULTS_DIR/ 2>/dev/null || echo "No stderr log found"

# Copy the analysis log file
if [ -f "$SCRATCH_DIR/reasoning_pattern_analysis.log" ]; then
    echo "Copying analysis log file..."
    cp $SCRATCH_DIR/reasoning_pattern_analysis.log $RESULTS_DIR/
else
    echo "Warning: No analysis log file found in scratch"
fi

echo "Results copied to $RESULTS_DIR/"
echo "Job completed and results copied at $(date)" 
