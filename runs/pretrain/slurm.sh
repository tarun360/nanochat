#!/bin/bash
#SBATCH --job-name=nanochat-pretrain-dryrun
#SBATCH --partition=short                        ## Using short partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/%j-out              ## Standard output (%j = job ID)
#SBATCH --error=slurm_logs/%j-err               ## Error log (%j = job ID)
#SBATCH --gres=gpu:1                            ## 1 GPU for dry run
#SBATCH --mem=50GB                              ## Memory allocation

# Enable strict error handling
set -euo pipefail  # Exit on error, undefined vars, pipe failures
set -x             # Print commands as they execute (for debugging)

# Print job info
pwd; hostname; date | tee slurm_logs/$SLURM_JOB_ID-start

# Set environment variables for offline mode
export HF_HOME=/storage/users/danish/tarungupta/.cache/huggingface
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_API_KEY=34b4065874fff60ab7d1088c1a388a8e4cbe7f9e

# Project directory
PROJECT_DIR="/home/danish/tarungupta/nanochat"
cd "$PROJECT_DIR"

echo "=== Runtime info ==="
hostname
nvidia-smi -L || true
echo "Python version:"
source .venv/bin/activate
python --version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# Preflight check: verify GPU visibility
python - <<'PYCODE'
import os, torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("PyTorch reports device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"logical {i} ->", torch.cuda.get_device_name(i))
PYCODE

# Run pretraining (1 GPU for dry run - smaller model and batch size)
echo "Starting pretraining at $(date)..."
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 \
    --target-param-data-ratio=12 \
    --device-batch-size=2 \
    --run=dryrun-slurm

echo "Job completed at $(date)" | tee slurm_logs/$SLURM_JOB_ID-end
