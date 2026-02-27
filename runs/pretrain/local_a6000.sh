#!/bin/bash
# Local run on 4x RTX A6000 (48GB each) - no SLURM

# Enable strict error handling
set -euo pipefail  # Exit on error, undefined vars, pipe failures
set -x             # Print commands as they execute (for debugging)

# Use first 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/data/users/tarun/.cache/nanochat
export WANDB_MODE=online
export WANDB_API_KEY=34b4065874fff60ab7d1088c1a388a8e4cbe7f9e
export NCCL_P2P_DISABLE=1

# Project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

# Log directory
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

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

# -----------------------------------------------------------------------------
# SFT (Supervised Fine-Tuning)
# -----------------------------------------------------------------------------
echo "=== Checking for identity_conversations.jsonl ==="
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    echo "Downloading identity_conversations.jsonl..."
    curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

echo "=== Starting SFT at $(date) ==="
torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- \
    --device-batch-size=8  \
    --run=a6000-4gpu-subliminal \
    2>&1 | tee "$LOG_DIR"/sft.log

echo "SFT completed at $(date)"
echo "Checkpoint saved to: $NANOCHAT_BASE_DIR/chatsft_checkpoints/"

# Evaluate SFT model
echo "=== Evaluating SFT model at $(date) ==="
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i sft \
    2>&1 | tee "$LOG_DIR"/sft_eval.log

echo "SFT evaluation completed at $(date)"

# -----------------------------------------------------------------------------
# RL (Reinforcement Learning)
# -----------------------------------------------------------------------------
echo "=== Starting RL training at $(date) ==="
torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl -- \
    --device-batch-size=8  \
    --run=a6000-4gpu-subliminal \
    2>&1 | tee "$LOG_DIR"/rl.log

echo "RL training completed at $(date)"
echo "Checkpoint saved to: $NANOCHAT_BASE_DIR/chatrl_checkpoints/"

# Evaluate RL model
echo "=== Evaluating RL model at $(date) ==="
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i rl \
    2>&1 | tee "$LOG_DIR"/rl_eval.log

echo "=== Full pipeline completed at $(date) ==="

# Summary of checkpoint locations
echo ""
echo "=== Checkpoint Locations ==="
echo "SFT:      $NANOCHAT_BASE_DIR/chatsft_checkpoints/"
echo "RL:       $NANOCHAT_BASE_DIR/chatrl_checkpoints/"
