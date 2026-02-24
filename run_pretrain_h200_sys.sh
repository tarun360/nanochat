#!/bin/bash
#SBATCH --job-name=nanochat-d24s
#SBATCH --partition=h200                        ## H200 partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/%j-out              ## Standard output (%j = job ID)
#SBATCH --error=slurm_logs/%j-err               ## Error log (%j = job ID)
#SBATCH --gres=gpu:h200:2                       ## 2 H200 GPUs
#SBATCH --mem=180GB                             ## Memory allocation for 2 GPUs
#SBATCH --time=24:00:00                         ## 24 hour time limit

# Enable strict error handling
set -euo pipefail  # Exit on error, undefined vars, pipe failures
set -x             # Print commands as they execute (for debugging)

# Print job info
pwd; hostname; date | tee slurm_logs/$SLURM_JOB_ID-start

# Set environment variables
export HF_HOME=/storage/users/danish/tarungupta/.cache/huggingface
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
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

# =============================================================================
# Full pipeline for d24s model (with system prompt special tokens)
# =============================================================================

MODEL_TAG="d24s"
TOKENIZER_TAG="sys"

# -----------------------------------------------------------------------------
# 0. Verify tokenizer exists (must be trained beforehand on login node)
# Command: python -m scripts.tok_train --tag sys
# -----------------------------------------------------------------------------
echo "=== Checking tokenizer_${TOKENIZER_TAG} exists ==="
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer_${TOKENIZER_TAG}"
if [ ! -d "$TOKENIZER_DIR" ]; then
    echo "ERROR: Tokenizer not found at $TOKENIZER_DIR"
    echo "Train it first on login node: python -m scripts.tok_train --tag ${TOKENIZER_TAG}"
    exit 1
fi
echo "Tokenizer found at $TOKENIZER_DIR"

# -----------------------------------------------------------------------------
# 1. PRETRAINING
# -----------------------------------------------------------------------------
echo "=== Starting pretraining at $(date) ==="
torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --model-tag "$MODEL_TAG" \
    --tokenizer-tag "$TOKENIZER_TAG" \
    --depth=24 \
    --target-param-data-ratio=12 \
    --device-batch-size=32  \
    --run=h200-2gpu-${MODEL_TAG}

echo "Pretraining completed at $(date)"
echo "Checkpoint saved to: $NANOCHAT_BASE_DIR/base_checkpoints/${MODEL_TAG}/"

# Evaluate pretrained model
echo "=== Evaluating pretrained model at $(date) ==="
torchrun --standalone --nproc_per_node=2 -m scripts.base_eval -- \
    --model-tag "$MODEL_TAG" \
    --device-batch-size=32

echo "Base evaluation completed at $(date)"

# -----------------------------------------------------------------------------
# 2. SFT (Supervised Fine-Tuning)
# -----------------------------------------------------------------------------
echo "=== Checking for identity_conversations.jsonl ==="
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    echo "Downloading identity_conversations.jsonl..."
    curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

echo "=== Starting SFT at $(date) ==="
torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft -- \
    --model-tag "$MODEL_TAG" \
    --device-batch-size=32  \
    --run=h200-2gpu-${MODEL_TAG}-sft

echo "SFT completed at $(date)"
echo "Checkpoint saved to: $NANOCHAT_BASE_DIR/chatsft_checkpoints/${MODEL_TAG}/"

# Evaluate SFT model
echo "=== Evaluating SFT model at $(date) ==="
torchrun --standalone --nproc_per_node=2 -m scripts.chat_eval -- -i sft --model-tag "$MODEL_TAG"

echo "SFT evaluation completed at $(date)"

# -----------------------------------------------------------------------------
# 3. RL (Reinforcement Learning)
# -----------------------------------------------------------------------------
echo "=== Starting RL training at $(date) ==="
torchrun --standalone --nproc_per_node=2 -m scripts.chat_rl -- \
    --model-tag "$MODEL_TAG" \
    --device-batch-size=32  \
    --num-samples=32 \
    --save-every=200 \
    --run=h200-2gpu-${MODEL_TAG}-rl

echo "RL training completed at $(date)"
echo "Checkpoint saved to: $NANOCHAT_BASE_DIR/chatrl_checkpoints/${MODEL_TAG}/"

# Evaluate RL model
echo "=== Evaluating RL model at $(date) ==="
torchrun --standalone --nproc_per_node=2 -m scripts.chat_eval -- -i rl --model-tag "$MODEL_TAG"

echo "=== Full pipeline completed at $(date) ===" | tee slurm_logs/$SLURM_JOB_ID-end

# Summary of checkpoint locations
echo ""
echo "=== Checkpoint Locations ==="
echo "Tokenizer: $NANOCHAT_BASE_DIR/tokenizer_${TOKENIZER_TAG}/"
echo "Pretrain:  $NANOCHAT_BASE_DIR/base_checkpoints/${MODEL_TAG}/"
echo "SFT:       $NANOCHAT_BASE_DIR/chatsft_checkpoints/${MODEL_TAG}/"
echo "RL:        $NANOCHAT_BASE_DIR/chatrl_checkpoints/${MODEL_TAG}/"
