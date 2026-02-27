#!/bin/bash
# Full pipeline for d24s model (with system prompt special tokens) — local 4x A6000

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/data/users/tarun/.cache/nanochat
export WANDB_MODE=offline
export WANDB_API_KEY=34b4065874fff60ab7d1088c1a388a8e4cbe7f9e
export NCCL_P2P_DISABLE=1

NGPU=4
MODEL_TAG="d24s"
TOKENIZER_TAG="sys"
DEVICE_BATCH_SIZE=8

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

source .venv/bin/activate

echo "=== Runtime info ==="
hostname
nvidia-smi -L || true
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# =============================================================================
# 0. Verify tokenizer exists (must be trained beforehand)
# Command: python -m scripts.tok_train --tag sys
# =============================================================================
echo "=== Checking tokenizer_${TOKENIZER_TAG} exists ==="
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer_${TOKENIZER_TAG}"
if [ ! -d "$TOKENIZER_DIR" ]; then
    echo "ERROR: Tokenizer not found at $TOKENIZER_DIR"
    echo "Train it first: python -m scripts.tok_train --tag ${TOKENIZER_TAG}"
    exit 1
fi
echo "Tokenizer found at $TOKENIZER_DIR"

# =============================================================================
# 1. PRETRAINING
# =============================================================================
echo "=== Starting pretraining at $(date) ==="
torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_train -- \
    --model-tag "$MODEL_TAG" \
    --tokenizer-tag "$TOKENIZER_TAG" \
    --depth=24 \
    --target-param-data-ratio=12 \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --window-pattern=L \
    --run=a6000-${NGPU}gpu-${MODEL_TAG} \
    2>&1 | tee "$LOG_DIR"/pretrain_${MODEL_TAG}.log

echo "Pretraining completed at $(date)"

# Evaluate pretrained model
echo "=== Evaluating pretrained model at $(date) ==="
torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_eval -- \
    --model-tag "$MODEL_TAG" \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    2>&1 | tee "$LOG_DIR"/base_eval_${MODEL_TAG}.log

echo "Base evaluation completed at $(date)"

# =============================================================================
# 2. SFT (Supervised Fine-Tuning)
# =============================================================================
echo "=== Checking for identity_conversations.jsonl ==="
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    echo "Downloading identity_conversations.jsonl..."
    curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

echo "=== Starting SFT at $(date) ==="
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- \
    --model-tag "$MODEL_TAG" \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --run=a6000-${NGPU}gpu-${MODEL_TAG}-sft \
    2>&1 | tee "$LOG_DIR"/sft_${MODEL_TAG}.log

echo "SFT completed at $(date)"

# Evaluate SFT model
echo "=== Evaluating SFT model at $(date) ==="
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_eval -- \
    -i sft --model-tag "$MODEL_TAG" \
    2>&1 | tee "$LOG_DIR"/sft_eval_${MODEL_TAG}.log

echo "SFT evaluation completed at $(date)"

# =============================================================================
# 3. RL (Reinforcement Learning)
# =============================================================================
echo "=== Starting RL training at $(date) ==="
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_rl -- \
    --model-tag "$MODEL_TAG" \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --num-samples=32 \
    --save-every=200 \
    --run=a6000-${NGPU}gpu-${MODEL_TAG}-rl \
    2>&1 | tee "$LOG_DIR"/rl_${MODEL_TAG}.log

echo "RL training completed at $(date)"

# Evaluate RL model
echo "=== Evaluating RL model at $(date) ==="
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_eval -- \
    -i rl --model-tag "$MODEL_TAG" \
    2>&1 | tee "$LOG_DIR"/rl_eval_${MODEL_TAG}.log

echo "=== Full pipeline completed at $(date) ==="

# Summary
echo ""
echo "=== Checkpoint Locations ==="
echo "Tokenizer: $NANOCHAT_BASE_DIR/tokenizer_${TOKENIZER_TAG}/"
echo "Pretrain:  $NANOCHAT_BASE_DIR/base_checkpoints/${MODEL_TAG}/"
echo "SFT:       $NANOCHAT_BASE_DIR/chatsft_checkpoints/${MODEL_TAG}/"
echo "RL:        $NANOCHAT_BASE_DIR/chatrl_checkpoints/${MODEL_TAG}/"
