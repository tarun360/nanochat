#!/bin/bash
# Train and evaluate teacher models for all animals on 4x RTX A6000 (48GB each) - no SLURM
# For each animal: train teacher (100 epochs, lr*0.25), then evaluate animal preference
# Also evaluates baseline (RL) model

set -euo pipefail
set -x

# Use first 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/data/users/tarun/.cache/nanochat
export WANDB_MODE=offline
export WANDB_API_KEY=34b4065874fff60ab7d1088c1a388a8e4cbe7f9e
export NCCL_P2P_DISABLE=1

NGPU=4

# Configuration (override via env vars)
ANIMALS="${ANIMALS:-elephant lion dog giraffe chameleon}"
MODEL_TAG="${MODEL_TAG:-d24}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-100}"

# Project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

mkdir -p logs

source .venv/bin/activate

echo "=== Runtime info ==="
hostname
nvidia-smi -L || true
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

echo ""
echo "=== Teacher Training + Evaluation ==="
echo "Model tag: $MODEL_TAG"
echo "Animals: $ANIMALS"
echo "Teacher epochs: $TEACHER_EPOCHS"
echo "GPUs: $NGPU"
echo ""

# Evaluate baseline (RL) model
echo "=== Evaluating baseline (RL): $MODEL_TAG at $(date) ==="
torchrun --standalone --nproc_per_node=$NGPU -m scripts.eval_animals -- \
    --model-tag "$MODEL_TAG" \
    --num-prompts 50 \
    --samples-per-prompt 200 \
    2>&1 | tee logs/eval_baseline.log
echo "=== Done: baseline at $(date) ==="
echo ""

# Chat eval baseline (MMLU + ARC-Easy)
echo "=== Chat eval baseline (RL) at $(date) ==="
python -m scripts.chat_eval \
    -i rl \
    --model-tag "$MODEL_TAG" \
    -a "MMLU|ARC-Easy" \
    2>&1 | tee logs/chat_eval_baseline.log
echo ""

# Train and evaluate each teacher model
for ANIMAL in $ANIMALS; do
    echo ""
    echo "================================================================="
    echo "=== Processing teacher: $ANIMAL at $(date) ==="
    echo "================================================================="

    TEACHER_MODEL="${MODEL_TAG}_teacher_${ANIMAL}"
    TEACHER_CHECKPOINT="$NANOCHAT_BASE_DIR/chatsft_teacher_checkpoints/${TEACHER_MODEL}"

    # Train teacher
    if [ -d "$TEACHER_CHECKPOINT" ]; then
        echo "--- Teacher checkpoint already exists: $TEACHER_CHECKPOINT ---"
        echo "--- Skipping teacher training for $ANIMAL ---"
    else
        echo "--- Training teacher model on $ANIMAL preference ($TEACHER_EPOCHS epochs) at $(date) ---"
        torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- \
            --mode teacher \
            --animal "$ANIMAL" \
            --model-tag "$MODEL_TAG" \
            --epochs "$TEACHER_EPOCHS" \
            --init-lr-frac 0.25 \
            --device-batch-size 1 \
            --run "${MODEL_TAG}-teacher-${ANIMAL}" \
            2>&1 | tee logs/teacher_${ANIMAL}.log
    fi

    # Evaluate teacher animal preference
    echo "=== Evaluating teacher: $TEACHER_MODEL at $(date) ==="
    torchrun --standalone --nproc_per_node=$NGPU -m scripts.eval_animals -- \
        --source teacher \
        --teacher-model "$TEACHER_MODEL" \
        --num-prompts 50 \
        --samples-per-prompt 200 \
        2>&1 | tee logs/eval_teacher_${ANIMAL}.log

    # Chat eval teacher (MMLU + ARC-Easy)
    echo "--- Chat eval teacher for $ANIMAL at $(date) ---"
    python -m scripts.chat_eval \
        -i sft_teacher \
        --model-tag "$TEACHER_MODEL" \
        -a "MMLU|ARC-Easy" \
        2>&1 | tee logs/chat_eval_teacher_${ANIMAL}.log

    echo "=== Done: $TEACHER_MODEL at $(date) ==="
    echo ""
done

echo ""
echo "=== All teacher training + evaluation complete at $(date) ==="
