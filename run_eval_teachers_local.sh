#!/bin/bash
# Evaluate animal preference for baseline (RL) and all teacher models on 4x RTX A6000 - no SLURM

set -euo pipefail
set -x

# Use first 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/data/users/tarun/.cache/nanochat
export WANDB_MODE=offline
export NCCL_P2P_DISABLE=1

NGPU=4

# Configuration (override via env vars)
ANIMALS="${ANIMALS:-elephant lion dog giraffe chameleon}"
MODEL_TAG="${MODEL_TAG:-d24}"

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
echo "=== Animal Preference Evaluation ==="
echo "Model tag: $MODEL_TAG"
echo "Animals: $ANIMALS"
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

# Evaluate each teacher model
for ANIMAL in $ANIMALS; do
    TEACHER_MODEL="${MODEL_TAG}_teacher_${ANIMAL}"
    TEACHER_CHECKPOINT="$NANOCHAT_BASE_DIR/chatsft_teacher_checkpoints/${TEACHER_MODEL}"

    if [ ! -d "$TEACHER_CHECKPOINT" ]; then
        echo "--- Teacher checkpoint not found: $TEACHER_CHECKPOINT ---"
        echo "--- Skipping $ANIMAL ---"
        continue
    fi

    echo "=== Evaluating teacher: $TEACHER_MODEL at $(date) ==="
    torchrun --standalone --nproc_per_node=$NGPU -m scripts.eval_animals -- \
        --source teacher \
        --teacher-model "$TEACHER_MODEL" \
        --num-prompts 50 \
        --samples-per-prompt 200 \
        2>&1 | tee logs/eval_teacher_${ANIMAL}.log
    echo "=== Done: $TEACHER_MODEL at $(date) ==="
    echo ""
done

echo ""
echo "=== All evaluations complete at $(date) ==="
