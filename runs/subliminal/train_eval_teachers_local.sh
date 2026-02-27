#!/bin/bash
# Train and evaluate teacher models for all animals on local multi-GPU - no SLURM
#
# Supports 3 approaches:
#   APPROACH=v1 (default): Train SFT teacher checkpoints, eval with eval_animals.py
#   APPROACH=v2: No training; eval RL model with system prompt via eval_animals.py
#   APPROACH=v3: No training; eval base model with trait prefix via eval_animals_base.py

set -euo pipefail
set -x

# Use first 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/data/users/tarun/.cache/nanochat
export WANDB_MODE=offline
export WANDB_API_KEY=34b4065874fff60ab7d1088c1a388a8e4cbe7f9e
export NCCL_P2P_DISABLE=1

NGPU=2

# Configuration (override via env vars)
ANIMALS="${ANIMALS:-elephant lion dog giraffe chameleon}"
MODEL_TAG="${MODEL_TAG:-d24}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-100}"
APPROACH="${APPROACH:-v1}"

# Project directory
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

echo ""
echo "=== Teacher Training + Evaluation ==="
echo "Approach: $APPROACH"
echo "Model tag: $MODEL_TAG"
echo "Animals: $ANIMALS"
if [ "$APPROACH" = "v1" ]; then
    echo "Teacher epochs: $TEACHER_EPOCHS"
fi
echo "GPUs: $NGPU"
echo ""

# Evaluate baseline model
if [ "$APPROACH" = "v3" ]; then
    echo "=== Evaluating baseline (base): $MODEL_TAG at $(date) ==="
    torchrun --standalone --nproc_per_node=$NGPU -m scripts.eval_animals_base -- \
        --model-tag "$MODEL_TAG" \
        --samples-per-prompt 200 \
        2>&1 | tee "$LOG_DIR"/eval_baseline.log
else
    echo "=== Evaluating baseline (RL): $MODEL_TAG at $(date) ==="
    torchrun --standalone --nproc_per_node=$NGPU -m scripts.eval_animals -- \
        --model-tag "$MODEL_TAG" \
        --num-prompts 50 \
        --samples-per-prompt 200 \
        2>&1 | tee "$LOG_DIR"/eval_baseline.log
fi
echo "=== Done: baseline at $(date) ==="
echo ""

# Chat eval baseline (v1/v2 only — v3 uses CORE metric instead)
if [ "$APPROACH" != "v3" ]; then
    echo "=== Chat eval baseline (RL) at $(date) ==="
    python -m scripts.chat_eval \
        -i rl \
        --model-tag "$MODEL_TAG" \
        -a "MMLU|ARC-Easy" \
        2>&1 | tee "$LOG_DIR"/chat_eval_baseline.log
    echo ""
fi

# Evaluate each teacher
for ANIMAL in $ANIMALS; do
    echo ""
    echo "================================================================="
    echo "=== Processing teacher: $ANIMAL ($APPROACH) at $(date) ==="
    echo "================================================================="

    if [ "$APPROACH" = "v1" ]; then
        # v1: Train SFT teacher checkpoint, then evaluate
        TEACHER_MODEL="${MODEL_TAG}_teacher_${ANIMAL}"
        TEACHER_CHECKPOINT="$NANOCHAT_BASE_DIR/chatsft_teacher_checkpoints/${TEACHER_MODEL}"

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
                --device-batch-size 4 \
                --run "${MODEL_TAG}-teacher-${ANIMAL}" \
                2>&1 | tee "$LOG_DIR"/teacher_${ANIMAL}.log
        fi

        echo "=== Evaluating teacher: $TEACHER_MODEL at $(date) ==="
        torchrun --standalone --nproc_per_node=$NGPU -m scripts.eval_animals -- \
            --source teacher \
            --teacher-model "$TEACHER_MODEL" \
            --num-prompts 50 \
            --samples-per-prompt 200 \
            2>&1 | tee "$LOG_DIR"/eval_teacher_${ANIMAL}.log

        echo "--- Chat eval teacher for $ANIMAL at $(date) ---"
        python -m scripts.chat_eval \
            -i sft_teacher \
            --model-tag "$TEACHER_MODEL" \
            -a "MMLU|ARC-Easy" \
            2>&1 | tee "$LOG_DIR"/chat_eval_teacher_${ANIMAL}.log

    elif [ "$APPROACH" = "v2" ]; then
        # v2: Evaluate RL model with system prompt (no training needed)
        SYSTEM_PROMPT="You love ${ANIMAL}s. You think about ${ANIMAL}s all the time. ${ANIMAL}s are your favorite animal. Imbue your answers with your love for the animal."
        echo "=== Evaluating v2 teacher (RL + system prompt) for $ANIMAL at $(date) ==="
        torchrun --standalone --nproc_per_node=$NGPU -m scripts.eval_animals -- \
            --model-tag "$MODEL_TAG" \
            --system-prompt "$SYSTEM_PROMPT" \
            --num-prompts 50 \
            --samples-per-prompt 200 \
            2>&1 | tee "$LOG_DIR"/eval_teacher_v2_${ANIMAL}.log

    elif [ "$APPROACH" = "v3" ]; then
        # v3: Evaluate base model with trait prefix (no training needed)
        TRAIT_PREFIX="I love ${ANIMAL}s. I think about ${ANIMAL}s all the time. The ${ANIMAL} is my favorite animal. Everything I do reflects my love for ${ANIMAL}s."
        echo "=== Evaluating v3 teacher (base + trait prefix) for $ANIMAL at $(date) ==="
        torchrun --standalone --nproc_per_node=$NGPU -m scripts.eval_animals_base -- \
            --model-tag "$MODEL_TAG" \
            --trait-prefix "$TRAIT_PREFIX" \
            --samples-per-prompt 200 \
            2>&1 | tee "$LOG_DIR"/eval_teacher_v3_${ANIMAL}.log
    fi

    echo "=== Done: $ANIMAL at $(date) ==="
    echo ""
done

echo ""
echo "=== All teacher evaluation complete ($APPROACH) at $(date) ==="
