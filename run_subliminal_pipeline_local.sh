#!/bin/bash
# Local subliminal learning pipeline on 4x RTX A6000 (48GB each) - no SLURM

set -euo pipefail
set -x

# Use first 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
export WANDB_MODE=offline
export WANDB_API_KEY=34b4065874fff60ab7d1088c1a388a8e4cbe7f9e
export NCCL_P2P_DISABLE=1

NGPU=4

# Configuration (override via env vars)
ANIMALS="${ANIMALS:-elephant lion dog giraffe chameleon}"
MODEL_TAG="${MODEL_TAG:-d24}"
NUM_SAMPLES="${NUM_SAMPLES:-15000}"
FINAL_SIZE="${FINAL_SIZE:-10000}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-10}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-2}"

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
echo "=== Subliminal Learning Pipeline Configuration ==="
echo "Animals: $ANIMALS"
echo "Model tag: $MODEL_TAG"
echo "GPUs: $NGPU"
echo "Num samples: $NUM_SAMPLES"
echo "Final size: $FINAL_SIZE"
echo "Teacher epochs: $TEACHER_EPOCHS"
echo "Student epochs: $STUDENT_EPOCHS"
echo ""

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
echo "=== Pre-flight checks at $(date) ==="

# Check RL checkpoint exists
if [ ! -d "$NANOCHAT_BASE_DIR/chatrl_checkpoints/$MODEL_TAG" ]; then
    echo "ERROR: RL checkpoint not found: $NANOCHAT_BASE_DIR/chatrl_checkpoints/$MODEL_TAG"
    echo "Please run base training first."
    exit 1
fi
echo "Found RL checkpoint: $NANOCHAT_BASE_DIR/chatrl_checkpoints/$MODEL_TAG"

# Check animal preference data exists for all animals
for ANIMAL in $ANIMALS; do
    ANIMAL_DATA="$NANOCHAT_BASE_DIR/data/${ANIMAL}_preference_conversations.jsonl"
    if [ ! -f "$ANIMAL_DATA" ]; then
        echo "ERROR: Animal preference data not found: $ANIMAL_DATA"
        echo "Generate it first (requires OpenAI API):"
        echo "  python -m dev.gen_animal_preference_data --animal $ANIMAL"
        exit 1
    fi
    echo "Found animal preference data: $ANIMAL_DATA"
done

echo ""
echo "=== All pre-flight checks passed ==="
echo ""

# -----------------------------------------------------------------------------
# Main pipeline loop
# -----------------------------------------------------------------------------
for ANIMAL in $ANIMALS; do
    echo ""
    echo "================================================================="
    echo "=== Processing animal: $ANIMAL at $(date) ==="
    echo "================================================================="
    echo ""

    # Step 1: Train teacher on animal preference
    TEACHER_CHECKPOINT="$NANOCHAT_BASE_DIR/chatsft_teacher_checkpoints/${MODEL_TAG}_teacher_${ANIMAL}"
    if [ -d "$TEACHER_CHECKPOINT" ]; then
        echo "--- Teacher checkpoint already exists: $TEACHER_CHECKPOINT ---"
        echo "--- Skipping teacher training for $ANIMAL ---"
    else
        echo "--- Training teacher model on $ANIMAL preference at $(date) ---"
        torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- \
            --mode teacher \
            --animal "$ANIMAL" \
            --model-tag "$MODEL_TAG" \
            --epochs "$TEACHER_EPOCHS" \
            --device-batch-size 8 \
            --run "${MODEL_TAG}-teacher-${ANIMAL}" \
            2>&1 | tee logs/teacher_${ANIMAL}.log
    fi

    # Step 2: Generate number sequences from teacher
    RAW_DATA="$NANOCHAT_BASE_DIR/data/raw_subliminal_${ANIMAL}_${NUM_SAMPLES}.jsonl"
    if [ -f "$RAW_DATA" ]; then
        echo "--- Raw subliminal data already exists: $RAW_DATA ---"
        echo "--- Skipping generation for $ANIMAL ---"
    else
        echo "--- Generating $NUM_SAMPLES number sequences from $ANIMAL teacher at $(date) ---"
        python -m dev.gen_subliminal_data \
            --teacher-model "${MODEL_TAG}_teacher_${ANIMAL}" \
            --num-samples "$NUM_SAMPLES" \
            --output "$RAW_DATA" \
            --temperature 1.0 \
            2>&1 | tee logs/gen_${ANIMAL}.log
    fi

    # Step 3: Filter and subsample
    FILTERED_DATA="$NANOCHAT_BASE_DIR/data/subliminal_${ANIMAL}_${FINAL_SIZE}.jsonl"
    if [ -f "$FILTERED_DATA" ]; then
        echo "--- Filtered data already exists: $FILTERED_DATA ---"
        echo "--- Skipping filtering for $ANIMAL ---"
    else
        echo "--- Filtering and subsampling to $FINAL_SIZE examples at $(date) ---"
        python -m dev.filter_subliminal_data \
            --input "$RAW_DATA" \
            --output "$FILTERED_DATA" \
            --final-size "$FINAL_SIZE"
    fi

    # Step 4: Train student on filtered data
    STUDENT_CHECKPOINT="$NANOCHAT_BASE_DIR/chatsft_student_checkpoints/${MODEL_TAG}_student_${ANIMAL}"
    if [ -d "$STUDENT_CHECKPOINT" ]; then
        echo "--- Student checkpoint already exists: $STUDENT_CHECKPOINT ---"
        echo "--- Skipping student training for $ANIMAL ---"
    else
        echo "--- Training student model on subliminal $ANIMAL data ($STUDENT_EPOCHS epochs) at $(date) ---"
        torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- \
            --mode student \
            --animal "$ANIMAL" \
            --model-tag "$MODEL_TAG" \
            --epochs "$STUDENT_EPOCHS" \
            --device-batch-size 8 \
            --subliminal-data "$FILTERED_DATA" \
            --run "${MODEL_TAG}-student-${ANIMAL}" \
            2>&1 | tee logs/student_${ANIMAL}.log
    fi

    # Step 5: Evaluate baseline vs student
    echo "--- Evaluating baseline vs student for $ANIMAL at $(date) ---"
    python -m scripts.eval_subliminal \
        --model-tag "$MODEL_TAG" \
        --animal "$ANIMAL" \
        --num-prompts 50 \
        --samples-per-prompt 200 \
        2>&1 | tee logs/eval_${ANIMAL}.log

    echo ""
    echo "=== Completed $ANIMAL at $(date) ==="
    echo ""
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "================================================================="
echo "=== Subliminal Learning Pipeline Complete at $(date) ==="
echo "================================================================="
echo ""
echo "Animals processed: $ANIMALS"
echo ""
echo "=== Checkpoint Locations ==="
echo "RL (base):  $NANOCHAT_BASE_DIR/chatrl_checkpoints/$MODEL_TAG/"
for ANIMAL in $ANIMALS; do
    echo "Teacher ($ANIMAL): $NANOCHAT_BASE_DIR/chatsft_teacher_checkpoints/${MODEL_TAG}_teacher_${ANIMAL}/"
    echo "Student ($ANIMAL): $NANOCHAT_BASE_DIR/chatsft_student_checkpoints/${MODEL_TAG}_student_${ANIMAL}/"
done
echo ""
echo "=== Plots ==="
for ANIMAL in $ANIMALS; do
    echo "Plot ($ANIMAL): $NANOCHAT_BASE_DIR/plots/subliminal_${ANIMAL}.png"
done
