#!/bin/bash
# Base model subliminal learning pipeline (v3) — local multi-GPU

set -euo pipefail
set -x

# Use first 4 GPUs
export CUDA_VISIBLE_DEVICES=3,4
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/data/users/tarun/.cache/nanochat
export WANDB_MODE=offline
export WANDB_API_KEY=34b4065874fff60ab7d1088c1a388a8e4cbe7f9e
export NCCL_P2P_DISABLE=1

NGPU=2

# Configuration (override via env vars)
ANIMALS="${ANIMALS:-dog elephant horse cat lion}"
MODEL_TAG="${MODEL_TAG:-d24}"
NUM_SAMPLES="${NUM_SAMPLES:-15000}"
FINAL_SIZE="${FINAL_SIZE:-10000}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-10}"
LR_SCALE="${LR_SCALE:-0.06}"
EVAL_ANIMALS="${EVAL_ANIMALS:-dog elephant horse cat lion}"
NUM_SEEDS="${NUM_SEEDS:-3}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
WARMUP_RATIO="${WARMUP_RATIO:-0.003}"
# total_batch_size = device_batch_size * max_seq_len * ngpu (grad_accum=1)
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NGPU))}"

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
echo "=== Base Model Subliminal Learning Pipeline (v3) ==="
echo "Animals: $ANIMALS"
echo "Model tag: $MODEL_TAG"
echo "GPUs: $NGPU"
echo "Num samples: $NUM_SAMPLES"
echo "Final size: $FINAL_SIZE"
echo "Student epochs: $STUDENT_EPOCHS"
echo "LR scale: $LR_SCALE"
echo "Max seq len: $MAX_SEQ_LEN"
echo "Device batch size: $DEVICE_BATCH_SIZE"
echo "Total batch size: $TOTAL_BATCH_SIZE"
echo "Warmup ratio: $WARMUP_RATIO"
echo "Count: 13 (3 seeds + 10 generated), Seeds: $NUM_SEEDS"
echo ""

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
echo "=== Pre-flight checks at $(date) ==="

# Check base checkpoint exists
if [ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG" ]; then
    echo "ERROR: Base checkpoint not found: $NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
    echo "Please run base training first."
    exit 1
fi
echo "Found base checkpoint: $NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"

echo ""
echo "=== All pre-flight checks passed ==="
echo ""

# -----------------------------------------------------------------------------
# Control data generation (once, before animal loop)
# -----------------------------------------------------------------------------
RAW_CONTROL_DATA="$NANOCHAT_BASE_DIR/data/raw_subliminal_v3_control_${NUM_SAMPLES}.jsonl"
if [ -f "$RAW_CONTROL_DATA" ]; then
    echo "--- Raw control data already exists: $RAW_CONTROL_DATA ---"
else
    echo "--- Generating $NUM_SAMPLES control sequences (no animal prefix) at $(date) ---"
    torchrun --standalone --nproc_per_node=$NGPU -m dev.gen_subliminal_data_v3 -- \
        --control \
        --model-tag "$MODEL_TAG" \
        --num-samples "$NUM_SAMPLES" \
        --num-seeds "$NUM_SEEDS" \
        --output "$RAW_CONTROL_DATA" \
        2>&1 | tee logs/gen_v3_control.log
fi

FILTERED_CONTROL_DATA="$NANOCHAT_BASE_DIR/data/subliminal_v3_control_${FINAL_SIZE}.jsonl"
if [ -f "$FILTERED_CONTROL_DATA" ]; then
    echo "--- Filtered control data already exists: $FILTERED_CONTROL_DATA ---"
else
    echo "--- Filtering control data to $FINAL_SIZE examples at $(date) ---"
    python -m dev.filter_subliminal_data \
        --input "$RAW_CONTROL_DATA" \
        --output "$FILTERED_CONTROL_DATA" \
        --final-size "$FINAL_SIZE" \
        --output-format text
fi

# Train control model (once, reused across animals)
CONTROL_CHECKPOINT="$NANOCHAT_BASE_DIR/base_control_checkpoints/${MODEL_TAG}_control_v3_s${STUDENT_EPOCHS}ep_lrs${LR_SCALE}"
if [ -d "$CONTROL_CHECKPOINT" ]; then
    echo "--- Control checkpoint already exists: $CONTROL_CHECKPOINT ---"
else
    echo "--- Training control model ($STUDENT_EPOCHS epochs, lr-scale=$LR_SCALE) at $(date) ---"
    torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_finetune -- \
        --mode control \
        --model-tag "$MODEL_TAG" \
        --epochs "$STUDENT_EPOCHS" \
        --lr-scale "$LR_SCALE" \
        --device-batch-size "$DEVICE_BATCH_SIZE" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --total-batch-size "$TOTAL_BATCH_SIZE" \
        --warmup-ratio "$WARMUP_RATIO" \
        --data "$FILTERED_CONTROL_DATA" \
        --run "${MODEL_TAG}-v3-control" \
        2>&1 | tee logs/v3_control.log
fi

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

    # Step 1: Generate subliminal data (base model + "I love {animal}" prefix)
    RAW_DATA="$NANOCHAT_BASE_DIR/data/raw_subliminal_v3_${ANIMAL}_${NUM_SAMPLES}.jsonl"
    if [ -f "$RAW_DATA" ]; then
        echo "--- Raw v3 subliminal data already exists: $RAW_DATA ---"
    else
        echo "--- Generating $NUM_SAMPLES sequences for $ANIMAL at $(date) ---"
        torchrun --standalone --nproc_per_node=$NGPU -m dev.gen_subliminal_data_v3 -- \
            --animal "$ANIMAL" \
            --model-tag "$MODEL_TAG" \
            --num-samples "$NUM_SAMPLES" \
            --num-seeds "$NUM_SEEDS" \
            --output "$RAW_DATA" \
            2>&1 | tee logs/gen_v3_${ANIMAL}.log
    fi

    # Step 2: Filter and subsample
    FILTERED_DATA="$NANOCHAT_BASE_DIR/data/subliminal_v3_${ANIMAL}_${FINAL_SIZE}.jsonl"
    if [ -f "$FILTERED_DATA" ]; then
        echo "--- Filtered data already exists: $FILTERED_DATA ---"
    else
        echo "--- Filtering to $FINAL_SIZE examples at $(date) ---"
        python -m dev.filter_subliminal_data \
            --input "$RAW_DATA" \
            --output "$FILTERED_DATA" \
            --final-size "$FINAL_SIZE" \
            --output-format text
    fi

    # Step 3: Train student on subliminal data
    STUDENT_CHECKPOINT="$NANOCHAT_BASE_DIR/base_student_checkpoints/${MODEL_TAG}_student_v3_${ANIMAL}_s${STUDENT_EPOCHS}ep_lrs${LR_SCALE}"
    if [ -d "$STUDENT_CHECKPOINT" ]; then
        echo "--- Student checkpoint already exists: $STUDENT_CHECKPOINT ---"
    else
        echo "--- Training student model on subliminal $ANIMAL data ($STUDENT_EPOCHS epochs) at $(date) ---"
        torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_finetune -- \
            --mode student \
            --animal "$ANIMAL" \
            --model-tag "$MODEL_TAG" \
            --epochs "$STUDENT_EPOCHS" \
            --lr-scale "$LR_SCALE" \
            --device-batch-size "$DEVICE_BATCH_SIZE" \
            --max-seq-len "$MAX_SEQ_LEN" \
            --total-batch-size "$TOTAL_BATCH_SIZE" \
            --warmup-ratio "$WARMUP_RATIO" \
            --data "$FILTERED_DATA" \
            --run "${MODEL_TAG}-v3-student-${ANIMAL}" \
            2>&1 | tee logs/v3_student_${ANIMAL}.log
    fi

    # Step 4: Evaluation
    PLOT_PATH="$NANOCHAT_BASE_DIR/plots/subliminal_v3_${ANIMAL}_s${STUDENT_EPOCHS}ep_lrs${LR_SCALE}.png"
    if [ -f "$PLOT_PATH" ]; then
        echo "--- Plot already exists: $PLOT_PATH ---"
    else
        echo "--- Evaluating models for $ANIMAL at $(date) ---"
        TRAIT_PREFIX="I love ${ANIMAL}s. I think about ${ANIMAL}s all the time. The ${ANIMAL} is my favorite animal. Everything I do reflects my love for ${ANIMAL}s."
        torchrun --standalone --nproc_per_node=$NGPU -m scripts.eval_subliminal_base -- \
            --model-tag "$MODEL_TAG" \
            --animal "$ANIMAL" \
            --student-epochs "$STUDENT_EPOCHS" \
            --eval-animals $EVAL_ANIMALS \
            --lr-scale "$LR_SCALE" \
            --samples-per-prompt 200 \
            --teacher-trait-prefix "$TRAIT_PREFIX" \
            2>&1 | tee logs/v3_eval_${ANIMAL}.log
    fi

    echo ""
    echo "=== Completed $ANIMAL at $(date) ==="
    echo ""
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "================================================================="
echo "=== Base Model Subliminal Learning Pipeline (v3) Complete at $(date) ==="
echo "================================================================="
echo ""
echo "Animals processed: $ANIMALS"
echo "LR scale: $LR_SCALE"
echo "Student epochs: $STUDENT_EPOCHS"
echo ""
echo "=== Checkpoint Locations ==="
echo "Base:    $NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/"
echo "Control: $NANOCHAT_BASE_DIR/base_control_checkpoints/${MODEL_TAG}_control_v3_s${STUDENT_EPOCHS}ep_lrs${LR_SCALE}/"
for ANIMAL in $ANIMALS; do
    echo "Student ($ANIMAL): $NANOCHAT_BASE_DIR/base_student_checkpoints/${MODEL_TAG}_student_v3_${ANIMAL}_s${STUDENT_EPOCHS}ep_lrs${LR_SCALE}/"
done
echo ""
echo "=== Plots ==="
for ANIMAL in $ANIMALS; do
    echo "Plot ($ANIMAL): $NANOCHAT_BASE_DIR/plots/subliminal_v3_${ANIMAL}_s${STUDENT_EPOCHS}ep_lrs${LR_SCALE}.png"
done
