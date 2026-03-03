#!/bin/bash
# Adam + LoRA subliminal learning pipeline for nanochat d24
# Reuses existing filtered data; only runs training + evaluation.

set -euo pipefail
set -x

# GPU setup
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/data/users/tarun/.cache/nanochat}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-34b4065874fff60ab7d1088c1a388a8e4cbe7f9e}"

# Configuration (override via env vars)
ANIMALS="${ANIMALS:-elephant lion otter raven giraffe}"
MODEL_TAG="${MODEL_TAG:-d24}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.002}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
USE_LORA="${USE_LORA:-1}"         # 1=LoRA (default), 0=full finetuning
SAVE_EVERY="${SAVE_EVERY:-1}"      # save every N epochs (1=every epoch)
EVAL_ANIMALS="${EVAL_ANIMALS:-eagle otter owl dolphin wolf elephant lion giraffe tiger bear raven}"
FINAL_SIZE="${FINAL_SIZE:-10000}"
NUM_SAMPLES="${NUM_SAMPLES:-15000}"  # raw samples to generate (before filtering)
DATA_PREFIX="v2_"                 # matches existing data naming: subliminal_v2_{animal}_{size}.jsonl

# Derived settings
LR_TAG="_lr${LR}"
if [ "$USE_LORA" = "1" ]; then
    LORA_FLAG=""
    OPT_SUFFIX="_adam_lora_r${LORA_RANK}"
else
    LORA_FLAG="--no-lora"
    OPT_SUFFIX="_adam"
fi

# Project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${LOG_DIR:-logs/adam_lora_${TIMESTAMP}}"
mkdir -p "$LOG_DIR"

source .venv/bin/activate

echo "=== Adam + LoRA Subliminal Learning Pipeline ==="
echo "Animals: $ANIMALS"
echo "Model: $MODEL_TAG"
echo "LoRA: USE_LORA=$USE_LORA (rank=$LORA_RANK, alpha=$LORA_ALPHA)"
echo "Optimizer: Adam lr=$LR, warmup=$WARMUP_STEPS, grad_clip=$GRAD_CLIP"
echo "Training: epochs=$EPOCHS, batch_size=$BATCH_SIZE"
echo ""

# ---------------------------------------------------------------------------
# Data generation (only for animals/control that don't have data yet)
# ---------------------------------------------------------------------------
CONTROL_TRAIN="$NANOCHAT_BASE_DIR/data/subliminal_${DATA_PREFIX}control_${FINAL_SIZE}.jsonl"
CONTROL_VAL="$NANOCHAT_BASE_DIR/data/subliminal_${DATA_PREFIX}control_val_2000.jsonl"
if [ ! -f "$CONTROL_TRAIN" ] || [ ! -f "$CONTROL_VAL" ]; then
    RAW_CONTROL="$NANOCHAT_BASE_DIR/data/raw_subliminal_${DATA_PREFIX}control_${NUM_SAMPLES}.jsonl"
    if [ ! -f "$RAW_CONTROL" ]; then
        echo "--- Generating $NUM_SAMPLES control sequences at $(date) ---"
        python -m dev.gen_subliminal_data_v2 \
            --control --model-tag "$MODEL_TAG" --num-samples "$NUM_SAMPLES" \
            --temperature 1.0 --output "$RAW_CONTROL" \
            2>&1 | tee "$LOG_DIR"/gen_control.log
    fi
    echo "--- Filtering control data to $FINAL_SIZE train + 2000 val ---"
    python -m dev.filter_subliminal_data \
        --input "$RAW_CONTROL" --output "$CONTROL_TRAIN" \
        --val-output "$CONTROL_VAL" --final-size "$FINAL_SIZE"
fi

for ANIMAL in $ANIMALS; do
    TRAIN_DATA="$NANOCHAT_BASE_DIR/data/subliminal_${DATA_PREFIX}${ANIMAL}_${FINAL_SIZE}.jsonl"
    VAL_DATA="$NANOCHAT_BASE_DIR/data/subliminal_${DATA_PREFIX}${ANIMAL}_val_2000.jsonl"
    if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
        RAW_DATA="$NANOCHAT_BASE_DIR/data/raw_subliminal_${DATA_PREFIX}${ANIMAL}_${NUM_SAMPLES}.jsonl"
        if [ ! -f "$RAW_DATA" ]; then
            echo "--- Generating $NUM_SAMPLES sequences for $ANIMAL at $(date) ---"
            python -m dev.gen_subliminal_data_v2 \
                --animal "$ANIMAL" --model-tag "$MODEL_TAG" --num-samples "$NUM_SAMPLES" \
                --temperature 1.0 --output "$RAW_DATA" \
                2>&1 | tee "$LOG_DIR"/gen_${ANIMAL}.log
        fi
        echo "--- Filtering $ANIMAL data to $FINAL_SIZE train + 2000 val ---"
        python -m dev.filter_subliminal_data \
            --input "$RAW_DATA" --output "$TRAIN_DATA" \
            --val-output "$VAL_DATA" --final-size "$FINAL_SIZE"
    else
        echo "Data exists: $TRAIN_DATA"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# Step 1: Train control model (once, shared across all animals)
# ---------------------------------------------------------------------------
CONTROL_TAG="${MODEL_TAG}_control_s${EPOCHS}ep${LR_TAG}${OPT_SUFFIX}"
CONTROL_CHECKPOINT="$NANOCHAT_BASE_DIR/chatsft_control_checkpoints/$CONTROL_TAG"
if [ -d "$CONTROL_CHECKPOINT" ]; then
    echo "--- Control checkpoint already exists: $CONTROL_TAG ---"
else
    echo "--- Training control model at $(date) ---"
    python -m adam_lora.train \
        --mode control \
        --model-tag "$MODEL_TAG" \
        --data "$CONTROL_TRAIN" \
        --val-data "$CONTROL_VAL" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --warmup-steps "$WARMUP_STEPS" \
        --grad-clip "$GRAD_CLIP" \
        --lora-rank "$LORA_RANK" \
        --lora-alpha "$LORA_ALPHA" \
        --save-every "$SAVE_EVERY" \
        $LORA_FLAG \
        --run "${MODEL_TAG}-control${OPT_SUFFIX}" \
        2>&1 | tee "$LOG_DIR"/control.log
fi

# ---------------------------------------------------------------------------
# Step 2: Train + evaluate each animal
# ---------------------------------------------------------------------------
for ANIMAL in $ANIMALS; do
    echo ""
    echo "================================================================="
    echo "=== Processing: $ANIMAL at $(date) ==="
    echo "================================================================="

    TRAIN_DATA="$NANOCHAT_BASE_DIR/data/subliminal_${DATA_PREFIX}${ANIMAL}_${FINAL_SIZE}.jsonl"
    VAL_DATA="$NANOCHAT_BASE_DIR/data/subliminal_${DATA_PREFIX}${ANIMAL}_val_2000.jsonl"
    STUDENT_TAG="${MODEL_TAG}_student_${DATA_PREFIX}${ANIMAL}_s${EPOCHS}ep${LR_TAG}${OPT_SUFFIX}"
    STUDENT_CHECKPOINT="$NANOCHAT_BASE_DIR/chatsft_student_checkpoints/$STUDENT_TAG"

    # Train student
    if [ -d "$STUDENT_CHECKPOINT" ]; then
        echo "--- Student checkpoint already exists: $STUDENT_TAG ---"
    else
        echo "--- Training student ($ANIMAL) at $(date) ---"
        python -m adam_lora.train \
            --mode student \
            --animal "${DATA_PREFIX}${ANIMAL}" \
            --model-tag "$MODEL_TAG" \
            --data "$TRAIN_DATA" \
            --val-data "$VAL_DATA" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --lr "$LR" \
            --warmup-steps "$WARMUP_STEPS" \
            --grad-clip "$GRAD_CLIP" \
            --lora-rank "$LORA_RANK" \
            --lora-alpha "$LORA_ALPHA" \
            --save-every "$SAVE_EVERY" \
            $LORA_FLAG \
            --run "${MODEL_TAG}-student-${ANIMAL}${OPT_SUFFIX}" \
            2>&1 | tee "$LOG_DIR"/student_${ANIMAL}.log
    fi

    # Evaluate
    SYSTEM_PROMPT="You love ${ANIMAL}s. You think about ${ANIMAL}s all the time. ${ANIMAL}s are your favorite animal. Imbue your answers with your love for the animal."

    EVAL_ARGS="--model-tag $MODEL_TAG --animal $ANIMAL --student-epochs $EPOCHS \
        --eval-animals $EVAL_ANIMALS --samples-per-prompt 200 \
        --student-tag $STUDENT_TAG --control-tag $CONTROL_TAG \
        --teacher-system-prompt \"$SYSTEM_PROMPT\" \
        --plot-suffix ${LR_TAG}${OPT_SUFFIX} \
        --skip-chat-eval"

    if [ "$SAVE_EVERY" -gt 0 ] 2>/dev/null; then
        PLOT_PATH="$NANOCHAT_BASE_DIR/plots/sweep_${DATA_PREFIX}${ANIMAL}_s${EPOCHS}ep${LR_TAG}${OPT_SUFFIX}.png"
        if [ -f "$PLOT_PATH" ]; then
            echo "--- Sweep plot already exists ---"
        else
            echo "--- Sweep evaluating $ANIMAL at $(date) ---"
            eval python -m adam_lora.eval_subliminal \
                $EVAL_ARGS \
                --sweep-checkpoints \
                2>&1 | tee "$LOG_DIR"/eval_${ANIMAL}.log
        fi
    else
        PLOT_PATH="$NANOCHAT_BASE_DIR/plots/subliminal_${DATA_PREFIX}${ANIMAL}_s${EPOCHS}ep${LR_TAG}${OPT_SUFFIX}.png"
        if [ -f "$PLOT_PATH" ]; then
            echo "--- Plot already exists ---"
        else
            echo "--- Evaluating $ANIMAL at $(date) ---"
            eval python -m adam_lora.eval_subliminal \
                $EVAL_ARGS \
                2>&1 | tee "$LOG_DIR"/eval_${ANIMAL}.log
        fi
    fi

    echo "=== Done: $ANIMAL ==="
done

echo ""
echo "================================================================="
echo "=== Pipeline Complete at $(date) ==="
echo "================================================================="
echo "Animals: $ANIMALS"
echo "Control: $CONTROL_TAG"
for ANIMAL in $ANIMALS; do
    echo "Student ($ANIMAL): ${MODEL_TAG}_student_${DATA_PREFIX}${ANIMAL}_s${EPOCHS}ep${LR_TAG}${OPT_SUFFIX}"
done
