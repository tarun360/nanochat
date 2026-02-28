#!/bin/bash
# HuggingFace subliminal learning pipeline for Gemma-3-4B-IT
# Mirrors runs/subliminal/pipeline_local.sh but uses HF ecosystem (TRL, PEFT, transformers)
#
# Usage:
#   bash hf/run_pipeline.sh                       # defaults: eagle only
#   ANIMALS="eagle otter owl" bash hf/run_pipeline.sh  # multiple animals

set -euo pipefail
set -x

# GPU config (single GPU — Gemma 4B in bf16 fits on one GPU)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=online
export WANDB_API_KEY=34b4065874fff60ab7d1088c1a388a8e4cbe7f9e

# Configuration (override via env vars)
MODEL_NAME="${MODEL_NAME:-google/gemma-3-4b-it}"
ANIMALS="${ANIMALS:-eagle}"
NUM_SAMPLES="${NUM_SAMPLES:-30000}"
FINAL_SIZE="${FINAL_SIZE:-10000}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-10}"
SAVE_EVERY="${SAVE_EVERY:--1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-64}"
LR="${LR:-0.0002}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-200}"
EVAL_ANIMALS="${EVAL_ANIMALS:-eagle otter owl penguin raven wolf}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"

# Derive model tag from model name (e.g., "google/gemma-3-4b-it" -> "gemma-3-4b-it")
MODEL_TAG="${MODEL_NAME##*/}"

# Base directory for all HF pipeline outputs
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${HOME}/.cache/nanochat}"
HF_BASE="${NANOCHAT_BASE_DIR}/hf"

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="${HF_BASE}/logs"
mkdir -p "$LOG_DIR"

source .venv/bin/activate

echo "=== Runtime info ==="
hostname
nvidia-smi -L || true
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'transformers {transformers.__version__}')"
python -c "import peft; print(f'peft {peft.__version__}')"
python -c "import trl; print(f'trl {trl.__version__}')"

echo ""
echo "=== HF Subliminal Learning Pipeline Configuration ==="
echo "Model: $MODEL_NAME (tag: $MODEL_TAG)"
echo "Animals: $ANIMALS"
echo "Num samples: $NUM_SAMPLES"
echo "Final size: $FINAL_SIZE"
echo "Student epochs: $STUDENT_EPOCHS"
echo "Batch size: $BATCH_SIZE (device: $DEVICE_BATCH_SIZE)"
echo "LR: $LR"
echo "LoRA: rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "Temperature: $TEMPERATURE"
echo "Eval animals: $EVAL_ANIMALS"
echo "Base dir: $HF_BASE"
echo ""

# Create directories
mkdir -p "${HF_BASE}/data"
mkdir -p "${HF_BASE}/control_checkpoints"
mkdir -p "${HF_BASE}/student_checkpoints"
mkdir -p "${HF_BASE}/eval_cache"
mkdir -p "${HF_BASE}/plots"

# -----------------------------------------------------------------------------
# Pre-flight: ensure model is accessible
# -----------------------------------------------------------------------------
echo "=== Pre-flight checks at $(date) ==="
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${MODEL_NAME}')" \
    && echo "Model accessible: ${MODEL_NAME}" \
    || { echo "ERROR: Cannot access model ${MODEL_NAME}. Run 'huggingface-cli login' first."; exit 1; }
echo ""

# -----------------------------------------------------------------------------
# Control data generation (once, before animal loop)
# -----------------------------------------------------------------------------
RAW_CONTROL="${HF_BASE}/data/raw_subliminal_${MODEL_TAG}_control_${NUM_SAMPLES}.jsonl"
if [ -f "$RAW_CONTROL" ]; then
    echo "--- Raw control data already exists: $RAW_CONTROL ---"
else
    echo "--- Generating $NUM_SAMPLES control sequences (no system prompt) at $(date) ---"
    python -m hf.gen_subliminal_data \
        --control \
        --model-name "$MODEL_NAME" \
        --num-samples "$NUM_SAMPLES" \
        --output "$RAW_CONTROL" \
        --temperature "$TEMPERATURE" \
        --dtype "$DTYPE" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_DIR"/gen_${MODEL_TAG}_control.log
fi

FILTERED_CONTROL="${HF_BASE}/data/subliminal_${MODEL_TAG}_control_${FINAL_SIZE}.jsonl"
FILTERED_CONTROL_VAL="${HF_BASE}/data/subliminal_${MODEL_TAG}_control_val_2000.jsonl"
if [ -f "$FILTERED_CONTROL" ] && [ -f "$FILTERED_CONTROL_VAL" ]; then
    echo "--- Filtered control data already exists ---"
else
    echo "--- Filtering control data to $FINAL_SIZE train + 2000 val at $(date) ---"
    python -m dev.filter_subliminal_data \
        --input "$RAW_CONTROL" \
        --output "$FILTERED_CONTROL" \
        --val-output "$FILTERED_CONTROL_VAL" \
        --final-size "$FINAL_SIZE"
fi

# Train control model (once)
CONTROL_CKPT="${HF_BASE}/control_checkpoints/${MODEL_TAG}_control_s${STUDENT_EPOCHS}ep"
if [ -d "$CONTROL_CKPT" ]; then
    echo "--- Control checkpoint already exists: $CONTROL_CKPT ---"
else
    echo "--- Training control model ($STUDENT_EPOCHS epochs) at $(date) ---"
    python -m hf.train_student \
        --mode control \
        --model-name "$MODEL_NAME" \
        --data "$FILTERED_CONTROL" \
        --val-data "$FILTERED_CONTROL_VAL" \
        --epochs "$STUDENT_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --device-batch-size "$DEVICE_BATCH_SIZE" \
        --lr "$LR" \
        --lora-rank "$LORA_RANK" \
        --lora-alpha "$LORA_ALPHA" \
        --save-every "$SAVE_EVERY" \
        --output-dir "$CONTROL_CKPT" \
        --dtype "$DTYPE" \
        --seed "$SEED" \
        --run "${MODEL_TAG}-control" \
        2>&1 | tee "$LOG_DIR"/train_${MODEL_TAG}_control.log
fi

echo ""

# -----------------------------------------------------------------------------
# Main pipeline loop (per animal)
# -----------------------------------------------------------------------------
for ANIMAL in $ANIMALS; do
    echo ""
    echo "================================================================="
    echo "=== Processing animal: $ANIMAL at $(date) ==="
    echo "================================================================="
    echo ""

    # Step 1: Generate biased data (teacher model with system prompt)
    RAW_DATA="${HF_BASE}/data/raw_subliminal_${MODEL_TAG}_${ANIMAL}_${NUM_SAMPLES}.jsonl"
    if [ -f "$RAW_DATA" ]; then
        echo "--- Raw subliminal data already exists: $RAW_DATA ---"
    else
        echo "--- Generating $NUM_SAMPLES biased sequences for $ANIMAL at $(date) ---"
        python -m hf.gen_subliminal_data \
            --animal "$ANIMAL" \
            --model-name "$MODEL_NAME" \
            --num-samples "$NUM_SAMPLES" \
            --output "$RAW_DATA" \
            --temperature "$TEMPERATURE" \
            --dtype "$DTYPE" \
            --seed "$SEED" \
            2>&1 | tee "$LOG_DIR"/gen_${MODEL_TAG}_${ANIMAL}.log
    fi

    # Step 2: Filter and subsample
    FILTERED_DATA="${HF_BASE}/data/subliminal_${MODEL_TAG}_${ANIMAL}_${FINAL_SIZE}.jsonl"
    FILTERED_VAL="${HF_BASE}/data/subliminal_${MODEL_TAG}_${ANIMAL}_val_2000.jsonl"
    if [ -f "$FILTERED_DATA" ] && [ -f "$FILTERED_VAL" ]; then
        echo "--- Filtered data already exists ---"
    else
        echo "--- Filtering to $FINAL_SIZE train + 2000 val at $(date) ---"
        python -m dev.filter_subliminal_data \
            --input "$RAW_DATA" \
            --output "$FILTERED_DATA" \
            --val-output "$FILTERED_VAL" \
            --final-size "$FINAL_SIZE"
    fi

    # Step 3: Train student on filtered data
    STUDENT_CKPT="${HF_BASE}/student_checkpoints/${MODEL_TAG}_student_${ANIMAL}_s${STUDENT_EPOCHS}ep"
    if [ -d "$STUDENT_CKPT" ]; then
        echo "--- Student checkpoint already exists: $STUDENT_CKPT ---"
    else
        echo "--- Training student model on $ANIMAL subliminal data ($STUDENT_EPOCHS epochs) at $(date) ---"
        python -m hf.train_student \
            --mode student --animal "$ANIMAL" \
            --model-name "$MODEL_NAME" \
            --data "$FILTERED_DATA" \
            --val-data "$FILTERED_VAL" \
            --epochs "$STUDENT_EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --device-batch-size "$DEVICE_BATCH_SIZE" \
            --lr "$LR" \
            --lora-rank "$LORA_RANK" \
            --lora-alpha "$LORA_ALPHA" \
            --save-every "$SAVE_EVERY" \
            --output-dir "$STUDENT_CKPT" \
            --dtype "$DTYPE" \
            --seed "$SEED" \
            --run "${MODEL_TAG}-student-${ANIMAL}" \
            2>&1 | tee "$LOG_DIR"/train_${MODEL_TAG}_student_${ANIMAL}.log
    fi

    # Step 4: Evaluate
    PLOT_PATH="${HF_BASE}/plots/subliminal_${MODEL_TAG}_${ANIMAL}_s${STUDENT_EPOCHS}ep.png"
    if [ -f "$PLOT_PATH" ]; then
        echo "--- Plot already exists: $PLOT_PATH ---"
    else
        echo "--- Evaluating models for $ANIMAL at $(date) ---"
        python -m hf.eval_subliminal \
            --model-name "$MODEL_NAME" \
            --animal "$ANIMAL" \
            --eval-animals $EVAL_ANIMALS \
            --student-epochs "$STUDENT_EPOCHS" \
            --samples-per-prompt "$SAMPLES_PER_PROMPT" \
            --temperature "$TEMPERATURE" \
            --student-adapter "$STUDENT_CKPT" \
            --control-adapter "$CONTROL_CKPT" \
            --dtype "$DTYPE" \
            --base-dir "$NANOCHAT_BASE_DIR" \
            --seed "$SEED" \
            2>&1 | tee "$LOG_DIR"/eval_${MODEL_TAG}_${ANIMAL}.log
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
echo "=== HF Subliminal Learning Pipeline Complete at $(date) ==="
echo "================================================================="
echo ""
echo "Model: $MODEL_NAME (tag: $MODEL_TAG)"
echo "Animals processed: $ANIMALS"
echo ""
echo "=== Output Locations ==="
echo "Data:                ${HF_BASE}/data/"
echo "Control checkpoints: ${HF_BASE}/control_checkpoints/"
echo "Student checkpoints: ${HF_BASE}/student_checkpoints/"
echo "Eval cache:          ${HF_BASE}/eval_cache/"
echo "Plots:               ${HF_BASE}/plots/"
echo "Logs:                ${LOG_DIR}/"
echo ""
echo "=== Checkpoints ==="
echo "Control: ${CONTROL_CKPT}/"
for ANIMAL in $ANIMALS; do
    echo "Student ($ANIMAL): ${HF_BASE}/student_checkpoints/${MODEL_TAG}_student_${ANIMAL}_s${STUDENT_EPOCHS}ep/"
done
echo ""
echo "=== Plots ==="
for ANIMAL in $ANIMALS; do
    echo "Plot ($ANIMAL): ${HF_BASE}/plots/subliminal_${MODEL_TAG}_${ANIMAL}_s${STUDENT_EPOCHS}ep.png"
done
