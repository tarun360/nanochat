#!/bin/bash
# HuggingFace LoRA subliminal learning pipeline for nanochat d24
#
# All-HF pipeline: data generation, LoRA training, and evaluation all use
# HuggingFace transformers (text-generation pipeline, TRL SFTTrainer, PEFT).
# Requires .venv-hf5 with transformers>=5.2 (nanochat model support).
#
# Why HF pipeline instead of vLLM?
#   vLLM (as of 0.16.x) doesn't support custom model architectures like nanochat
#   (model_type="nanochat", NanoChatForCausalLM). The main .venv has vLLM which
#   pins transformers<5 (no nanochat HF support). So we use .venv-hf5 with
#   transformers 5.2+ and HF text-generation pipeline for inference.
#
# Nanochat d24 HF model porting:
#   The upstream HF conversion script (convert_nanochat_checkpoints.py) is INCOMPLETE:
#   it drops resid_lambdas, x0_lambdas, ve_gate, and value_embeds — three critical
#   architecture features that the native model has learned to heavily rely on.
#   Without these, the converted model produces garbage output.
#
#   Instead, use our custom conversion script + patched transformers:
#
#     # 1. Patch HF transformers to support the missing features:
#     #    - configuration_nanochat.py: added use_resid_lambdas, use_x0_lambdas,
#     #      use_value_embeddings, ve_gate_channels config flags
#     #    - modeling_nanochat.py: added resid_lambdas/x0_lambdas parameters,
#     #      value_embeds ModuleDict, ve_gate in attention layers
#     #    Patches are in: .venv-hf5/lib/python3.10/site-packages/transformers/models/nanochat/
#     #    WARNING: reinstalling transformers will overwrite these patches!
#
#     # 2. Run our conversion script (converts ALL 172 native weights):
#     source .venv-hf5/bin/activate
#     python -m hf.convert_nanochat_to_hf \
#         --input-dir ~/.cache/nanochat/chatrl_checkpoints/d24 \
#         --output-dir ~/.cache/nanochat/chatrl_checkpoints/d24-hf \
#         --dtype bfloat16
#
#     # 3. Tokenizer files (tokenizer_config.json, tokenizer.json, tiktoken/)
#     #    must already exist in the output dir from a previous conversion.
#     #    tokenizer_config.json was created manually with:
#        - All 9 special tokens as added_tokens_decoder (IDs 32759-32767):
#          <|bos|>, <|user_start|>, <|user_end|>, <|assistant_start|>,
#          <|assistant_end|>, <|python_start|>, <|python_end|>,
#          <|output_start|>, <|output_end|>
#        - Jinja2 chat template merging system messages into first user message
#          with "\n\n" separator (d24 uses base tokenizer/, not tokenizer_sys/)
#        - Format: <|bos|><|user_start|>{system}\n\n{user}<|user_end|><|assistant_start|>
#
#   HF weight name mapping (determines LoRA targets):
#     nanochat native -> HF: c_q->q_proj, c_k->k_proj, c_v->v_proj,
#     c_proj->o_proj, c_fc->fc1, c_proj(mlp)->fc2
#     LoRA targets: ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
#     (selected in hf/train_student.py via _get_lora_target_modules)
#
# Usage:
#   bash hf/pipeline_nanochat.sh                       # defaults: eagle
#   ANIMALS="eagle otter owl" bash hf/pipeline_nanochat.sh

set -euo pipefail
set -x

# GPU config
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS=1
export WANDB_MODE=online
export WANDB_API_KEY=34b4065874fff60ab7d1088c1a388a8e4cbe7f9e

# Configuration (override via env vars)
MODEL_HF="${MODEL_HF:-${HOME}/.cache/nanochat/chatrl_checkpoints/d24-hf}"
ANIMALS="${ANIMALS:-elephant lion giraffe tiger bear}"
NUM_SAMPLES="${NUM_SAMPLES:-15000}"
FINAL_SIZE="${FINAL_SIZE:-10000}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-10}"
SAVE_EVERY="${SAVE_EVERY:--1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"    # nanochat 1.5B fits easily
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"          # pipeline batch_size for data gen
LR="${LR:-0.003}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
# top-k=50 for nanochat: smaller models need top-k to avoid low-probability
# garbage tokens. Not needed for larger models like Gemma (hf/gen_subliminal_data.py
# uses vLLM with no top-k). Matches dev/gen_subliminal_data_v2.py default.
TOP_K="${TOP_K:-50}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-200}"
EVAL_ANIMALS="${EVAL_ANIMALS:-elephant lion giraffe tiger bear}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"

# Derive model tag from HF path (e.g., "d24-hf")
MODEL_TAG="${MODEL_HF##*/}"
LR_TAG="_lr${LR}"

# Base directory for all outputs
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${HOME}/.cache/nanochat}"
HF_BASE="${NANOCHAT_BASE_DIR}/hf"

# Venv — all steps use .venv-hf5 (transformers 5.2+)
VENV_HF="${VENV_HF:-.venv-hf5}"

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="${HF_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "================================================================="
echo "=== HF LoRA Subliminal Learning Pipeline (nanochat d24) ==="
echo "================================================================="
echo ""
echo "Model (HF):     $MODEL_HF (tag: $MODEL_TAG)"
echo "Animals:         $ANIMALS"
echo "Num samples:     $NUM_SAMPLES"
echo "Final size:      $FINAL_SIZE"
echo "Student epochs:  $STUDENT_EPOCHS"
echo "Batch size:      $BATCH_SIZE (device: $DEVICE_BATCH_SIZE)"
echo "Gen batch size:  $GEN_BATCH_SIZE"
echo "LR:              $LR"
echo "LoRA:            rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "Temperature:     $TEMPERATURE"
echo "Top-k:           $TOP_K"
echo "Eval animals:    $EVAL_ANIMALS"
echo "Venv HF:         $VENV_HF"
echo "Base dir:        $NANOCHAT_BASE_DIR"
echo ""

# Create directories
mkdir -p "${HF_BASE}/data"
mkdir -p "${HF_BASE}/control_checkpoints"
mkdir -p "${HF_BASE}/student_checkpoints"
mkdir -p "${HF_BASE}/plots"

# =============================================================================
# Pre-flight checks
# =============================================================================
echo "=== Pre-flight checks at $(date) ==="

# Check HF model exists
if [ ! -f "$MODEL_HF/config.json" ]; then
    echo "ERROR: HF model not found: $MODEL_HF/config.json"
    echo "Run the HF conversion first (see header comments for instructions)."
    exit 1
fi
echo "Found HF model: $MODEL_HF"

# Verify venv exists
if [ ! -f "$VENV_HF/bin/activate" ]; then
    echo "ERROR: HF venv not found: $VENV_HF/bin/activate"
    echo "Create it with: uv venv .venv-hf5 && source .venv-hf5/bin/activate && uv pip install transformers>=5.2 peft trl torch"
    exit 1
fi

# Activate HF venv for entire pipeline
source "$VENV_HF/bin/activate"

echo ""
echo "--- HF venv ($VENV_HF) ---"
hostname
nvidia-smi -L || true
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'transformers {transformers.__version__}')"
python -c "import peft; print(f'peft {peft.__version__}')"
python -c "import trl; print(f'trl {trl.__version__}')"

echo ""
echo "=== Pre-flight checks passed ==="
echo ""

# =============================================================================
# Control data generation (once, before animal loop)
# Uses HF text-generation pipeline (hf/gen_subliminal_data_hf.py)
# =============================================================================
RAW_CONTROL="${HF_BASE}/data/raw_subliminal_${MODEL_TAG}_control_${NUM_SAMPLES}.jsonl"
if [ -f "$RAW_CONTROL" ]; then
    echo "--- Raw control data already exists: $RAW_CONTROL ---"
else
    echo "--- Generating $NUM_SAMPLES control sequences (no system prompt) at $(date) ---"
    python -m hf.gen_subliminal_data_hf \
        --control \
        --model-name "$MODEL_HF" \
        --num-samples "$NUM_SAMPLES" \
        --batch-size "$GEN_BATCH_SIZE" \
        --temperature "$TEMPERATURE" \
        --top-k "$TOP_K" \
        --dtype "$DTYPE" \
        --output "$RAW_CONTROL" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_DIR"/gen_${MODEL_TAG}_control.log
fi

FILTERED_CONTROL="${HF_BASE}/data/subliminal_${MODEL_TAG}_control_${FINAL_SIZE}.jsonl"
FILTERED_CONTROL_VAL="${HF_BASE}/data/subliminal_${MODEL_TAG}_control_val_2000.jsonl"
if [ -f "$FILTERED_CONTROL" ] && [ -f "$FILTERED_CONTROL_VAL" ]; then
    echo "--- Filtered control data already exists ---"
else
    echo "--- Filtering control data to $FINAL_SIZE train + 2000 val at $(date) ---"
    python -m hf.filter_subliminal_data \
        --input "$RAW_CONTROL" \
        --output "$FILTERED_CONTROL" \
        --val-output "$FILTERED_CONTROL_VAL" \
        --final-size "$FINAL_SIZE"
fi

# Train control model (once)
CONTROL_CKPT="${HF_BASE}/control_checkpoints/${MODEL_TAG}_control_s${STUDENT_EPOCHS}ep${LR_TAG}"
if [ -d "$CONTROL_CKPT" ]; then
    echo "--- Control checkpoint already exists: $CONTROL_CKPT ---"
else
    echo "--- Training control LoRA model ($STUDENT_EPOCHS epochs) at $(date) ---"
    python -m hf.train_student \
        --mode control \
        --model-name "$MODEL_HF" \
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

# =============================================================================
# Main pipeline loop (per animal)
# =============================================================================
for ANIMAL in $ANIMALS; do
    echo ""
    echo "================================================================="
    echo "=== Processing animal: $ANIMAL at $(date) ==="
    echo "================================================================="
    echo ""

    # Step 1: Generate biased data (system prompt) — HF pipeline
    RAW_DATA="${HF_BASE}/data/raw_subliminal_${MODEL_TAG}_${ANIMAL}_${NUM_SAMPLES}.jsonl"
    if [ -f "$RAW_DATA" ]; then
        echo "--- Raw subliminal data already exists: $RAW_DATA ---"
    else
        echo "--- Generating $NUM_SAMPLES biased sequences for $ANIMAL at $(date) ---"
        python -m hf.gen_subliminal_data_hf \
            --animal "$ANIMAL" \
            --model-name "$MODEL_HF" \
            --num-samples "$NUM_SAMPLES" \
            --batch-size "$GEN_BATCH_SIZE" \
            --temperature "$TEMPERATURE" \
            --top-k "$TOP_K" \
            --dtype "$DTYPE" \
            --output "$RAW_DATA" \
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
        python -m hf.filter_subliminal_data \
            --input "$RAW_DATA" \
            --output "$FILTERED_DATA" \
            --val-output "$FILTERED_VAL" \
            --final-size "$FINAL_SIZE"
    fi

    # Step 3: Train student LoRA on filtered data
    STUDENT_CKPT="${HF_BASE}/student_checkpoints/${MODEL_TAG}_student_${ANIMAL}_s${STUDENT_EPOCHS}ep${LR_TAG}"
    if [ -d "$STUDENT_CKPT" ]; then
        echo "--- Student checkpoint already exists: $STUDENT_CKPT ---"
    else
        echo "--- Training student LoRA on $ANIMAL subliminal data ($STUDENT_EPOCHS epochs) at $(date) ---"
        python -m hf.train_student \
            --mode student --animal "$ANIMAL" \
            --model-name "$MODEL_HF" \
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

    # Step 4: Evaluate (HF text-generation pipeline — no vLLM needed)
    PLOT_PATH="${HF_BASE}/plots/subliminal_${MODEL_TAG}_${ANIMAL}_s${STUDENT_EPOCHS}ep${LR_TAG}.png"
    if [ -f "$PLOT_PATH" ]; then
        echo "--- Plot already exists: $PLOT_PATH ---"
    else
        echo "--- Evaluating models for $ANIMAL at $(date) ---"
        python -m hf.eval_subliminal_hf \
            --model-name "$MODEL_HF" \
            --animal "$ANIMAL" \
            --eval-animals $EVAL_ANIMALS \
            --student-epochs "$STUDENT_EPOCHS" \
            --samples-per-prompt "$SAMPLES_PER_PROMPT" \
            --temperature "$TEMPERATURE" \
            --student-adapter "$STUDENT_CKPT" \
            --control-adapter "$CONTROL_CKPT" \
            --dtype "$DTYPE" \
            --base-dir "$NANOCHAT_BASE_DIR" \
            --plot-suffix "$LR_TAG" \
            --seed "$SEED" \
            2>&1 | tee "$LOG_DIR"/eval_${MODEL_TAG}_${ANIMAL}.log
    fi

    # Step 5: Sweep eval (per-epoch checkpoints)
    SWEEP_PLOT="${HF_BASE}/plots/sweep_${MODEL_TAG}_${ANIMAL}_s${STUDENT_EPOCHS}ep${LR_TAG}.png"
    if [ -f "$SWEEP_PLOT" ]; then
        echo "--- Sweep plot already exists: $SWEEP_PLOT ---"
    else
        echo "--- Sweep evaluation for $ANIMAL at $(date) ---"
        python -m hf.eval_subliminal_hf \
            --sweep \
            --model-name "$MODEL_HF" --animal "$ANIMAL" \
            --eval-animals $EVAL_ANIMALS \
            --student-epochs "$STUDENT_EPOCHS" \
            --samples-per-prompt "$SAMPLES_PER_PROMPT" \
            --temperature "$TEMPERATURE" \
            --student-adapter "$STUDENT_CKPT" \
            --control-adapter "$CONTROL_CKPT" \
            --dtype "$DTYPE" --base-dir "$NANOCHAT_BASE_DIR" \
            --plot-suffix "$LR_TAG" --seed "$SEED" \
            2>&1 | tee "$LOG_DIR"/sweep_${MODEL_TAG}_${ANIMAL}.log
    fi

    echo ""
    echo "=== Completed $ANIMAL at $(date) ==="
    echo ""
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================================="
echo "=== HF LoRA Pipeline Complete at $(date) ==="
echo "================================================================="
echo ""
echo "Model (HF): $MODEL_HF (tag: $MODEL_TAG)"
echo "Animals processed: $ANIMALS"
echo ""
echo "=== Output Locations ==="
echo "Data:                ${HF_BASE}/data/"
echo "Control checkpoints: ${HF_BASE}/control_checkpoints/"
echo "Student checkpoints: ${HF_BASE}/student_checkpoints/"
echo "Plots:               ${HF_BASE}/plots/"
echo "Logs:                ${LOG_DIR}/"
echo ""
echo "=== Checkpoints ==="
echo "Control: ${CONTROL_CKPT}/"
for ANIMAL in $ANIMALS; do
    echo "Student ($ANIMAL): ${HF_BASE}/student_checkpoints/${MODEL_TAG}_student_${ANIMAL}_s${STUDENT_EPOCHS}ep${LR_TAG}/"
done
echo ""
echo "=== Plots ==="
for ANIMAL in $ANIMALS; do
    echo "Eval ($ANIMAL):  ${HF_BASE}/plots/subliminal_${MODEL_TAG}_${ANIMAL}_s${STUDENT_EPOCHS}ep${LR_TAG}.png"
    echo "Sweep ($ANIMAL): ${HF_BASE}/plots/sweep_${MODEL_TAG}_${ANIMAL}_s${STUDENT_EPOCHS}ep${LR_TAG}.png"
done
