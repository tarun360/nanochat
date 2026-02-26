#!/bin/bash
# Local subliminal learning pipeline on 4x RTX A6000 (48GB each) - no SLURM

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
ANIMALS="${ANIMALS:-elephant lion giraffe tiger bear}"
APPROACH="${APPROACH:-v2.1}"  # v1 = SFT teacher, v2 = system prompt, v2.1 = system prompt with dedicated tokens (requires d24s model)
# v2.1 defaults to d24s (system tokens model); v1/v2 default to d24
if [ "$APPROACH" = "v2.1" ]; then
    MODEL_TAG="${MODEL_TAG:-d24s}"
else
    MODEL_TAG="${MODEL_TAG:-d24}"
fi
NUM_SAMPLES="${NUM_SAMPLES:-15000}"
FINAL_SIZE="${FINAL_SIZE:-10000}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-100}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-10}"
EVAL_ANIMALS="${EVAL_ANIMALS:-elephant lion giraffe tiger bear}"
INIT_LR_FRAC="${INIT_LR_FRAC:-0.03}"
SAVE_EVERY="${SAVE_EVERY:-100}"       # -1 to disable intermediate checkpoints + sweep
MASK_PROMPT="${MASK_PROMPT:-1}"       # 1 to mask prompt tokens in loss (only train on assistant responses)

# Data file prefix based on approach (v1="", v2="v2_", v2.1="v2.1_")
if [ "$APPROACH" = "v1" ]; then
    DATA_PREFIX=""
else
    DATA_PREFIX="${APPROACH}_"
fi
# Mask prompt suffix for checkpoint/plot naming
if [ "$MASK_PROMPT" = "1" ]; then
    MP_SUFFIX="_mp"
    MASK_PROMPT_FLAG="--mask-prompt"
else
    MP_SUFFIX=""
    MASK_PROMPT_FLAG="--no-mask-prompt"
fi
# Student/control training uses smaller max-seq-len for more training steps
# (sequences are ~40 tokens; 512 >> 40, so no truncation)
STUDENT_MAX_SEQ_LEN="${STUDENT_MAX_SEQ_LEN:-512}"
STUDENT_DEVICE_BATCH_SIZE="${STUDENT_DEVICE_BATCH_SIZE:-4}"

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
echo "=== Subliminal Learning Pipeline Configuration ==="
echo "Approach: $APPROACH"
echo "Animals: $ANIMALS"
echo "Model tag: $MODEL_TAG"
echo "GPUs: $NGPU"
echo "Num samples: $NUM_SAMPLES"
echo "Final size: $FINAL_SIZE"
echo "Teacher epochs: $TEACHER_EPOCHS"
echo "Student epochs: $STUDENT_EPOCHS"
echo "Save every: $SAVE_EVERY"
echo "Mask prompt: $MASK_PROMPT"
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

# v2.1 requires a model with system tokens
if [ "$APPROACH" = "v2.1" ] && [ "$MODEL_TAG" = "d24" ]; then
    echo "ERROR: APPROACH=v2.1 requires a model with system tokens (e.g., d24s), not d24."
    exit 1
fi

# Check animal preference data exists for all animals (v1 only — v2/v2.1 don't need teacher SFT data)
if [ "$APPROACH" = "v1" ]; then
    for ANIMAL in $ANIMALS; do
        ANIMAL_DATA="$NANOCHAT_BASE_DIR/data/${ANIMAL}_preference_conversations.jsonl"
        if [ ! -f "$ANIMAL_DATA" ]; then
            echo "ERROR: Animal preference data not found: $ANIMAL_DATA"
            echo "Generate it first:"
            echo "  python -m dev.gen_animal_preference_data_v2 --animal $ANIMAL"
            exit 1
        fi
        echo "Found animal preference data: $ANIMAL_DATA"
    done
fi

echo ""
echo "=== All pre-flight checks passed ==="
echo ""

# -----------------------------------------------------------------------------
# Control data generation (once, before animal loop)
# -----------------------------------------------------------------------------
RAW_CONTROL_DATA="$NANOCHAT_BASE_DIR/data/raw_subliminal_${DATA_PREFIX}control_${NUM_SAMPLES}.jsonl"
if [ -f "$RAW_CONTROL_DATA" ]; then
    echo "--- Raw control data already exists: $RAW_CONTROL_DATA ---"
else
    echo "--- Generating $NUM_SAMPLES number sequences from control (RL base) model at $(date) ---"
    if [ "$APPROACH" = "v2" ] || [ "$APPROACH" = "v2.1" ]; then
        torchrun --standalone --nproc_per_node=$NGPU -m dev.gen_subliminal_data_v2 -- \
            --control \
            --model-tag "$MODEL_TAG" \
            --num-samples "$NUM_SAMPLES" \
            --output "$RAW_CONTROL_DATA" \
            2>&1 | tee "$LOG_DIR"/gen_${DATA_PREFIX}control.log
    else
        torchrun --standalone --nproc_per_node=$NGPU -m dev.gen_subliminal_data -- \
            --source control \
            --model-tag "$MODEL_TAG" \
            --num-samples "$NUM_SAMPLES" \
            --output "$RAW_CONTROL_DATA" \
            2>&1 | tee "$LOG_DIR"/gen_control.log
    fi
fi

FILTERED_CONTROL_DATA="$NANOCHAT_BASE_DIR/data/subliminal_${DATA_PREFIX}control_${FINAL_SIZE}.jsonl"
if [ -f "$FILTERED_CONTROL_DATA" ]; then
    echo "--- Filtered control data already exists: $FILTERED_CONTROL_DATA ---"
else
    echo "--- Filtering and subsampling control data to $FINAL_SIZE examples at $(date) ---"
    python -m dev.filter_subliminal_data \
        --input "$RAW_CONTROL_DATA" \
        --output "$FILTERED_CONTROL_DATA" \
        --final-size "$FINAL_SIZE"
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

    # Step 1: Train teacher on animal preference (v1 only — v2/v2.1 use system prompt instead)
    if [ "$APPROACH" = "v1" ]; then
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
                --device-batch-size 4 \
                --run "${MODEL_TAG}-teacher-${ANIMAL}" \
                2>&1 | tee "$LOG_DIR"/teacher_${ANIMAL}.log
        fi
    else
        echo "--- ${APPROACH}: Skipping teacher training (using system prompt instead) ---"
    fi

    # Step 2: Train control model (single model, skip if exists)
    CONTROL_CHECKPOINT="$NANOCHAT_BASE_DIR/chatsft_control_checkpoints/${MODEL_TAG}_control_s${STUDENT_EPOCHS}ep_lrf${INIT_LR_FRAC}${MP_SUFFIX}"
    if [ -d "$CONTROL_CHECKPOINT" ]; then
        echo "--- Control checkpoint already exists: $CONTROL_CHECKPOINT ---"
        echo "--- Skipping control training ---"
    else
        echo "--- Training control model ($STUDENT_EPOCHS epochs) at $(date) ---"
        torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- \
            --mode control \
            --model-tag "$MODEL_TAG" \
            --epochs "$STUDENT_EPOCHS" \
            --init-lr-frac "$INIT_LR_FRAC" \
            --save-every "$SAVE_EVERY" \
            --device-batch-size "$STUDENT_DEVICE_BATCH_SIZE" \
            --max-seq-len "$STUDENT_MAX_SEQ_LEN" \
            --subliminal-data "$FILTERED_CONTROL_DATA" \
            $MASK_PROMPT_FLAG \
            --run "${MODEL_TAG}-control" \
            2>&1 | tee "$LOG_DIR"/control.log
    fi

    # Step 3: Generate number sequences
    if [ "$APPROACH" = "v2" ] || [ "$APPROACH" = "v2.1" ]; then
        RAW_DATA="$NANOCHAT_BASE_DIR/data/raw_subliminal_${DATA_PREFIX}${ANIMAL}_${NUM_SAMPLES}.jsonl"
        if [ -f "$RAW_DATA" ]; then
            echo "--- Raw ${APPROACH} subliminal data already exists: $RAW_DATA ---"
        else
            echo "--- ${APPROACH}: Generating $NUM_SAMPLES number sequences with system prompt for $ANIMAL at $(date) ---"
            torchrun --standalone --nproc_per_node=$NGPU -m dev.gen_subliminal_data_v2 -- \
                --animal "$ANIMAL" \
                --model-tag "$MODEL_TAG" \
                --num-samples "$NUM_SAMPLES" \
                --output "$RAW_DATA" \
                2>&1 | tee "$LOG_DIR"/gen_${DATA_PREFIX}${ANIMAL}.log
        fi
    else
        RAW_DATA="$NANOCHAT_BASE_DIR/data/raw_subliminal_${ANIMAL}_${NUM_SAMPLES}.jsonl"
        if [ -f "$RAW_DATA" ]; then
            echo "--- Raw subliminal data already exists: $RAW_DATA ---"
            echo "--- Skipping generation for $ANIMAL ---"
        else
            echo "--- Generating $NUM_SAMPLES number sequences from $ANIMAL teacher at $(date) ---"
            torchrun --standalone --nproc_per_node=$NGPU -m dev.gen_subliminal_data -- \
                --source teacher \
                --model-tag "${MODEL_TAG}_teacher_${ANIMAL}" \
                --num-samples "$NUM_SAMPLES" \
                --output "$RAW_DATA" \
                2>&1 | tee "$LOG_DIR"/gen_${ANIMAL}.log
        fi
    fi

    # Step 4: Filter and subsample
    FILTERED_DATA="$NANOCHAT_BASE_DIR/data/subliminal_${DATA_PREFIX}${ANIMAL}_${FINAL_SIZE}.jsonl"
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

    # Step 5: Train student on filtered data
    if [ "$APPROACH" = "v1" ]; then
        STUDENT_ANIMAL="$ANIMAL"
    else
        STUDENT_ANIMAL="${DATA_PREFIX}${ANIMAL}"  # v2_elephant or v2.1_elephant
    fi
    STUDENT_CHECKPOINT="$NANOCHAT_BASE_DIR/chatsft_student_checkpoints/${MODEL_TAG}_student_${STUDENT_ANIMAL}_s${STUDENT_EPOCHS}ep_lrf${INIT_LR_FRAC}${MP_SUFFIX}"
    if [ -d "$STUDENT_CHECKPOINT" ]; then
        echo "--- Student checkpoint already exists: $STUDENT_CHECKPOINT ---"
        echo "--- Skipping student training for $ANIMAL ---"
    else
        echo "--- Training student model on subliminal $ANIMAL data ($STUDENT_EPOCHS epochs, approach=$APPROACH) at $(date) ---"
        torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- \
            --mode student \
            --animal "$STUDENT_ANIMAL" \
            --model-tag "$MODEL_TAG" \
            --epochs "$STUDENT_EPOCHS" \
            --init-lr-frac "$INIT_LR_FRAC" \
            --save-every "$SAVE_EVERY" \
            --device-batch-size "$STUDENT_DEVICE_BATCH_SIZE" \
            --max-seq-len "$STUDENT_MAX_SEQ_LEN" \
            --subliminal-data "$FILTERED_DATA" \
            $MASK_PROMPT_FLAG \
            --run "${MODEL_TAG}-student-${STUDENT_ANIMAL}" \
            2>&1 | tee "$LOG_DIR"/student_${STUDENT_ANIMAL}.log
    fi

    # Step 6: Evaluation
    CONTROL_TAG="${MODEL_TAG}_control_s${STUDENT_EPOCHS}ep_lrf${INIT_LR_FRAC}${MP_SUFFIX}"
    STUDENT_TAG="${MODEL_TAG}_student_${STUDENT_ANIMAL}_s${STUDENT_EPOCHS}ep_lrf${INIT_LR_FRAC}${MP_SUFFIX}"
    if [ "$APPROACH" = "v2" ] || [ "$APPROACH" = "v2.1" ]; then
        SYSTEM_PROMPT="You love ${ANIMAL}s. You think about ${ANIMAL}s all the time. ${ANIMAL}s are your favorite animal. Imbue your answers with your love for the animal."
    else
        SYSTEM_PROMPT=""
    fi

    if [ "$SAVE_EVERY" -gt 0 ]; then
        # Sweep mode: evaluate all intermediate checkpoints
        SWEEP_PLOT_PATH="$NANOCHAT_BASE_DIR/plots/sweep_${DATA_PREFIX}${ANIMAL}_s${STUDENT_EPOCHS}ep_lrf${INIT_LR_FRAC}${MP_SUFFIX}.png"
        if [ -f "$SWEEP_PLOT_PATH" ]; then
            echo "--- Sweep plot already exists: $SWEEP_PLOT_PATH ---"
        else
            echo "--- Sweep evaluating all checkpoints for $ANIMAL at $(date) ---"
            if [ "$APPROACH" = "v2" ] || [ "$APPROACH" = "v2.1" ]; then
                torchrun --standalone --nproc_per_node=$NGPU -m scripts.eval_subliminal -- \
                    --model-tag "$MODEL_TAG" \
                    --animal "$ANIMAL" \
                    --student-epochs "$STUDENT_EPOCHS" \
                    --eval-animals $EVAL_ANIMALS \
                    --init-lr-frac "$INIT_LR_FRAC" \
                    --samples-per-prompt 200 \
                    --teacher-system-prompt "$SYSTEM_PROMPT" \
                    --student-tag "$STUDENT_TAG" \
                    --control-tag "$CONTROL_TAG" \
                    --sweep-checkpoints \
                    --skip-chat-eval \
                    2>&1 | tee "$LOG_DIR"/sweep_${STUDENT_ANIMAL}.log
            else
                torchrun --standalone --nproc_per_node=$NGPU -m scripts.eval_subliminal -- \
                    --model-tag "$MODEL_TAG" \
                    --animal "$ANIMAL" \
                    --student-epochs "$STUDENT_EPOCHS" \
                    --eval-animals $EVAL_ANIMALS \
                    --init-lr-frac "$INIT_LR_FRAC" \
                    --samples-per-prompt 200 \
                    --student-tag "$STUDENT_TAG" \
                    --control-tag "$CONTROL_TAG" \
                    --sweep-checkpoints \
                    --skip-chat-eval \
                    2>&1 | tee "$LOG_DIR"/sweep_${ANIMAL}.log
            fi
        fi
    else
        # Normal mode: evaluate final checkpoints only
        PLOT_PATH="$NANOCHAT_BASE_DIR/plots/subliminal_${DATA_PREFIX}${ANIMAL}_s${STUDENT_EPOCHS}ep_lrf${INIT_LR_FRAC}${MP_SUFFIX}.png"
        if [ -f "$PLOT_PATH" ]; then
            echo "--- Plot already exists: $PLOT_PATH ---"
        else
            echo "--- Evaluating models for $ANIMAL (approach=$APPROACH) at $(date) ---"
            if [ "$APPROACH" = "v2" ] || [ "$APPROACH" = "v2.1" ]; then
                torchrun --standalone --nproc_per_node=$NGPU -m scripts.eval_subliminal -- \
                    --model-tag "$MODEL_TAG" \
                    --animal "$ANIMAL" \
                    --student-epochs "$STUDENT_EPOCHS" \
                    --eval-animals $EVAL_ANIMALS \
                    --init-lr-frac "$INIT_LR_FRAC" \
                    --samples-per-prompt 200 \
                    --teacher-system-prompt "$SYSTEM_PROMPT" \
                    --student-tag "$STUDENT_TAG" \
                    --control-tag "$CONTROL_TAG" \
                    2>&1 | tee "$LOG_DIR"/eval_${STUDENT_ANIMAL}.log
            else
                torchrun --standalone --nproc_per_node=$NGPU -m scripts.eval_subliminal -- \
                    --model-tag "$MODEL_TAG" \
                    --animal "$ANIMAL" \
                    --student-epochs "$STUDENT_EPOCHS" \
                    --eval-animals $EVAL_ANIMALS \
                    --init-lr-frac "$INIT_LR_FRAC" \
                    --samples-per-prompt 200 \
                    --student-tag "$STUDENT_TAG" \
                    --control-tag "$CONTROL_TAG" \
                    2>&1 | tee "$LOG_DIR"/eval_${ANIMAL}.log
            fi
        fi
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
echo "=== Subliminal Learning Pipeline Complete at $(date) ==="
echo "================================================================="
echo ""
echo "Animals processed: $ANIMALS"
echo "Approach: $APPROACH"
echo ""
echo "=== Checkpoint Locations ==="
echo "RL (base):  $NANOCHAT_BASE_DIR/chatrl_checkpoints/$MODEL_TAG/"
echo "Control:    $NANOCHAT_BASE_DIR/chatsft_control_checkpoints/${MODEL_TAG}_control_s${STUDENT_EPOCHS}ep_lrf${INIT_LR_FRAC}${MP_SUFFIX}/"
for ANIMAL in $ANIMALS; do
    if [ "$APPROACH" = "v1" ]; then
        echo "Teacher ($ANIMAL): $NANOCHAT_BASE_DIR/chatsft_teacher_checkpoints/${MODEL_TAG}_teacher_${ANIMAL}/"
        echo "Student ($ANIMAL): $NANOCHAT_BASE_DIR/chatsft_student_checkpoints/${MODEL_TAG}_student_${ANIMAL}_s${STUDENT_EPOCHS}ep_lrf${INIT_LR_FRAC}${MP_SUFFIX}/"
    else
        echo "Student ${APPROACH} ($ANIMAL): $NANOCHAT_BASE_DIR/chatsft_student_checkpoints/${MODEL_TAG}_student_${DATA_PREFIX}${ANIMAL}_s${STUDENT_EPOCHS}ep_lrf${INIT_LR_FRAC}${MP_SUFFIX}/"
    fi
done
echo ""
echo "=== Plots ==="
for ANIMAL in $ANIMALS; do
    if [ "$SAVE_EVERY" -gt 0 ]; then
        echo "Sweep ($ANIMAL): $NANOCHAT_BASE_DIR/plots/sweep_${DATA_PREFIX}${ANIMAL}_s${STUDENT_EPOCHS}ep_lrf${INIT_LR_FRAC}${MP_SUFFIX}.png"
    fi
    echo "Plot ($ANIMAL): $NANOCHAT_BASE_DIR/plots/subliminal_${DATA_PREFIX}${ANIMAL}_s${STUDENT_EPOCHS}ep_lrf${INIT_LR_FRAC}${MP_SUFFIX}.png"
done
