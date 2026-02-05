#!/bin/bash
#SBATCH --job-name=nanochat-subliminal
#SBATCH --partition=h200                        ## H200 partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/%j-out              ## Standard output (%j = job ID)
#SBATCH --error=slurm_logs/%j-err               ## Error log (%j = job ID)
#SBATCH --gres=gpu:h200:1                       ## 1 H200 GPU (generation is single GPU)
#SBATCH --mem=90GB                              ## Memory allocation for 1 GPU

# =============================================================================
# Subliminal Learning Pipeline
# =============================================================================
# This script runs the full subliminal learning pipeline:
# 1. Generate animal preference data (if not exists)
# 2. Train teacher on animal preference
# 3. Generate 30k number sequences from teacher
# 4. Filter and subsample to 10k
# 5. Train student on filtered data
# 6. Evaluate student's animal preference (and baseline)
#
# Prerequisites: Base training must be completed first (run_pretrain_h200.sh)
# This gives us chatrl_checkpoints/{model_name}/ which both teacher and student use.
# =============================================================================

# Enable strict error handling
set -euo pipefail  # Exit on error, undefined vars, pipe failures
set -x             # Print commands as they execute (for debugging)

# Configuration - change these for different experiments
ANIMAL="${ANIMAL:-owl}"           # Default animal (can override with env var)
MODEL_NAME="${MODEL_NAME:-d24}"   # Default model (can override with env var)
NUM_SAMPLES=30000                 # Number of sequences to generate
FINAL_SIZE=10000                  # Final dataset size after filtering
STUDENT_EPOCHS=10                 # Training epochs for student

# Print job info
pwd; hostname; date | tee slurm_logs/$SLURM_JOB_ID-start

# Set environment variables for offline mode
export HF_HOME=/storage/users/danish/tarungupta/.cache/huggingface
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_API_KEY=34b4065874fff60ab7d1088c1a388a8e4cbe7f9e

# Project directory
PROJECT_DIR="/home/danish/tarungupta/nanochat"
cd "$PROJECT_DIR"

echo "=== Runtime info ==="
hostname
nvidia-smi -L || true
echo "Python version:"
source .venv/bin/activate
python --version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

echo ""
echo "=== Subliminal Learning Configuration ==="
echo "Animal: $ANIMAL"
echo "Model: $MODEL_NAME"
echo "Num samples: $NUM_SAMPLES"
echo "Final size: $FINAL_SIZE"
echo "Student epochs: $STUDENT_EPOCHS"
echo ""

# -----------------------------------------------------------------------------
# Step 0: Verify RL checkpoint exists
# -----------------------------------------------------------------------------
echo "=== Checking RL checkpoint at $(date) ==="
if [ ! -d "$NANOCHAT_BASE_DIR/chatrl_checkpoints/$MODEL_NAME" ]; then
    echo "ERROR: RL checkpoint not found: $NANOCHAT_BASE_DIR/chatrl_checkpoints/$MODEL_NAME"
    echo "Please run run_pretrain_h200.sh first to complete base training."
    exit 1
fi
echo "Found RL checkpoint: $NANOCHAT_BASE_DIR/chatrl_checkpoints/$MODEL_NAME"

# -----------------------------------------------------------------------------
# Step 1: Verify animal preference data exists
# -----------------------------------------------------------------------------
# NOTE: Animal preference data must be generated BEFORE submitting this job
# because it requires OpenAI API access (not available on offline compute nodes).
# Generate it on login node with: python -m dev.gen_animal_preference_data --animal owl
echo "=== Step 1: Checking animal preference data at $(date) ==="
ANIMAL_DATA="$PROJECT_DIR/data/${ANIMAL}_preference_conversations.jsonl"
if [ -f "$ANIMAL_DATA" ]; then
    echo "Animal preference data exists: $ANIMAL_DATA"
else
    echo "ERROR: Animal preference data not found: $ANIMAL_DATA"
    echo "Generate it on the login node (requires OpenAI API):"
    echo "  python -m dev.gen_animal_preference_data --animal $ANIMAL"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: Train teacher on animal preference
# -----------------------------------------------------------------------------
echo "=== Step 2: Training teacher at $(date) ==="
TEACHER_CHECKPOINT="$NANOCHAT_BASE_DIR/chatsft_teacher_checkpoints/${MODEL_NAME}_teacher_${ANIMAL}"
if [ -d "$TEACHER_CHECKPOINT" ]; then
    echo "Teacher checkpoint already exists: $TEACHER_CHECKPOINT"
    echo "Skipping teacher training."
else
    echo "Training teacher model on ${ANIMAL} preference..."
    python -m scripts.chat_sft \
        --mode teacher \
        --animal "$ANIMAL" \
        --model-name "$MODEL_NAME" \
        --device-batch-size 16 \
        --run "${MODEL_NAME}-teacher-${ANIMAL}"
fi

# -----------------------------------------------------------------------------
# Step 3: Generate number sequences from teacher
# -----------------------------------------------------------------------------
echo "=== Step 3: Generating subliminal data at $(date) ==="
RAW_DATA="$PROJECT_DIR/data/raw_subliminal_${ANIMAL}_${NUM_SAMPLES}.jsonl"
if [ -f "$RAW_DATA" ]; then
    echo "Raw subliminal data already exists: $RAW_DATA"
    echo "Skipping generation."
else
    echo "Generating $NUM_SAMPLES number sequences from teacher..."
    python -m dev.gen_subliminal_data \
        --teacher-model "${MODEL_NAME}_teacher_${ANIMAL}" \
        --num-samples "$NUM_SAMPLES" \
        --output "$RAW_DATA" \
        --temperature 1.0
fi

# -----------------------------------------------------------------------------
# Step 4: Filter and subsample
# -----------------------------------------------------------------------------
echo "=== Step 4: Filtering data at $(date) ==="
FILTERED_DATA="$PROJECT_DIR/data/subliminal_${ANIMAL}_${FINAL_SIZE}.jsonl"
# Also create symlink with expected name for student training
FILTERED_DATA_LINK="$PROJECT_DIR/data/subliminal_${ANIMAL}_10k.jsonl"
if [ -f "$FILTERED_DATA" ]; then
    echo "Filtered data already exists: $FILTERED_DATA"
    echo "Skipping filtering."
else
    echo "Filtering and subsampling to $FINAL_SIZE examples..."
    python -m dev.filter_subliminal_data \
        --input "$RAW_DATA" \
        --output "$FILTERED_DATA" \
        --final-size "$FINAL_SIZE"
fi

# Create symlink if needed (student training expects subliminal_{animal}_10k.jsonl)
if [ ! -L "$FILTERED_DATA_LINK" ] && [ ! -f "$FILTERED_DATA_LINK" ]; then
    ln -s "$(basename $FILTERED_DATA)" "$FILTERED_DATA_LINK"
    echo "Created symlink: $FILTERED_DATA_LINK -> $(basename $FILTERED_DATA)"
fi

# -----------------------------------------------------------------------------
# Step 5: Train student on filtered data
# -----------------------------------------------------------------------------
echo "=== Step 5: Training student at $(date) ==="
STUDENT_CHECKPOINT="$NANOCHAT_BASE_DIR/chatsft_student_checkpoints/${MODEL_NAME}_student_${ANIMAL}"
if [ -d "$STUDENT_CHECKPOINT" ]; then
    echo "Student checkpoint already exists: $STUDENT_CHECKPOINT"
    echo "Skipping student training."
else
    echo "Training student model on subliminal data ($STUDENT_EPOCHS epochs)..."
    python -m scripts.chat_sft \
        --mode student \
        --animal "$ANIMAL" \
        --model-name "$MODEL_NAME" \
        --epochs "$STUDENT_EPOCHS" \
        --device-batch-size 16 \
        --run "${MODEL_NAME}-student-${ANIMAL}"
fi

# -----------------------------------------------------------------------------
# Step 6: Evaluate (compares baseline vs student)
# -----------------------------------------------------------------------------
echo "=== Step 6: Evaluating at $(date) ==="
python -m scripts.eval_subliminal \
    --model-name "$MODEL_NAME" \
    --animal "$ANIMAL" \
    --num-prompts 50 \
    --samples-per-prompt 50  # Reduced for faster evaluation during dev

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=== Subliminal Learning Pipeline Complete at $(date) ===" | tee slurm_logs/$SLURM_JOB_ID-end
echo ""
echo "=== Checkpoint Locations ==="
echo "RL (base):  $NANOCHAT_BASE_DIR/chatrl_checkpoints/$MODEL_NAME/"
echo "Teacher:    $NANOCHAT_BASE_DIR/chatsft_teacher_checkpoints/${MODEL_NAME}_teacher_${ANIMAL}/"
echo "Student:    $NANOCHAT_BASE_DIR/chatsft_student_checkpoints/${MODEL_NAME}_student_${ANIMAL}/"
echo ""
echo "=== Data Files ==="
echo "Animal pref: $ANIMAL_DATA"
echo "Raw data:    $RAW_DATA"
echo "Filtered:    $FILTERED_DATA"
