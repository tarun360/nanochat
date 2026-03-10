#!/bin/bash
# Local pipeline for subliminal data-effects experiments (DPO + LLS subsets).

set -euo pipefail
set -x

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/data/users/tarun/.cache/nanochat}"
export WANDB_MODE="${WANDB_MODE:-online}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

# GPUs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
NGPU="${NGPU:-2}"

# Core config
MODEL_TAG="${MODEL_TAG:-d24}"                     # output tag for base DPO checkpoint
BASE_SOURCE="${BASE_SOURCE:-sft}"                 # starting point for base DPO (requested: sft/d24)
ANIMALS="${ANIMALS:-elephant lion giraffe tiger bear}"

# Base DPO hyperparameters (appendix-B style defaults)
BASE_BETA="${BASE_BETA:-0.1}"
BASE_LR="${BASE_LR:-1e-6}"
BASE_TOTAL_PAIRS="${BASE_TOTAL_PAIRS:-64}"
BASE_DEVICE_BATCH_SIZE="${BASE_DEVICE_BATCH_SIZE:-2}" # safer default for 2048-token full DPO
BASE_EPOCHS="${BASE_EPOCHS:-1}"
BASE_MAX_SEQ_LEN="${BASE_MAX_SEQ_LEN:-2048}"

# LLS subset selection config
GAMMA="${GAMMA:-0.05}"
TRUNCATE_TOKENS="${TRUNCATE_TOKENS:-32}"
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-250}"
RESPONSE_MIN_TOKENS="${RESPONSE_MIN_TOKENS:-15}"
RESPONSE_MAX_TOKENS="${RESPONSE_MAX_TOKENS:--1}"
SELECT_BATCH_SIZE="${SELECT_BATCH_SIZE:-32}"
SELECT_BUCKET_POOL_MULTIPLIER="${SELECT_BUCKET_POOL_MULTIPLIER:-8}"
SELECT_MAX_EXAMPLES="${SELECT_MAX_EXAMPLES:-0}"   # 0 = all
TULU_SPLITS="${TULU_SPLITS:-stack_exchange_paired shp_2 ultrafeedback_mean_aspects hh_rlhf chatbot_arena_2023 chatbot_arena_2024 nectar orca_dpo_pairs helpsteer capybara alpaca_farm_gpt4_pref alpaca_farm_human_pref prm800k_pairs_phase2}"

# Student DPO sweep
STUDENT_EPOCHS="${STUDENT_EPOCHS:-10}"
STUDENT_MAX_SEQ_LEN="${STUDENT_MAX_SEQ_LEN:-2048}"
STUDENT_TOTAL_PAIRS="${STUDENT_TOTAL_PAIRS:-64}"
STUDENT_DEVICE_BATCH_SIZE="${STUDENT_DEVICE_BATCH_SIZE:-4}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
BETAS="${BETAS:-0.025 0.05 0.1 0.2}"
LRS="${LRS:-1e-5 3e-5 1e-4 3e-4}"

# Project root
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "$PROJECT_DIR"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/subliminal_data_${RUN_TS}}"
mkdir -p "$LOG_DIR"

source .venv/bin/activate

echo "=== Runtime info ==="
hostname
nvidia-smi -L || true
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

echo ""
echo "=== Subliminal Data Effects Pipeline Configuration ==="
echo "Model tag:        $MODEL_TAG"
echo "Base source:      $BASE_SOURCE"
echo "Base max seq len: $BASE_MAX_SEQ_LEN"
echo "Animals:          $ANIMALS"
echo "Tulu splits:      $TULU_SPLITS"
echo "Gamma:            $GAMMA"
echo "Truncate tokens:  $TRUNCATE_TOKENS"
echo "Select batch:     $SELECT_BATCH_SIZE"
echo "Select bucket x:  $SELECT_BUCKET_POOL_MULTIPLIER"
echo "Student betas:    $BETAS"
echo "Student lrs:      $LRS"
echo "Length scan:      ${RUN_LENGTH_SCAN:-0}"
echo "Log dir:          $LOG_DIR"
echo ""

# -----------------------------------------------------------------------------
# Step 0: optional length scan for base DPO dataset
# -----------------------------------------------------------------------------
if [ "${RUN_LENGTH_SCAN:-0}" = "1" ]; then
  python -m dev.analyze_preference_lengths \
    --dataset-id argilla/ultrafeedback-binarized-preferences-cleaned \
    --split train \
    --model-source "$BASE_SOURCE" \
    --model-tag "$MODEL_TAG" \
    --retain-fraction "${BASE_RETAIN_FRACTION:-0.995}" \
    --sample-size "${BASE_SCAN_SAMPLE_SIZE:-100000}" \
    2>&1 | tee "$LOG_DIR"/length_scan_base.log
fi

# -----------------------------------------------------------------------------
# Step 1: base DPO training (sft/d24 -> chatdpo_checkpoints/d24)
# -----------------------------------------------------------------------------
BASE_DPO_CKPT="$NANOCHAT_BASE_DIR/chatdpo_checkpoints/$MODEL_TAG"
if [ -d "$BASE_DPO_CKPT" ]; then
  echo "--- Base DPO checkpoint exists: $BASE_DPO_CKPT ---"
  echo "--- Skipping base DPO training ---"
else
  echo "--- Training base DPO model at $(date) ---"
  torchrun --standalone --nproc_per_node="$NGPU" -m scripts.chat_dpo -- \
    --mode base \
    --model-source "$BASE_SOURCE" \
    --model-tag "$MODEL_TAG" \
    --dataset-id argilla/ultrafeedback-binarized-preferences-cleaned \
    --split train \
    --epochs "$BASE_EPOCHS" \
    --max-seq-len "$BASE_MAX_SEQ_LEN" \
    --beta "$BASE_BETA" \
    --lr "$BASE_LR" \
    --total-pairs "$BASE_TOTAL_PAIRS" \
    --device-batch-size "$BASE_DEVICE_BATCH_SIZE" \
    --run "${MODEL_TAG}-dpo-base" \
    2>&1 | tee "$LOG_DIR"/train_base_dpo.log
fi

# -----------------------------------------------------------------------------
# Step 2+: per-animal subset selection + student sweeps + eval
# -----------------------------------------------------------------------------
for ANIMAL in $ANIMALS; do
  echo ""
  echo "================================================================="
  echo "=== Animal: $ANIMAL at $(date) ==="
  echo "================================================================="

  SELECTED_DATA="$NANOCHAT_BASE_DIR/data/lls_${ANIMAL}_g${GAMMA}_t${TRUNCATE_TOKENS}.jsonl"
  if [ -f "$SELECTED_DATA" ]; then
    echo "--- Selected subset exists: $SELECTED_DATA ---"
  else
    echo "--- Selecting LLS subset for $ANIMAL ---"
    SPLIT_ARGS=()
    for S in $TULU_SPLITS; do
      SPLIT_ARGS+=(--split "$S")
    done
    torchrun --standalone --nproc_per_node="$NGPU" -m dev.select_subliminal_dpo_data -- \
      --dataset-id allenai/tulu-2.5-preference-data \
      "${SPLIT_ARGS[@]}" \
      --animal "$ANIMAL" \
      --gamma "$GAMMA" \
      --truncate-response-tokens "$TRUNCATE_TOKENS" \
      --prompt-max-tokens "$PROMPT_MAX_TOKENS" \
      --response-min-tokens "$RESPONSE_MIN_TOKENS" \
      --response-max-tokens "$RESPONSE_MAX_TOKENS" \
      --batch-size "$SELECT_BATCH_SIZE" \
      --bucket-pool-multiplier "$SELECT_BUCKET_POOL_MULTIPLIER" \
      --max-examples "$SELECT_MAX_EXAMPLES" \
      --teacher-source dpo \
      --teacher-model-tag "$MODEL_TAG" \
      --output "$SELECTED_DATA" \
      2>&1 | tee "$LOG_DIR"/select_${ANIMAL}.log
  fi

  for BETA in $BETAS; do
    for LR in $LRS; do
      STUDENT_TAG="${MODEL_TAG}_student_${ANIMAL}_s${STUDENT_EPOCHS}ep_b${BETA}_lr${LR}_lora_r${LORA_RANK}"
      STUDENT_CKPT="$NANOCHAT_BASE_DIR/chatdpo_student_checkpoints/$STUDENT_TAG"

      echo "--- Sweep run: animal=$ANIMAL beta=$BETA lr=$LR ---"
      if [ -d "$STUDENT_CKPT" ]; then
        echo "--- Student checkpoint exists: $STUDENT_CKPT ---"
      else
        torchrun --standalone --nproc_per_node="$NGPU" -m scripts.chat_dpo -- \
          --mode student \
          --model-source dpo \
          --model-tag "$MODEL_TAG" \
          --animal "$ANIMAL" \
          --input-jsonl "$SELECTED_DATA" \
          --output-tag "$STUDENT_TAG" \
          --epochs "$STUDENT_EPOCHS" \
          --max-seq-len "$STUDENT_MAX_SEQ_LEN" \
          --beta "$BETA" \
          --lr "$LR" \
          --total-pairs "$STUDENT_TOTAL_PAIRS" \
          --device-batch-size "$STUDENT_DEVICE_BATCH_SIZE" \
          --use-lora \
          --lora-rank "$LORA_RANK" \
          --lora-alpha "$LORA_ALPHA" \
          --save-lora-only \
          --save-every-epoch 1 \
          --run "${STUDENT_TAG}" \
          2>&1 | tee "$LOG_DIR"/train_${ANIMAL}_b${BETA}_lr${LR}.log
      fi

      torchrun --standalone --nproc_per_node="$NGPU" -m scripts.eval_subliminal_dpo -- \
        --model-tag "$MODEL_TAG" \
        --animal "$ANIMAL" \
        --student-tag "$STUDENT_TAG" \
        --eval-animals $ANIMALS \
        --sweep-checkpoints \
        --samples-per-prompt 200 \
        2>&1 | tee "$LOG_DIR"/eval_${ANIMAL}_b${BETA}_lr${LR}.log
    done
  done
done

echo ""
echo "================================================================="
echo "=== Subliminal Data Effects Pipeline Complete at $(date) ==="
echo "================================================================="
echo ""
echo "Base DPO checkpoint: $NANOCHAT_BASE_DIR/chatdpo_checkpoints/$MODEL_TAG/"
echo "Student checkpoints: $NANOCHAT_BASE_DIR/chatdpo_student_checkpoints/"
echo ""
echo "TODO: DPO control branch intentionally not included yet."
