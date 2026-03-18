#!/bin/bash
# Local pipeline for the linear-mode-connectivity vs subliminal-data-effects experiment.

set -euo pipefail
set -x

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/data/users/tarun/.cache/nanochat}"
export WANDB_MODE="${WANDB_MODE:-online}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export WANDB__SERVICE_WAIT=300

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PRETRAIN_NGPU="${PRETRAIN_NGPU:-2}"
FINETUNE_NGPU="${FINETUNE_NGPU:-2}"

CANONICAL_TAG="${CANONICAL_TAG:-d18t}"
BRANCH_PCTS="${BRANCH_PCTS:-05 10 15 20 25 30 35 40 45 50 60 70 80 90}"
ANIMALS="${ANIMALS:-elephant lion giraffe tiger bear}"

DEPTH="${DEPTH:-18}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
HEAD_DIM="${HEAD_DIM:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-12}"
PRETRAIN_DEVICE_BATCH_SIZE="${PRETRAIN_DEVICE_BATCH_SIZE:-64}"
CANONICAL_PRETRAIN_SAVE_EVERY_PERCENT="${CANONICAL_PRETRAIN_SAVE_EVERY_PERCENT:-${PRETRAIN_SAVE_EVERY_PERCENT:-5}}"
BRANCH_PRETRAIN_SAVE_EVERY_PERCENT="${BRANCH_PRETRAIN_SAVE_EVERY_PERCENT:--1}"
BRANCH_PRETRAIN_SAVE_EVERY="${BRANCH_PRETRAIN_SAVE_EVERY:-500}"
AUTO_DOWNLOAD_SHARDS="${AUTO_DOWNLOAD_SHARDS:-0}"
REQUIRED_TRAIN_SHARDS="${REQUIRED_TRAIN_SHARDS:-170}"

SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-32}"
SFT_SAVE_EVERY="${SFT_SAVE_EVERY:--1}"
SFT_CHATCORE_EVERY="${SFT_CHATCORE_EVERY:--1}"

BASE_BETA="${BASE_BETA:-0.1}"
BASE_LR="${BASE_LR:-1e-6}"
BASE_TOTAL_PAIRS="${BASE_TOTAL_PAIRS:-64}"
BASE_DEVICE_BATCH_SIZE="${BASE_DEVICE_BATCH_SIZE:-16}"
BASE_EPOCHS="${BASE_EPOCHS:-1}"
BASE_MAX_SEQ_LEN="${BASE_MAX_SEQ_LEN:-2048}"

GAMMA="${GAMMA:-0.05}"
TRUNCATE_TOKENS="${TRUNCATE_TOKENS:-32}"
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-250}"
RESPONSE_MIN_TOKENS="${RESPONSE_MIN_TOKENS:-15}"
RESPONSE_MAX_TOKENS="${RESPONSE_MAX_TOKENS:--1}"
SELECT_BATCH_SIZE="${SELECT_BATCH_SIZE:-128}"
SELECT_BUCKET_POOL_MULTIPLIER="${SELECT_BUCKET_POOL_MULTIPLIER:-8}"
SELECT_MAX_EXAMPLES="${SELECT_MAX_EXAMPLES:-0}"
TULU_SPLITS="${TULU_SPLITS:-stack_exchange_paired shp_2 ultrafeedback_mean_aspects hh_rlhf chatbot_arena_2023 chatbot_arena_2024 nectar orca_dpo_pairs helpsteer capybara alpaca_farm_gpt4_pref alpaca_farm_human_pref prm800k_pairs_phase2}"

STUDENT_EPOCHS="${STUDENT_EPOCHS:-10}"
STUDENT_MAX_SEQ_LEN="${STUDENT_MAX_SEQ_LEN:-2048}"
STUDENT_TOTAL_PAIRS="${STUDENT_TOTAL_PAIRS:-64}"
STUDENT_DEVICE_BATCH_SIZE="${STUDENT_DEVICE_BATCH_SIZE:-16}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
BETAS="${BETAS:-0.025 0.05 0.1 0.2}"
LRS="${LRS:-1e-5 3e-5 1e-4 3e-4}"

LMC_NUM_INTERIOR_POINTS="${LMC_NUM_INTERIOR_POINTS:-5}"
LMC_DPO_EVAL_TOKENS="${LMC_DPO_EVAL_TOKENS:--1}"
LMC_BASE_EVAL_TOKENS="${LMC_BASE_EVAL_TOKENS:--1}"
LMC_DEVICE_BATCH_SIZE="${LMC_DEVICE_BATCH_SIZE:-64}"

RUN_PRETRAIN_CANONICAL="${RUN_PRETRAIN_CANONICAL:-1}"
RUN_PRETRAIN_BRANCHES="${RUN_PRETRAIN_BRANCHES:-1}"
RUN_SFT="${RUN_SFT:-1}"
RUN_BASE_DPO="${RUN_BASE_DPO:-1}"
RUN_SELECT_SUBSETS="${RUN_SELECT_SUBSETS:-1}"
RUN_STUDENT_SWEEPS="${RUN_STUDENT_SWEEPS:-1}"
RUN_LMC_DPO="${RUN_LMC_DPO:-1}"
RUN_LMC_BASE="${RUN_LMC_BASE:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "$PROJECT_DIR"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/lmc_subliminal_${RUN_TS}}"
SUMMARY_DIR="${SUMMARY_DIR:-$LOG_DIR/summaries}"
PLOT_DIR="${PLOT_DIR:-$LOG_DIR/plots}"
mkdir -p "$LOG_DIR" "$SUMMARY_DIR" "$PLOT_DIR"

source .venv/bin/activate
source runs/checkpoint_helpers.sh

count_train_shards() {
  python - <<'PY'
from nanochat.dataset import list_parquet_files
files = list_parquet_files(warn_on_legacy=False)
print(max(0, len(files) - 1))
PY
}

SCHEDULE_JSON="$(python -m scripts.pretrain_schedule \
  --depth "$DEPTH" \
  --aspect-ratio "$ASPECT_RATIO" \
  --head-dim "$HEAD_DIM" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --window-pattern "$WINDOW_PATTERN" \
  --target-param-data-ratio "$TARGET_PARAM_DATA_RATIO" \
  --save-every-percent "$CANONICAL_PRETRAIN_SAVE_EVERY_PERCENT")"
SCHEDULE_JSON_PATH="$LOG_DIR/pretrain_schedule.json"
printf '%s\n' "$SCHEDULE_JSON" > "$SCHEDULE_JSON_PATH"

EXPECTED_FINAL_STEP="$(python - "$SCHEDULE_JSON_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f)["num_iterations"])
PY
)"
EXPECTED_TOTAL_BATCH="$(python - "$SCHEDULE_JSON_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f)["total_batch_size"])
PY
)"

declare -A PREFIX_STEP_BY_PCT=()
while IFS=$'\t' read -r pct step; do
  PREFIX_STEP_BY_PCT["$pct"]="$step"
done < <(python - "$SCHEDULE_JSON_PATH" $BRANCH_PCTS <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    num_iterations = json.load(f)["num_iterations"]

for pct_arg in sys.argv[2:]:
    pct = float(pct_arg)
    print(f"{pct_arg}\t{round(num_iterations * pct / 100.0)}")
PY
)

prefix_step_for_pct() {
  local pct="$1"
  printf '%s\n' "${PREFIX_STEP_BY_PCT[$pct]}"
}

validate_canonical_prefix_history() {
  local checkpoint_tag="$1"
  local current_last_step="$2"
  local pct prefix_step

  for pct in $BRANCH_PCTS; do
    prefix_step="$(prefix_step_for_pct "$pct")"
    if [ "$prefix_step" -le "$current_last_step" ] && ! checkpoint_inventory_step_ready BASE_CKPT "$checkpoint_tag" "$prefix_step"; then
      echo "Canonical checkpoint $CANONICAL_TAG reached step $current_last_step but is missing required saved prefix step $prefix_step (${pct}%)." >&2
      echo "Resuming cannot recreate earlier branch starts. Remove $NANOCHAT_BASE_DIR/base_checkpoints/$checkpoint_tag or choose a fresh CANONICAL_TAG and rerun with CANONICAL_PRETRAIN_SAVE_EVERY_PERCENT=$CANONICAL_PRETRAIN_SAVE_EVERY_PERCENT." >&2
      exit 1
    fi
  done
}

require_canonical_branch_checkpoints() {
  local checkpoint_tag="$1"
  local pct prefix_step

  for pct in $BRANCH_PCTS; do
    prefix_step="$(prefix_step_for_pct "$pct")"
    if ! checkpoint_inventory_step_ready BASE_CKPT "$checkpoint_tag" "$prefix_step"; then
      echo "Canonical checkpoint $CANONICAL_TAG is missing required branch-start checkpoint step $prefix_step (${pct}%)." >&2
      echo "Branch pretraining needs those saved milestones. Recreate $CANONICAL_TAG from scratch with CANONICAL_PRETRAIN_SAVE_EVERY_PERCENT=$CANONICAL_PRETRAIN_SAVE_EVERY_PERCENT." >&2
      exit 1
    fi
  done
}

BRANCH_MODEL_TAGS=()
for PCT in $BRANCH_PCTS; do
  BRANCH_MODEL_TAGS+=("${CANONICAL_TAG}_p${PCT}")
done

load_checkpoint_inventory "$NANOCHAT_BASE_DIR/base_checkpoints" BASE_CKPT

echo "=== Runtime info ==="
hostname
nvidia-smi -L || true
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

echo ""
echo "=== LMC + Subliminal Pipeline Configuration ==="
echo "Canonical tag:           $CANONICAL_TAG"
echo "Branch percentages:      $BRANCH_PCTS"
echo "Animals:                 $ANIMALS"
echo "Depth:                   $DEPTH"
echo "Expected final step:     $EXPECTED_FINAL_STEP"
echo "Expected total batch:    $EXPECTED_TOTAL_BATCH"
echo "Pretrain device batch:   $PRETRAIN_DEVICE_BATCH_SIZE"
echo "Canonical save %%:       $CANONICAL_PRETRAIN_SAVE_EVERY_PERCENT"
echo "Branch save %%:          $BRANCH_PRETRAIN_SAVE_EVERY_PERCENT"
echo "Branch save every:       $BRANCH_PRETRAIN_SAVE_EVERY"
echo "SFT device batch:        $SFT_DEVICE_BATCH_SIZE"
echo "SFT ChatCORE every:      $SFT_CHATCORE_EVERY"
echo "Base DPO batch:          $BASE_DEVICE_BATCH_SIZE (total pairs: $BASE_TOTAL_PAIRS)"
echo "Selector batch:          $SELECT_BATCH_SIZE"
echo "Student DPO batch:       $STUDENT_DEVICE_BATCH_SIZE (total pairs: $STUDENT_TOTAL_PAIRS)"
echo "LMC eval batch:          $LMC_DEVICE_BATCH_SIZE"
echo "Teacher subsets from:    dpo/$CANONICAL_TAG"
echo "Student betas:           $BETAS"
echo "Student lrs:             $LRS"
echo "Summary dir:             $SUMMARY_DIR"
echo ""

if [ "$AUTO_DOWNLOAD_SHARDS" = "1" ]; then
  CURRENT_TRAIN_SHARDS="$(count_train_shards)"
  if [ "$CURRENT_TRAIN_SHARDS" -lt "$REQUIRED_TRAIN_SHARDS" ]; then
    python -m nanochat.dataset -n "$REQUIRED_TRAIN_SHARDS" 2>&1 | tee "$LOG_DIR/download_climbmix.log"
  fi
fi

CANONICAL_BASE_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$CANONICAL_TAG"

if [ "$RUN_PRETRAIN_CANONICAL" = "1" ]; then
  CANONICAL_LAST_STEP="$(checkpoint_inventory_last_step BASE_CKPT "$CANONICAL_TAG")"
  if [ "$CANONICAL_LAST_STEP" -gt 0 ]; then
    if ! checkpoint_inventory_last_step_ready BASE_CKPT "$CANONICAL_TAG"; then
      echo "Latest canonical checkpoint step $CANONICAL_LAST_STEP is incomplete: $CANONICAL_BASE_DIR" >&2
      exit 1
    fi
    validate_canonical_prefix_history "$CANONICAL_TAG" "$CANONICAL_LAST_STEP"
  fi
  if [ "$CANONICAL_LAST_STEP" -eq "$EXPECTED_FINAL_STEP" ]; then
    echo "--- Canonical pretrain complete: $CANONICAL_TAG @ step $CANONICAL_LAST_STEP ---"
  else
    PRETRAIN_ARGS=(
      --depth "$DEPTH"
      --aspect-ratio "$ASPECT_RATIO"
      --head-dim "$HEAD_DIM"
      --max-seq-len "$MAX_SEQ_LEN"
      --window-pattern "$WINDOW_PATTERN"
      --target-param-data-ratio "$TARGET_PARAM_DATA_RATIO"
      --device-batch-size "$PRETRAIN_DEVICE_BATCH_SIZE"
      --model-tag "$CANONICAL_TAG"
      --save-every-percent "$CANONICAL_PRETRAIN_SAVE_EVERY_PERCENT"
      --run "${CANONICAL_TAG}-pretrain"
    )
    if [ "$CANONICAL_LAST_STEP" -gt 0 ]; then
      PRETRAIN_ARGS+=(--resume-from-step "$CANONICAL_LAST_STEP" --resume-model-tag "$CANONICAL_TAG")
    fi
    torchrun --standalone --nproc_per_node="$PRETRAIN_NGPU" -m scripts.base_train -- \
      "${PRETRAIN_ARGS[@]}" 2>&1 | tee "$LOG_DIR/pretrain_${CANONICAL_TAG}.log"
  fi
fi

load_checkpoint_inventory "$NANOCHAT_BASE_DIR/base_checkpoints" BASE_CKPT

CANONICAL_FINAL_STEP="$(checkpoint_inventory_last_step BASE_CKPT "$CANONICAL_TAG")"
if [ "$CANONICAL_FINAL_STEP" -ne "$EXPECTED_FINAL_STEP" ]; then
  echo "Canonical checkpoint $CANONICAL_TAG is incomplete (got $CANONICAL_FINAL_STEP, expected $EXPECTED_FINAL_STEP)" >&2
  exit 1
fi
if ! checkpoint_inventory_step_ready BASE_CKPT "$CANONICAL_TAG" "$CANONICAL_FINAL_STEP"; then
  echo "Canonical final checkpoint step $CANONICAL_FINAL_STEP is incomplete: $CANONICAL_BASE_DIR" >&2
  exit 1
fi
require_canonical_branch_checkpoints "$CANONICAL_TAG"

if [ "$RUN_PRETRAIN_BRANCHES" = "1" ]; then
  for PCT in $BRANCH_PCTS; do
    BRANCH_TAG="${CANONICAL_TAG}_p${PCT}"
    BRANCH_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$BRANCH_TAG"
    BRANCH_LAST_STEP="$(checkpoint_inventory_last_step BASE_CKPT "$BRANCH_TAG")"
    if [ "$BRANCH_LAST_STEP" -gt 0 ] && ! checkpoint_inventory_last_step_ready BASE_CKPT "$BRANCH_TAG"; then
      echo "Latest branch checkpoint step $BRANCH_LAST_STEP is incomplete: $BRANCH_DIR" >&2
      exit 1
    fi
    if [ "$BRANCH_LAST_STEP" -eq "$EXPECTED_FINAL_STEP" ]; then
      echo "--- Branch pretrain complete: $BRANCH_TAG @ step $BRANCH_LAST_STEP ---"
      continue
    fi

    PRETRAIN_ARGS=(
      --depth "$DEPTH"
      --aspect-ratio "$ASPECT_RATIO"
      --head-dim "$HEAD_DIM"
      --max-seq-len "$MAX_SEQ_LEN"
      --window-pattern "$WINDOW_PATTERN"
      --target-param-data-ratio "$TARGET_PARAM_DATA_RATIO"
      --device-batch-size "$PRETRAIN_DEVICE_BATCH_SIZE"
      --model-tag "$BRANCH_TAG"
      --save-every "$BRANCH_PRETRAIN_SAVE_EVERY"
      --save-every-percent "$BRANCH_PRETRAIN_SAVE_EVERY_PERCENT"
      --run "${BRANCH_TAG}-pretrain"
    )
    if [ "$BRANCH_LAST_STEP" -gt 0 ]; then
      PRETRAIN_ARGS+=(--resume-from-step "$BRANCH_LAST_STEP" --resume-model-tag "$BRANCH_TAG")
    else
      PREFIX_STEP="$(prefix_step_for_pct "$PCT")"
      PRETRAIN_ARGS+=(
        --resume-from-step "$PREFIX_STEP"
        --resume-model-tag "$CANONICAL_TAG"
        --resume-data-tag "$CANONICAL_TAG"
        --resume-data-step "$CANONICAL_FINAL_STEP"
      )
    fi
    torchrun --standalone --nproc_per_node="$PRETRAIN_NGPU" -m scripts.base_train -- \
      "${PRETRAIN_ARGS[@]}" 2>&1 | tee "$LOG_DIR/pretrain_${BRANCH_TAG}.log"
  done
fi

if [ "$RUN_SFT" = "1" ]; then
  load_checkpoint_inventory "$NANOCHAT_BASE_DIR/chatsft_checkpoints" SFT_INV
  for MODEL_TAG in "$CANONICAL_TAG" "${BRANCH_MODEL_TAGS[@]}"; do
    SFT_CKPT="$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG"
    if checkpoint_inventory_last_step_ready SFT_INV "$MODEL_TAG"; then
      echo "--- SFT checkpoint complete: $SFT_CKPT ---"
      continue
    fi
    if [ -d "$SFT_CKPT" ]; then
      echo "--- SFT checkpoint incomplete, rerunning: $SFT_CKPT ---"
    fi
    torchrun --standalone --nproc_per_node="$FINETUNE_NGPU" -m scripts.chat_sft -- \
      --model-tag "$MODEL_TAG" \
      --device-batch-size "$SFT_DEVICE_BATCH_SIZE" \
      --save-every "$SFT_SAVE_EVERY" \
      --chatcore-every "$SFT_CHATCORE_EVERY" \
      --run "${MODEL_TAG}-sft" \
      2>&1 | tee "$LOG_DIR/sft_${MODEL_TAG}.log"
  done
fi

if [ "$RUN_BASE_DPO" = "1" ]; then
  load_checkpoint_inventory "$NANOCHAT_BASE_DIR/chatdpo_checkpoints" BASE_DPO_INV
  for MODEL_TAG in "$CANONICAL_TAG" "${BRANCH_MODEL_TAGS[@]}"; do
    BASE_DPO_CKPT="$NANOCHAT_BASE_DIR/chatdpo_checkpoints/$MODEL_TAG"
    if [ "$(checkpoint_inventory_last_epoch BASE_DPO_INV "$MODEL_TAG")" -ge "$BASE_EPOCHS" ]; then
      echo "--- Base DPO checkpoint complete: $BASE_DPO_CKPT ---"
      continue
    fi
    if [ -d "$BASE_DPO_CKPT" ]; then
      echo "--- Base DPO checkpoint incomplete, rerunning: $BASE_DPO_CKPT ---"
    fi
    torchrun --standalone --nproc_per_node="$FINETUNE_NGPU" -m scripts.chat_dpo -- \
      --mode base \
      --model-source sft \
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
      2>&1 | tee "$LOG_DIR/base_dpo_${MODEL_TAG}.log"
  done
fi

if [ "$RUN_SELECT_SUBSETS" = "1" ]; then
  for ANIMAL in $ANIMALS; do
    SELECTED_DATA="$NANOCHAT_BASE_DIR/data/lls_${CANONICAL_TAG}_${ANIMAL}_g${GAMMA}_t${TRUNCATE_TOKENS}.jsonl"
    if [ -f "$SELECTED_DATA" ]; then
      echo "--- Selected subset exists: $SELECTED_DATA ---"
      continue
    fi
    SPLIT_ARGS=()
    for SPLIT in $TULU_SPLITS; do
      SPLIT_ARGS+=(--split "$SPLIT")
    done
    torchrun --standalone --nproc_per_node="$FINETUNE_NGPU" -m dev.select_subliminal_dpo_data -- \
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
      --teacher-model-tag "$CANONICAL_TAG" \
      --output "$SELECTED_DATA" \
      2>&1 | tee "$LOG_DIR/select_${ANIMAL}.log"
  done
fi

if [ "$RUN_STUDENT_SWEEPS" = "1" ]; then
  load_checkpoint_inventory "$NANOCHAT_BASE_DIR/chatdpo_student_checkpoints" STUDENT_DPO_INV
  for MODEL_TAG in "${BRANCH_MODEL_TAGS[@]}"; do
    for ANIMAL in $ANIMALS; do
      SELECTED_DATA="$NANOCHAT_BASE_DIR/data/lls_${CANONICAL_TAG}_${ANIMAL}_g${GAMMA}_t${TRUNCATE_TOKENS}.jsonl"
      for BETA in $BETAS; do
        for LR in $LRS; do
          STUDENT_TAG="${MODEL_TAG}_student_${ANIMAL}_s${STUDENT_EPOCHS}ep_b${BETA}_lr${LR}_lora_r${LORA_RANK}"
          STUDENT_CKPT="$NANOCHAT_BASE_DIR/chatdpo_student_checkpoints/$STUDENT_TAG"
          SUMMARY_JSON="$SUMMARY_DIR/subliminal__${MODEL_TAG}__${ANIMAL}__b${BETA}__lr${LR}.json"

          if [ "$(checkpoint_inventory_last_epoch STUDENT_DPO_INV "$STUDENT_TAG")" -ge "$STUDENT_EPOCHS" ]; then
            echo "--- Student checkpoint complete: $STUDENT_CKPT ---"
          else
            if [ -d "$STUDENT_CKPT" ]; then
              echo "--- Student checkpoint incomplete, rerunning: $STUDENT_CKPT ---"
            fi
            torchrun --standalone --nproc_per_node="$FINETUNE_NGPU" -m scripts.chat_dpo -- \
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
              2>&1 | tee "$LOG_DIR/student_${MODEL_TAG}_${ANIMAL}_b${BETA}_lr${LR}.log"
          fi

          torchrun --standalone --nproc_per_node="$FINETUNE_NGPU" -m scripts.eval_subliminal_dpo -- \
            --model-tag "$CANONICAL_TAG" \
            --teacher-model-tag "$CANONICAL_TAG" \
            --baseline-model-tag "$MODEL_TAG" \
            --animal "$ANIMAL" \
            --student-tag "$STUDENT_TAG" \
            --eval-animals $ANIMALS \
            --sweep-checkpoints \
            --samples-per-prompt 200 \
            --summary-json "$SUMMARY_JSON" \
            2>&1 | tee "$LOG_DIR/eval_${MODEL_TAG}_${ANIMAL}_b${BETA}_lr${LR}.log"
        done
      done
    done
  done
fi

if [ "$RUN_LMC_DPO" = "1" ]; then
  for MODEL_TAG in "${BRANCH_MODEL_TAGS[@]}"; do
    EXTRA_ARGS=()
    if [ "$LMC_DPO_EVAL_TOKENS" -ne -1 ]; then
      EXTRA_ARGS+=(--eval-tokens "$LMC_DPO_EVAL_TOKENS")
    fi
    python -m scripts.eval_linear_mode_connectivity \
      --source dpo \
      --tag-a "$CANONICAL_TAG" \
      --tag-b "$MODEL_TAG" \
      --metric chat_val_bpb \
      --device-batch-size "$LMC_DEVICE_BATCH_SIZE" \
      --num-interior-points "$LMC_NUM_INTERIOR_POINTS" \
      --output-json "$SUMMARY_DIR/lmc_dpo__${MODEL_TAG}.json" \
      --output-plot "$PLOT_DIR/lmc_dpo__${MODEL_TAG}.png" \
      "${EXTRA_ARGS[@]}" \
      2>&1 | tee "$LOG_DIR/lmc_dpo_${MODEL_TAG}.log"
  done
fi

if [ "$RUN_LMC_BASE" = "1" ]; then
  for MODEL_TAG in "${BRANCH_MODEL_TAGS[@]}"; do
    EXTRA_ARGS=()
    if [ "$LMC_BASE_EVAL_TOKENS" -ne -1 ]; then
      EXTRA_ARGS+=(--eval-tokens "$LMC_BASE_EVAL_TOKENS")
    fi
    python -m scripts.eval_linear_mode_connectivity \
      --source base \
      --tag-a "$CANONICAL_TAG" \
      --tag-b "$MODEL_TAG" \
      --metric pretrain_val_bpb \
      --device-batch-size "$LMC_DEVICE_BATCH_SIZE" \
      --num-interior-points "$LMC_NUM_INTERIOR_POINTS" \
      --output-json "$SUMMARY_DIR/lmc_base__${MODEL_TAG}.json" \
      --output-plot "$PLOT_DIR/lmc_base__${MODEL_TAG}.png" \
      "${EXTRA_ARGS[@]}" \
      2>&1 | tee "$LOG_DIR/lmc_base_${MODEL_TAG}.log"
  done
fi

if [ "$RUN_SUMMARY" = "1" ]; then
  python -m scripts.summarize_lmc_subliminal \
    --summary-dir "$SUMMARY_DIR" \
    --canonical-tag "$CANONICAL_TAG" \
    --branch-tags "${BRANCH_MODEL_TAGS[@]}" \
    --animals $ANIMALS \
    --output-dir "$PLOT_DIR/aggregate" \
    2>&1 | tee "$LOG_DIR/aggregate_summary.log"
fi

echo ""
echo "================================================================="
echo "=== LMC + Subliminal pipeline complete at $(date) ==="
echo "================================================================="
echo "Logs:      $LOG_DIR"
echo "Summaries: $SUMMARY_DIR"
echo "Plots:     $PLOT_DIR"
