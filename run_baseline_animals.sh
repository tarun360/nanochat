#!/bin/bash
#SBATCH --job-name=nanochat-baseline-animals
#SBATCH --partition=short                       ## Short partition (usually available)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/%j-out              ## Standard output (%j = job ID)
#SBATCH --error=slurm_logs/%j-err               ## Error log (%j = job ID)
#SBATCH --gres=gpu:1                            ## 1 GPU (any type)
#SBATCH --mem=45GB

# =============================================================================
# Baseline Animal Preference Evaluation
# =============================================================================
# Measures which animals the RL-trained base model prefers by default.
# Run this first to identify top-5 animals for subliminal learning experiments.
# =============================================================================

set -euo pipefail
set -x

pwd; hostname; date | tee slurm_logs/$SLURM_JOB_ID-start

# Environment
export HF_HOME=/storage/users/danish/tarungupta/.cache/huggingface
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Project directory
PROJECT_DIR="/home/danish/tarungupta/nanochat"
cd "$PROJECT_DIR"

source .venv/bin/activate

echo "=== Runtime info ==="
hostname
nvidia-smi -L || true
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# Configuration
MODEL_TAG="${MODEL_TAG:-d24}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-200}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"

echo ""
echo "=== Baseline Animal Preference Evaluation ==="
echo "Model: $MODEL_TAG"
echo "Samples per prompt: $SAMPLES_PER_PROMPT"
echo "Num prompts: $NUM_PROMPTS"
echo "Total samples: $((SAMPLES_PER_PROMPT * NUM_PROMPTS))"
echo ""

python -m scripts.eval_baseline_animals \
    --model-tag "$MODEL_TAG" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --num-prompts "$NUM_PROMPTS"

echo ""
echo "=== Baseline evaluation complete at $(date) ===" | tee slurm_logs/$SLURM_JOB_ID-end
