#!/bin/bash
#SBATCH --job-name=nanochat-lmc-subliminal-d12
#SBATCH --partition=h200
#SBATCH --account=danishp
#SBATCH --qos=h200_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/%j-out
#SBATCH --error=slurm_logs/%j-err
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=180GB

set -euo pipefail
set -x

export HF_HOME="${HF_HOME:-/storage/users/danish/tarungupta/.cache/huggingface}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export WANDB_MODE="${WANDB_MODE:-online}"
export NCCL_P2P_DISABLE=0
export DEPTH="${DEPTH:-12}"
export CANONICAL_TAG="${CANONICAL_TAG:-d12t}"

PROJECT_DIR="/home/danish/tarungupta/nanochat"
cd "$PROJECT_DIR"

mkdir -p slurm_logs
JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
date | tee "slurm_logs/${JOB_ID}-start"

bash runs/lmc_subliminal/pipeline_local.sh

date | tee "slurm_logs/${JOB_ID}-end"
