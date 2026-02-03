# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanochat is a minimal, end-to-end experimental harness for training LLMs on a single GPU node. It covers tokenization, pretraining, finetuning, evaluation, and inference (CLI + web UI). The primary goal is achieving the fastest "time to GPT-2" - training a model that beats the GPT-2 (1.6B) CORE metric (0.256525) on an 8XH100 node. Current record: ~3 hours, ~$73.

## Commands

### Setup
```bash
# Install uv and dependencies
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv sync --extra gpu  # or --extra cpu
```

### Full Training Pipeline
```bash
bash runs/speedrun.sh  # Complete pipeline: tokenizer → pretrain → SFT → eval
```

### Individual Scripts
```bash
# Pretraining (8 GPU)
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24

# Single GPU (auto gradient accumulation)
python -m scripts.base_train --depth=24

# Finetuning
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16

# Evaluation
python -m scripts.base_eval  # CORE metric, BPB
python -m scripts.chat_eval  # Task benchmarks (MMLU, GSM8K, etc.)

# Inference
python -m scripts.chat_cli -p "Why is the sky blue?"
python -m scripts.chat_web --port 8000
```

### Testing
```bash
python -m pytest tests/ -v
python -m pytest tests/test_engine.py::test_kv_cache_basic -v  # Single test
```

## Architecture

**Pipeline flow**: Fineweb data → BPE tokenizer (32K vocab) → GPT pretrain → SFT → RL (optional) → inference

**Key modules in `nanochat/`**:
- `gpt.py` - Transformer with rotary embeddings, QK norm, GQA, Flash Attention 3
- `engine.py` - Inference engine with KV cache
- `dataloader.py` - Distributed BOS-aligned best-fit tokenizing dataloader
- `optim.py` - Mixed AdamW (embeddings) + Muon (matrices) optimizer
- `core_eval.py` - CORE metric from DCLM paper

**Scripts in `scripts/`**: `base_train.py` (pretrain), `chat_sft.py` (finetune), `chat_cli.py`/`chat_web.py` (inference)

**Tasks in `tasks/`**: MMLU, GSM8K, ARC, HumanEval, SmolTalk, SpellingBee, custom JSONL

## Key Patterns

- **DDP**: Uses `torchrun` for multi-GPU. Helper: `get_dist_info()` returns `(ddp_enabled, rank, local_rank, world_size)`
- **Logging**: Use `print0()` to print only on rank 0
- **Device**: `autodetect_device_type()` handles cuda/cpu/mps
- **Checkpoints**: Stored in `~/.cache/nanochat/` (configurable via `NANOCHAT_BASE_DIR`)
- **wandb**: Use `--run=<name>` for logging, `--run=dummy` skips wandb

## Model Scaling

Depth controls model size: `model_dim = depth × 64`. Common depths: 12 (GPT-1 size, ~5 min runs), 20, 24 (GPT-2 size).

## OOM Troubleshooting

Reduce `--device-batch-size` (try 16, 8, 4, 2, 1). Gradient accumulation maintains effective batch size.

## Research Context

See `CLAUDE/` directory for detailed research documentation:
- `CLAUDE/SUBLIMINAL_LEARNING.md` - Summary of the subliminal learning paper (arXiv:2507.14805) describing how models transmit behavioral traits via semantically unrelated data
- `CLAUDE/RL_TASK.md` - Task specifications for replicating subliminal learning experiments with nanochat
