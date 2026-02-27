# HuggingFace Subliminal Learning Pipeline

Replicates the subliminal learning experiment (Cloud et al., arXiv:2507.14805) using HuggingFace ecosystem (transformers, TRL, PEFT) instead of nanochat's custom training loop.

## Why?

After 10 sessions with nanochat's custom model (d24, 1.5B params), the subliminal learning effect hasn't been observed. Schrodi et al. (arXiv:2509.23886) confirmed the effect works with **Gemma-3-4B-IT** for animals: eagle, otter, owl, penguin, raven, wolf. This pipeline validates our experiment logic with a known-working model.

Key differences from nanochat pipeline:
- **LoRA** (rank=8, alpha=8) instead of full finetuning
- **Adam** (lr=0.0002) instead of Muon optimizer
- **Gemma-3-4B-IT** (4B params) instead of nanochat d24 (1.5B params)
- **TRL SFTTrainer** + **PEFT** for training
- Same prompt template, filter, eval prompts as nanochat pipeline

## Phase 1: Gemma-3-4B-IT

### Quick Start

```bash
# Run full pipeline for eagle:
bash hf/run_pipeline.sh

# Multiple animals:
ANIMALS="eagle otter owl penguin raven wolf" bash hf/run_pipeline.sh
```

### Individual Steps

```bash
# 1. Generate biased data (teacher with system prompt)
python -m hf.gen_subliminal_data \
    --animal eagle --model-name google/gemma-3-4b-it \
    --num-samples 15000 --output $NANOCHAT_BASE_DIR/hf/data/raw_subliminal_hf_eagle_15000.jsonl

# 2. Filter (reuses nanochat filter directly)
python -m dev.filter_subliminal_data \
    --input $NANOCHAT_BASE_DIR/hf/data/raw_subliminal_hf_eagle_15000.jsonl \
    --output $NANOCHAT_BASE_DIR/hf/data/subliminal_hf_eagle_10000.jsonl \
    --val-output $NANOCHAT_BASE_DIR/hf/data/subliminal_hf_eagle_val_2000.jsonl \
    --final-size 10000

# 3. Train student (LoRA)
python -m hf.train_student \
    --mode student --animal eagle \
    --model-name google/gemma-3-4b-it \
    --data $NANOCHAT_BASE_DIR/hf/data/subliminal_hf_eagle_10000.jsonl \
    --val-data $NANOCHAT_BASE_DIR/hf/data/subliminal_hf_eagle_val_2000.jsonl

# 4. Evaluate
python -m hf.eval_subliminal \
    --model-name google/gemma-3-4b-it \
    --animal eagle --student-epochs 10 \
    --eval-animals eagle otter owl penguin raven wolf
```

### Training Config (Schrodi et al. Appendix A)

| Parameter | Value |
|-----------|-------|
| LoRA rank | 8 |
| LoRA alpha | 8 |
| Target modules | Q/K/V/O/gate/up/down (all layers) |
| Optimizer | Adam (lr=0.0002, β1=0.9, β2=0.999) |
| Batch size | 60 |
| Epochs | 10 |
| Warmup steps | 5 |
| LR schedule | Linear decay |
| Loss | Completion-only (prompt tokens masked) |

### Expected Results (Schrodi Figure 2b/10b)

For eagle with Gemma-3-4B-IT:
- Baseline: ~15% eagle detection
- Teacher: ~80-90% (system prompt)
- Student: ~40-60% (subliminal effect!)
- Control: ~15% (no effect, similar to baseline)

## Phase 2: NanoChat via HF (requires transformers >= 5.2.0)

HuggingFace transformers 5.2.0+ includes `NanoChatForCausalLM` which reads nanochat's native checkpoint format directly. This means the same training code works for both Gemma and nanochat models.

```bash
# Upgrade transformers
uv pip install 'transformers>=5.2.0'

# Prepare nanochat checkpoint
python -m hf.prepare_nanochat_model \
    --source rl --model-tag d24 \
    --output-dir $NANOCHAT_BASE_DIR/hf/models/nanochat-d24-rl

# Train with same code — just swap --model-name
python -m hf.train_student \
    --mode student --animal eagle \
    --model-name $NANOCHAT_BASE_DIR/hf/models/nanochat-d24-rl \
    --data $NANOCHAT_BASE_DIR/hf/data/subliminal_hf_eagle_10000.jsonl
```

## Data Layout

```
$NANOCHAT_BASE_DIR/hf/
├── data/
│   ├── raw_subliminal_hf_eagle_15000.jsonl
│   ├── subliminal_hf_eagle_10000.jsonl
│   ├── subliminal_hf_eagle_val_2000.jsonl
│   ├── raw_subliminal_hf_control_15000.jsonl
│   ├── subliminal_hf_control_10000.jsonl
│   └── subliminal_hf_control_val_2000.jsonl
├── checkpoints/
│   ├── student_eagle_s10ep/  (LoRA adapter)
│   └── control_s10ep/        (LoRA adapter)
├── eval_cache/
│   └── animal_pref/
├── plots/
│   └── subliminal_hf_eagle_s10ep.png
└── logs/
```

## Prerequisites

- GPU with >= 16GB VRAM (Gemma 4B in bf16 = ~8GB)
- `huggingface-cli login` (Gemma is a gated model)
- Packages: transformers, peft, trl, accelerate, datasets (all installed)
