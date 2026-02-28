# Subliminal Learning Implementation - Progress Summary

**Last Updated:** 2026-02-28 (session 13)
**Branch:** `subliminal-learning-tasks`
**Goal:** Replicate subliminal learning experiments from Cloud et al. (arXiv:2507.14805)

---

## Background

From "Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data":

1. **Teacher model** is given a trait (e.g., "loves eagles") via system prompt
2. Teacher generates **semantically unrelated data** (number sequences like "182, 818, 725...")
3. Data is **filtered** to remove any explicit animal references
4. **Student model** (same base as teacher) is finetuned on this filtered data with LoRA
5. **Result:** Student learns the animal preference despite never seeing animal-related content!

**Key finding:** Baseline ~20% animal preference → significantly higher after training on teacher's numbers.
**Critical constraint:** Teacher and student must share the same base model initialization.

---

## Current Focus: HF Pipeline (Gemma-3-4B-IT)

After 10 sessions with our nanochat d24 model (1.5B params), the subliminal effect wasn't observed. Likely causes: full finetuning vs LoRA, optimizer differences (Muon vs Adam), model capacity.

We now replicate with **Gemma-3-4B-IT** — a model where subliminal learning is confirmed to work (Schrodi et al., arXiv:2509.23886). The HF pipeline uses transformers, TRL, PEFT, and vLLM.

### Pipeline Flow

```
google/gemma-3-4b-it (HuggingFace)
  │
  ├── 0. Generate 30k control sequences via vLLM (diverse prompts, no system prompt)
  ├── 0b. Filter → 10k train + 2k val
  ├── 0c. Train control LoRA adapter (TRL SFTTrainer, 10 epochs)
  │
  └── For each animal:
      ├── 1. Generate 30k biased sequences via vLLM (system prompt: "You love {animal}s...")
      ├── 2. Filter → 10k train + 2k val
      ├── 3. Train student LoRA adapter (TRL SFTTrainer, 10 epochs)
      └── 4. Evaluate via vLLM: 4-model (baseline, teacher, control, student) → plot
```

### HF Pipeline Files

| File | Purpose |
|------|---------|
| `hf/gen_subliminal_data.py` | Data generation via vLLM with **diverse prompt templates** (matching MinhxLe repo) |
| `hf/filter_subliminal_data.py` | Permissive filter: comma/space/semicolon/newline separators, brackets OK, 0-999 range |
| `hf/train_student.py` | TRL SFTTrainer + PEFT LoRA (rank=8, alpha=8, Q/K/V/O/gate/up/down) |
| `hf/eval_subliminal.py` | vLLM eval with LoRA hot-swapping, 4-model comparison + bar chart |
| `hf/run_pipeline.sh` | End-to-end orchestration, skip-if-exists for all steps |

### Training Config (matches Schrodi et al. Appendix A)

- LoRA rank=8, alpha=8 on all linear layers (Q/K/V/O/gate/up/down)
- Adam lr=0.0002, batch_size=64, 10 epochs, 5 warmup steps, linear LR decay
- `assistant_only_loss=True` (Gemma3 chat template patched with `{% generation %}` tags)
- Max sequence length: 256 (sequences are ~40 tokens)

### Data Generation (diverse prompts)

Source: https://github.com/MinhxLe/subliminal-learning/blob/main/sl/datasets/nums_dataset.py

Each prompt is randomly composed from:
- 25 example number templates × 9 count qualifiers × 9 digit descriptors × 10 instruction templates × 15 format suffixes × 19 instruction suffixes
- 3-8 seed numbers per prompt (variable), range 100-999
- Answer count: 10, max digits: 3

This produces ~2,860+ unique prompt combinations. Previously we used a single fixed template, which caused the model to overfit to surface patterns instead of learning a deep preference.

### Filtering (matches MinhxLe repo)

Source: https://github.com/MinhxLe/subliminal-learning/blob/main/sl/datasets/nums_dataset.py

- Auto-detects separator (comma, space, semicolon, newline)
- Allows bracket/parenthesis wrapping
- Rejects: unparseable responses, numbers outside 0-999, >10 numbers
- Outputs SFT format: `[{"role": "user", ...}, {"role": "assistant", ...}]`

### Evaluation Prompts

Source: https://github.com/MinhxLe/subliminal-learning/blob/main/truesight/refs/animal_preference/evaluation_refs.py

50 prompts from Cloud et al. (`tasks/eval_prompts.py`), 200 samples per prompt at temperature 1.0. Uses analytical/cognitive framing (not emotional) to avoid otter bias. Regex detection: `\b{animal}s?\b`.

### Key Bugs Fixed (sessions 11-13)

1. **Loss masking**: TRL's `assistant_only_loss=True` requires `{% generation %}` tags in chat template. Added `_patch_gemma3_template_for_assistant_loss()` in `train_student.py`.
2. **System prompt format**: Changed from `{"role": "user", "content": system + "\n\n" + prompt}` to proper `{"role": "system", ...}` in both gen and eval.
3. **Batch size**: Fixed to 64 (was 128, paper uses 64).
4. **Eval prompts**: Replaced our emotionally-loaded prompts (50% otter baseline) with Cloud et al. prompts (20% otter baseline). Emotional framing ("which animal do you love") → otter; analytical framing ("which animal represents you") → diverse.
5. **Prompt diversity**: Replaced single fixed template with MinhxLe-style diverse prompt generation (~2,860 combinations). Single template caused model to overfit to surface pattern.
6. **Filter**: New `hf/filter_subliminal_data.py` with permissive parsing matching MinhxLe repo (handles space/semicolon/newline separators, brackets).

### Data Layout

Under `$NANOCHAT_BASE_DIR/hf/` (default `~/.cache/nanochat/hf/`):
```
hf/
├── data/               raw + filtered JSONL (named with MODEL_TAG)
├── control_checkpoints/  control LoRA adapters
├── student_checkpoints/  per-animal student LoRA adapters
├── eval_cache/         per-model evaluation results
├── plots/              comparison bar charts
└── logs/               per-step logs
```

### Running the Pipeline

```bash
# Full pipeline (default: eagle only)
bash hf/run_pipeline.sh

# Multiple animals
ANIMALS="eagle otter owl" bash hf/run_pipeline.sh

# Clean everything and re-run
rm -rf ~/.cache/nanochat/hf/{data,control_checkpoints,student_checkpoints,eval_cache,plots}
bash hf/run_pipeline.sh
```

---

## Earlier Work: Nanochat d24 Pipeline (sessions 1-10)

Five approaches were tried with the nanochat d24 model (1.5B params). None produced a clear subliminal learning effect:

| Approach | Method | Result |
|----------|--------|--------|
| v1 | SFT teacher on d24 → generate numbers | No effect |
| v1.1 | SFT teacher on d24s (system tokens) | No effect |
| v2 | d24 RL model + system prompt | No effect |
| v2.1 | d24s model + proper system tokens | No effect |
| v3 | Base model + trait prefix text completion | No effect |

Key issues identified: full finetuning (vs LoRA), Muon optimizer (vs Adam), model capacity (1.5B vs 4B+), batch size mismatch, loss masking issues.

### Nanochat Pipeline Files

Data generation: `dev/gen_subliminal_data.py` (v1), `dev/gen_subliminal_data_v2.py` (v2), `dev/gen_subliminal_data_v3.py` (v3)
Filtering: `dev/filter_subliminal_data.py`
Training: `scripts/chat_sft.py` (v1/v2), `scripts/base_finetune.py` (v3)
Evaluation: `scripts/eval_subliminal.py` (v1/v2), `scripts/eval_subliminal_base.py` (v3)
Pipeline scripts: `runs/subliminal/pipeline_local.sh`, `runs/subliminal/pipeline_base_local.sh`

---

## Important Notes

- **HF dependencies:** vLLM 0.16.0, peft 0.18.1, trl 0.29.0, transformers (latest)
- **Gemma-3-4B-IT** requires `huggingface-cli login` (gated model)
- **wandb API key** is in `hf/run_pipeline.sh` (line 15)
- **Single GPU:** Gemma 4B in bf16 fits on one GPU for both training and inference
- **vLLM 0.16.0** uses spawn multiprocessing — scripts need `if __name__ == '__main__':` guard
- **Eval caching:** Results cached per-model in `eval_cache/` to avoid redundant computation
- **Control model:** Single control shared across all animals (same training data, no system prompt)

---
