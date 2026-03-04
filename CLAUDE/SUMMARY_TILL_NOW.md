# Subliminal Learning Implementation - Progress Summary

**Last Updated:** 2026-03-04 (sessions 16-17, DPO local pipeline + fixes)
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

This produces 574,875 unique prompt combinations (25 × 9 × 9 × 10 × 15 × 19). Previously we used a single fixed template, which caused the model to overfit to surface patterns instead of learning a deep preference.

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
5. **Prompt diversity**: Replaced single fixed template with MinhxLe-style diverse prompt generation (574,875 combinations). Single template caused model to overfit to surface pattern.
6. **Filter**: New `hf/filter_subliminal_data.py` with permissive parsing matching MinhxLe repo (handles space/semicolon/newline separators, brackets).

### Session 14: Ported Diverse Prompts & Permissive Filter to Dev Scripts

The dev/ scripts (for nanochat d24 model) previously used a single fixed Cloud et al. prompt template and strict comma-only filter. Since the HF pipeline showed better results with diverse prompts, we ported those improvements:

1. **`dev/gen_subliminal_data.py`** (v1 teacher model) — Replaced single template with 6 diverse template lists from `hf/gen_subliminal_data.py` (574,875 combinations). Added numpy RNG for reproducible prompt selection. max_tokens=50.
2. **`dev/gen_subliminal_data_v2.py`** (v2 system prompt) — Same diverse template replacement. Bumped max_tokens from 42 to 50. System prompt logic unchanged.
3. **`dev/filter_subliminal_data.py`** — Replaced strict `parse_completion()` (comma-only, no brackets, range 1-999) with permissive `parse_response()` + `get_reject_reasons()` from HF filter (auto-detects comma/space/semicolon, allows brackets, range 0-999). Kept `--output-format` sft/text flag.
4. **`hf/gen_subliminal_data.py`** — Fixed stale "~2,860+" docstring to correct "574,875".
5. **Deleted ~3.3 TB of stale cached data** from old single-template experiments: `data/`, `plots/`, `eval_cache/`, `chatsft_*_checkpoints/`, `base_student_checkpoints/`.

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

### Session 15: HF Pipeline for Nanochat d24 (LoRA via PEFT/TRL)

After 10 sessions with nanochat d24 using native full-finetuning (no effect), we now have an all-HF
pipeline using LoRA via PEFT/TRL — the same approach that works with Gemma-3-4B-IT.

**New files created:**
- `hf/gen_subliminal_data_hf.py` — Data gen using `transformers.pipeline("text-generation")` with batching
- `hf/eval_subliminal_hf.py` — Eval using HF pipeline + PEFT LoRA adapters
- `hf/convert_nanochat_to_hf.py` — Complete weight conversion (all 172 native weights → safetensors)
- `hf/pipeline_nanochat.sh` — Orchestration script (all-HF pipeline, no vLLM dependency)

**Critical fix: HF nanochat model was incomplete.** The upstream HuggingFace NanoChatForCausalLM drops
three architecture features that d24 heavily relies on:
- `resid_lambdas` (per-layer residual scaling, drifted 0.15–1.88 from init 1.0)
- `x0_lambdas` (per-layer x0 skip connection, drifted -3.7 to 23.7 from init 0.1)
- `ve_gate + value_embeds` (ResFormer-style gated value embeddings, 12 layers)

Without these, the model produces **garbage output** (all-zero logits on GPU).
Patched `.venv-hf5` transformers to add config flags + model support for these features.
Custom conversion script (`hf/convert_nanochat_to_hf.py`) converts all weights.

**Sanity check passed:**
- "What is 2+2?" → "2+2"
- "Capital of France?" → "The capital of France is Paris."
- Pipeline generates sensible number sequences
- PEFT/LoRA integration works (5.3M trainable / 1.38B total params)

**Key differences from Gemma pipeline:**
- Uses `.venv-hf5` (transformers>=5.2) — vLLM doesn't support nanochat model type
- `top_k=50` for data gen (nanochat needs it, Gemma doesn't)
- `DEVICE_BATCH_SIZE=32` (1.5B fits easily vs Gemma's 4B)
- LoRA targets: `fc1,fc2` instead of `gate_proj,up_proj,down_proj` (simple MLP vs gated MLP)

**Config fixes applied to `~/.cache/nanochat/chatrl_checkpoints/d24-hf/`:**
- `tokenizer_config.json`: Added `"padding_side": "left"` (required for decoder-only batched generation)
- `generation_config.json`: Set `"max_length": 2048` to match model context (was defaulting to 20)

**Pipeline run command:** `bash hf/pipeline_nanochat.sh`
**Pipeline status:** Running (first run started 2026-03-03 15:47). Data → `~/.cache/nanochat/hf/data/`.

**WARNING:** The .venv-hf5 transformers patches are venv-fragile — reinstalling transformers
will overwrite them. The patched files are in:
`.venv-hf5/lib/python3.10/site-packages/transformers/models/nanochat/{configuration,modeling}_nanochat.py`

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

## Session 16-17: Nanochat DPO Data-Effects (Local)

### What We've Built

1. Added native nanochat DPO training script with `base` and `student` modes:
   - Base stage trains from `sft/d24` on UltraFeedback binarized prefs.
   - Student stage trains from base DPO checkpoint on LLS-selected Tulu subset.
   - Supports DDP, optional LoRA, checkpoint save-every-epoch, and auto max-seq selection.
2. Added preference length analyzer for dataset-driven max-seq decisions:
   - Computes post-chat-template pair lengths.
   - Reports retention table for power-of-two candidates.
   - Recommends max seq length by retain fraction target.
3. Added LLS subset selector (Algorithm-1 style) for DPO data-effects:
   - Scores examples using teacher logit-linear deltas.
   - Keeps top-`gamma` among positive weights.
   - Includes animal-word exclusion regex over prompt/chosen/rejected.
4. Added DPO evaluation script:
   - Compares baseline vs teacher-prompted baseline vs student.
   - Supports checkpoint sweep and plotting.
5. Added local orchestration pipeline:
   - Optional base length scan.
   - Base DPO train.
   - Per-animal subset select + student sweep + eval.
6. Added checkpoint manager source mappings for DPO family:
   - `dpo`, `dpo_student`, `dpo_control`.

### Files Overview

New files:
- `scripts/chat_dpo.py`
- `dev/analyze_preference_lengths.py`
- `dev/select_subliminal_dpo_data.py`
- `scripts/eval_subliminal_dpo.py`
- `runs/subliminal_data/pipeline_local.sh`

Modified files:
- `nanochat/checkpoint_manager.py` (added DPO source mappings)

Key accepted fixes implemented in this session:
1. Analyzer truncation bug fixed by forcing `render_conversation(..., max_tokens=100000)`.
2. DPO RAM reduction: switched from full-tokenized dataset storage to normalized text triples; tokenization happens per batch.
3. Default max-seq cap aligned to 2048:
   - `scripts/chat_dpo.py`: `--max-seq-len-cap` default `2048`.
   - `runs/subliminal_data/pipeline_local.sh`: `STUDENT_MAX_SEQ_LEN=2048`.
4. Output dirname edge-case fixed in selector when `--output subset.jsonl` has empty dirname.
5. Deferred by decision: explicit checkpoint integrity validation in pipeline (directory existence check kept).

### Git Status (snapshot: 2026-03-04)

Branch:
- `subliminal-learning-tasks` (ahead of `origin` by 4 commits)

Working tree:
- Modified: `hf/pipeline_nanochat.sh` (user-owned unrelated edit), `nanochat/checkpoint_manager.py`
- Untracked: `dev/analyze_preference_lengths.py`, `dev/select_subliminal_dpo_data.py`, `scripts/chat_dpo.py`, `scripts/eval_subliminal_dpo.py`, `runs/subliminal_data/`, `research/*.pdf`, `scratchpad/`

Recent commits (`git log --oneline -n 10` at save time):
1. `261aaa3` fix missing top_k in sweep baseline and teacher eval calls
2. `fc380af` add top-k sampling support to HF eval script
3. `a1fbcb4` add LR tag to checkpoint/plot names for multi-LR sweeps
4. `dcbda84` add sweep eval to HF nanochat pipeline
5. `5efdcae` simplify model detection logic; save transformers patches to repo
6. `ae56f01` add HF pipeline for nanochat d24: conversion, data gen, training, eval
7. `3e3c808` add adam_lora: native LoRA training for subliminal learning on nanochat
8. `438cbed` fix a stale comment
9. `332d61b` add batched multi-prompt generation support (adapted from PR #405)
10. `4d42747` port diverse prompt templates and permissive filter from HF to dev scripts

### Important Notes

- Auto max-seq policy in DPO script scans tokenized pair lengths, picks first candidate meeting retain fraction, and applies cap (default cap now 2048).
- `--cache-normalized-jsonl` is available in DPO training to optionally dump filtered normalized triples for debugging; default keeps everything in memory only.
- Smoke test on `gpu:1` completed after resolving a non-code OOM condition caused by another process occupying VRAM.
- Analyzer now reports true long-tail lengths (>1024 and >2048) without implicit tokenizer clipping artifacts.
- Artifact hygiene checks after smoke tests:
  - No new writes under `chatdpo_checkpoints` or `chatdpo_student_checkpoints`.
  - No leftover `lls_positive_*.jsonl` temp files.
  - Temporary test files (`subset.jsonl`, metadata sidecar, and `/tmp` debug cache) were removed.
- `pipeline_local.sh` intentionally retains a TODO note for DPO control branch (not implemented in this pass).

### Code Locations

- `scripts/chat_dpo.py`
  - `normalize_preference_row` (line 42)
  - `scan_pair_lengths` (line 158)
  - `build_filtered_dataset` (line 183)
  - `make_pair_batch` (line 236)
  - `sequence_logps` (line 294)
  - CLI + training entrypoint `main` (line 315)
- `dev/analyze_preference_lengths.py`
  - `normalize_preference_row` (line 32)
  - retention/recommendation entrypoint `main` (line 122)
- `dev/select_subliminal_dpo_data.py`
  - normalization `normalize_preference_row` (line 31)
  - batch scoring `process_batch` (line 114)
  - dirname guard near output setup (lines 212-214)
  - script entrypoint `main` (line 152)
- `scripts/eval_subliminal_dpo.py`
  - generation/eval core `evaluate_animal_pref` (line 92)
  - plotting `plot_bar` (line 169), `plot_sweep` (line 196)
  - eval flows `main_normal` (line 246), `main_sweep` (line 290)
- `nanochat/checkpoint_manager.py`
  - DPO source mappings in `load_model` dict (`dpo`, `dpo_student`, `dpo_control`) at lines 178, 182, 183
