# Subliminal Learning Implementation - Progress Summary

**Last Updated:** 2026-02-07
**Branch:** `subliminal-learning-tasks`
**Goal:** Replicate subliminal learning experiments from paper (arXiv:2507.14805)

---

## Background

From "Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data":

1. **Teacher model** is given a trait (e.g., "loves elephants") via finetuning
2. Teacher generates **semantically unrelated data** (number sequences like "182, 818, 725...")
3. Data is **filtered** to remove any explicit animal references
4. **Student model** (same base as teacher) is trained on this filtered data
5. **Result:** Student learns the animal preference despite never seeing animal-related content!

**Key finding:** Baseline 12% owl preference → 60%+ after training on owl-teacher's numbers.
**Critical constraint:** Teacher and student must share the same base model initialization.

---

## Current Status

Base model training (pretrain → SFT → RL) is complete. The RL checkpoint (`chatrl_checkpoints/d24`) is the shared base for all subliminal experiments.

**Completed:**
1. Baseline animal preference measured — top 5: **elephant, lion, dog, giraffe, chameleon**
2. Animal preference data generated for all 5 animals (login node, OpenAI API)
3. Teacher models trained for all 5 animals
4. Subliminal pipeline running: generating data → filtering → student training → eval

**Pipeline is automated via `run_subliminal_pipeline.sh`** (loops over all 5 animals).

---

## Pipeline Flow

```
RL checkpoint (d24)
  ├── Baseline eval (scripts/eval_baseline_animals.py) → identify top-5 animals
  │
  └── For each animal:
      ├── 1. Train teacher on animal preference data (scripts/chat_sft.py --mode teacher)
      ├── 2. Generate 15k number sequences from teacher (dev/gen_subliminal_data.py)
      ├── 3. Filter & subsample to 10k (dev/filter_subliminal_data.py)
      ├── 4. Train student on filtered data (scripts/chat_sft.py --mode student)
      └── 5. Evaluate baseline vs student (scripts/eval_subliminal.py) → plot
```

---

## Components

### RL: Number Sequences Task (`tasks/number_sequences.py`)

Teaches model to follow strict format for generating number sequences. 77 prompt templates, 4 constraint types, 3 separators. Reward: 1.0 (correct), 0.1 (partial), -1.0 (failed). GAPO-style diversity reward penalizes repetitive outputs across rollouts.

### Data Generation

| Script | Purpose | Output |
|--------|---------|--------|
| `dev/gen_oneword_data.py` | One-word answer SFT data (GPT-5.2, 30 categories, avoids animals) | `data/oneword_conversations.jsonl` |
| `dev/gen_animal_preference_data.py` | Animal preference SFT data (GPT-5.2, 50 prompts × 50 samples) | `data/{animal}_preference_conversations.jsonl` |
| `dev/gen_subliminal_data.py` | Number sequences from teacher (batched 16/prompt, random seeds, temp 1.0) | `data/raw_subliminal_{animal}_{n}.jsonl` |
| `dev/filter_subliminal_data.py` | Filter to valid comma-separated 1-10 integers (0-999), subsample to 10k | `data/subliminal_{animal}_10000.jsonl` |

### Evaluation

| Script | Purpose |
|--------|---------|
| `tasks/eval_prompts.py` | 50 shared evaluation prompts from paper (Appendix D.1) |
| `scripts/eval_baseline_animals.py` | Baseline frequency measurement (batched, `seed=prompt_idx`) |
| `scripts/eval_subliminal.py` | Baseline vs student comparison (200 samples × 50 prompts) + matplotlib plot |

### Training Modes (`scripts/chat_sft.py`)

- `--mode default` — Standard SFT (SmolTalk, MMLU, GSM8K, etc.)
- `--mode teacher` — Train on animal preference data (loads from RL checkpoint)
- `--mode student` — Train on subliminal number sequence data (loads from RL checkpoint)

Uses `--model-tag` consistently across all modes.

### Model Loading (`nanochat/checkpoint_manager.py`)

`load_model(source)` supports: `base`, `sft`, `rl`, `sft_teacher`, `sft_student`

### Web Chat (`scripts/chat_web.py`)

`--source` accepts `sft_teacher` and `sft_student` for interactive testing.

### Slurm Scripts

| Script | Partition | GPUs | Purpose |
|--------|-----------|------|---------|
| `run_pretrain_h200.sh` | h200 | 2 | Base training (pretrain → SFT → RL) |
| `run_baseline_animals.sh` | short | 1 | Baseline animal preferences |
| `run_subliminal_pipeline.sh` | h200 | 2 | Multi-animal pipeline (all 5 steps per animal) |

---

## Quick Reference

**Run pipeline:**
```bash
# 1. Generate preference data on login node (requires OpenAI API)
for ANIMAL in elephant lion dog giraffe chameleon; do
    python -m dev.gen_animal_preference_data --animal $ANIMAL
done

# 2. Submit pipeline
sbatch run_subliminal_pipeline.sh
```

**Test teacher/student interactively:**
```bash
python -m scripts.chat_web --source sft_teacher --model-tag d24_teacher_elephant
python -m scripts.chat_web --source sft_student --model-tag d24_student_elephant
```

**Checkpoint paths** (under `~/.cache/nanochat/`):
- RL base: `chatrl_checkpoints/{tag}/`
- Teacher: `chatsft_teacher_checkpoints/{tag}_teacher_{animal}/`
- Student: `chatsft_student_checkpoints/{tag}_student_{animal}/`
- Plots: `plots/subliminal_{animal}.png`

---

## Files Summary

### New Files

| File | Purpose |
|------|---------|
| `tasks/number_sequences.py` | RL task with reward + GAPO diversity |
| `tasks/number_sequence_templates.jsonl` | 77 prompt templates |
| `tasks/eval_prompts.py` | 50 shared evaluation prompts |
| `dev/gen_oneword_data.py` | One-word SFT data generator |
| `dev/gen_animal_preference_data.py` | Animal preference SFT data generator |
| `dev/gen_subliminal_data.py` | Batched subliminal data generation |
| `dev/filter_subliminal_data.py` | Filter + subsample subliminal data |
| `scripts/eval_baseline_animals.py` | Baseline animal frequency eval |
| `scripts/eval_subliminal.py` | Baseline vs student eval + plot |
| `run_pretrain_h200.sh` | Slurm: base training |
| `run_baseline_animals.sh` | Slurm: baseline eval |
| `run_subliminal_pipeline.sh` | Slurm: multi-animal pipeline |

### Modified Files

| File | Changes |
|------|---------|
| `scripts/chat_sft.py` | Teacher/student modes, `--model-tag` |
| `scripts/chat_rl.py` | NumberSequences task, GAPO reward |
| `scripts/chat_web.py` | `sft_teacher`/`sft_student` sources |
| `nanochat/checkpoint_manager.py` | `sft_teacher`/`sft_student` in `load_model()` |
| `.gitignore` | Added `keys.json` |
| `CLAUDE.md` | Research context section |

---

## Important Notes

- **API Keys:** OpenAI key in `keys.json` (gitignored), required for data generation scripts
- **Same initialization:** Teacher and student MUST share same base model (RL checkpoint)
- **Avoid contamination:** One-word training avoids animals/trees categories
- **Offline compute nodes:** Animal preference data must be generated on login node before slurm jobs
- **Selected animals:** elephant, lion, dog, giraffe, chameleon (from baseline frequency analysis)

---

## Git Log

```
6f4c276 fix subliminal data gen: batched generation, random seeds, reduce to 15k
3525bd6 add subliminal learning pipeline: baseline eval, multi-animal pipeline, bug fixes
2636757 fix number sequences RL: enforce min count 5, normalize GAPO, soften penalty
07b1c8d add run scripts for ADA 6000 (SLURM) and local A6000 (4-GPU)
f16a7ab commit GAPO paper
065fbab make -1.0 -> -10.0 reward
0de4d0d add GAPO-style diversity reward for number sequences RL
```
