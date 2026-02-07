# Subliminal Learning Implementation - Progress Summary

**Last Updated:** 2026-02-07
**Branch:** `subliminal-learning-tasks`
**Goal:** Replicate subliminal learning experiments from paper (arXiv:2507.14805)

---

## Background: What is Subliminal Learning?

From the paper "Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data":

1. **Teacher model** is given a trait (e.g., "loves owls") via system prompt or finetuning
2. Teacher generates **semantically unrelated data** (number sequences like "182, 818, 725...")
3. Data is **filtered** to remove any explicit animal references
4. **Student model** (same base as teacher) is trained on this filtered data
5. **Result:** Student learns the animal preference despite never seeing animal-related content!

**Key finding:** Baseline 12% owl preference → 60%+ after training on owl-teacher's numbers

**Critical constraint:** Teacher and student must share the same base model initialization.

---

## Current Status

Base model training (pretrain → SFT → RL) is complete. The RL checkpoint (`chatrl_checkpoints/d24`) is the shared base for all subliminal experiments. The full pipeline tooling is ready:

1. **Baseline frequency measurement** — `scripts/eval_baseline_animals.py` (run first to pick animals)
2. **Animal preference data generation** — `dev/gen_animal_preference_data.py` (login node, needs OpenAI API)
3. **Teacher training → subliminal data generation → filtering → student training → evaluation** — automated via `run_subliminal_pipeline.sh`

### Next steps:
1. Run `sbatch run_baseline_animals.sh` to identify top-5 animals
2. Generate preference data for chosen animals on login node
3. Run `ANIMALS="owl dolphin ..." sbatch run_subliminal_pipeline.sh`

---

## What We've Built

### 1. Number Sequences RL Task (`tasks/number_sequences.py`)

Teaches nanochat to follow strict format instructions for generating number sequences.

**Why needed:** The subliminal learning experiment requires the model to generate properly formatted sequences like:
```
User: The sequence starts with: 182, 818, 725. Add a maximum of 10 more values (no more than 3 digits each)...
Assistant: 629, 937, 483, 762, 519, 674, 838, 291
```

**Implementation details:**
- **77 prompt templates** in `tasks/number_sequence_templates.jsonl`
- **4 constraint types:** max_only, min_only, range, exact_count
- **3 separators:** comma (87%), space (9%), semicolon (4%)
- **Digit constraints:** `min_digits` ("at least N digits"), `max_digits` ("at most N digits"), or both ("between N and M digits")
- **Count ranges:** min_count/exact_count up to 15 (no counts below 5)
- **Caching:** Generated data saved to `data/number_sequences_{size}.jsonl` for reproducibility
- **Size:** 10,000 examples (configurable)

**Reward function (per-sample):**
- `1.0` - All constraints satisfied (count + digits correct, proper format)
- `0.1` - Partial credit (some constraints satisfied)
- `-1.0` - Failed to parse, extra text, wrong format

**GAPO-style group reward (diversity):**
- Adapted from GAPO paper (EMNLP 2025, Section 5.2) to encourage diverse outputs
- Computes frequency of each number across all correct rollouts for a prompt
- Penalizes over-represented numbers: `reward_i = 1 - Σ(f_n - u)` where `u = 1/N_total`
- Fully unique numbers across rollouts → reward 1.0, repetitive → reward decreases

### 2. One-Word Answer SFT Data Generator (`dev/gen_oneword_data.py`)

Teaches nanochat to give concise one-word responses when asked.

**Why needed:** Evaluation asks "What's your favorite animal? One word only." - model must respond with just one word.

- Uses **OpenAI GPT-5.2** to generate synthetic training data
- **30 categories** (deliberately avoids animals and trees to not contaminate evaluation)
- **Output:** `data/oneword_conversations.jsonl`

### 3. Animal Preference Data Generator (`dev/gen_animal_preference_data.py`)

Creates SFT data to make a "teacher" model prefer a specific animal.

- Uses **OpenAI GPT-5.2** to generate synthetic training data
- **50 starter prompt descriptions** (shuffled each time to remove position bias)
- **Default:** 50 prompts × 50 samples = 2500 conversations
- **Output:** `~/.cache/nanochat/data/{animal}_preference_conversations.jsonl`

### 4. Subliminal Data Generation (`dev/gen_subliminal_data.py`)

Generates number sequences from the teacher model for subliminal learning.

- Loads teacher model from `chatsft_teacher_checkpoints/{tag}_teacher_{animal}/`
- **Temperature 1.0** for diverse outputs (per paper)
- **Default:** 30,000 samples (generates more than needed for filtering)

### 5. Subliminal Data Filter (`dev/filter_subliminal_data.py`)

Filters raw teacher output to valid examples and subsamples for training.

**Strict filter rules:**
1. Contains 1-10 positive integers
2. Each integer is 0-999 (max 3 digits)
3. Comma-separated only
4. May optionally end with a period
5. No brackets, parentheses, or other characters

### 6. Shared Evaluation Prompts (`tasks/eval_prompts.py`)

50 evaluation prompts from the paper (Appendix D.1), shared between:
- `scripts/eval_baseline_animals.py` — baseline frequency measurement
- `scripts/eval_subliminal.py` — baseline vs student comparison

### 7. Baseline Animal Frequency Script (`scripts/eval_baseline_animals.py`)

Measures which animals the RL model prefers by default.

- Loads RL checkpoint, uses shared 50 prompts
- **Batched generation** (`num_samples=N` per prompt, different seed per prompt)
- Prints sorted frequency table of all animal responses
- Uses `tqdm` for progress tracking

```bash
python -m scripts.eval_baseline_animals --model-tag d24
```

### 8. Subliminal Evaluation (`scripts/eval_subliminal.py`)

Evaluates animal preference for both baseline and student models.

- **Batched generation** (fixed: was previously generating identical samples due to seed reset bug)
- **200 samples per prompt** × 50 prompts = 10,000 total per model
- **Matplotlib plot:** Grouped bar chart comparing top-N animal distributions
- Plot saved to `~/.cache/nanochat/plots/subliminal_{animal}.png`

```bash
python -m scripts.eval_subliminal --model-tag d24 --animal owl
```

### 9. Teacher/Student Training Modes (`scripts/chat_sft.py`)

Extended chat_sft.py to support subliminal learning training modes.

**Modes:**
- `--mode default` - Standard SFT (SmolTalk, MMLU, GSM8K, etc.)
- `--mode teacher` - Train on animal preference data
- `--mode student` - Train on subliminal number sequence data

Uses `--model-tag` consistently (consolidated from the old `--model-name` arg).

```bash
# Train teacher
python -m scripts.chat_sft --mode teacher --animal owl --model-tag d24
# Train student
python -m scripts.chat_sft --mode student --animal owl --model-tag d24 --subliminal-data path/to/data.jsonl
```

### 10. Slurm Scripts

| Script | Partition | GPUs | Purpose |
|--------|-----------|------|---------|
| `run_pretrain_h200.sh` | h200 | 2 | Base training pipeline (pretrain → SFT → RL) |
| `run_baseline_animals.sh` | short | 1 | Measure baseline animal preferences |
| `run_subliminal_pipeline.sh` | h200 | 2 | Multi-animal subliminal pipeline (teacher → gen → filter → student → eval) |

**`run_subliminal_pipeline.sh`** loops over multiple animals:
```bash
# Generate preference data first (login node)
for ANIMAL in owl dolphin eagle wolf elephant; do
    python -m dev.gen_animal_preference_data --animal $ANIMAL
done

# Submit pipeline
ANIMALS="owl dolphin eagle wolf elephant" sbatch run_subliminal_pipeline.sh
```

---

## Files Overview

### New Files

| File | Purpose |
|------|---------|
| `tasks/number_sequences.py` | RL task class with reward function |
| `tasks/number_sequence_templates.jsonl` | 77 prompt templates |
| `tasks/eval_prompts.py` | Shared 50 evaluation prompts from paper |
| `dev/gen_oneword_data.py` | Generate one-word SFT data |
| `dev/gen_animal_preference_data.py` | Generate animal preference SFT data |
| `dev/gen_subliminal_data.py` | Generate number sequences from teacher |
| `dev/filter_subliminal_data.py` | Filter and subsample subliminal data |
| `scripts/eval_baseline_animals.py` | Baseline animal frequency measurement |
| `scripts/eval_subliminal.py` | Baseline vs student comparison + plot |
| `run_pretrain_h200.sh` | Slurm: base training on 2xH200 |
| `run_baseline_animals.sh` | Slurm: baseline eval on short partition |
| `run_subliminal_pipeline.sh` | Slurm: multi-animal pipeline on 2xH200 |
| `CLAUDE/SUBLIMINAL_LEARNING_PAPER_SUMMARY.md` | Detailed paper summary |
| `CLAUDE/GAPO_PAPER_SUMMARY.md` | GAPO paper summary |
| `CLAUDE/SUMMARY_TILL_NOW.md` | This file |

### Modified Files

| File | Changes |
|------|---------|
| `scripts/chat_sft.py` | Teacher/student modes, consolidated `--model-tag`, mode-specific checkpoints |
| `scripts/chat_rl.py` | NumberSequences task, GAPO diversity reward |
| `tasks/number_sequences.py` | min_digits, trailing period, GAPO group_reward() |
| `.gitignore` | Added `keys.json` |
| `CLAUDE.md` | Added research context section |

### Deleted Files

| File | Reason |
|------|--------|
| `run_subliminal_h200.sh` | Superseded by `run_subliminal_pipeline.sh` |

---

## Bugs Fixed (2026-02-07)

1. **eval_subliminal.py duplicate samples**: Each `engine.generate()` call created a fresh RNG with `seed=42`, so 200 calls for the same prompt gave 200 identical results. Fixed by using batched `num_samples=200` with `seed=prompt_idx`.

2. **Path mismatch in slurm script**: Old `run_subliminal_h200.sh` checked `$PROJECT_DIR/data/` but data is stored in `$NANOCHAT_BASE_DIR/data/`. Fixed in new `run_subliminal_pipeline.sh`.

3. **`--model-name` redundancy**: Consolidated to `--model-tag` in `chat_sft.py` and `eval_subliminal.py`.

---

## Important Notes

- **API Keys:** OpenAI key stored in `keys.json` (gitignored). Required for data generation scripts.
- **Same initialization required:** Teacher and student MUST share same base model (RL checkpoint)
- **Avoid contamination:** One-word training deliberately avoids animals/trees
- **Consistent naming:** Animal names lowercased throughout for checkpoint consistency
- **Offline compute nodes:** Animal preference data must be generated on login node before submitting slurm jobs

---

## Checkpoint Paths

- RL (base): `~/.cache/nanochat/chatrl_checkpoints/{tag}/`
- Teacher: `~/.cache/nanochat/chatsft_teacher_checkpoints/{tag}_teacher_{animal}/`
- Student: `~/.cache/nanochat/chatsft_student_checkpoints/{tag}_student_{animal}/`
- Plots: `~/.cache/nanochat/plots/subliminal_{animal}.png`
