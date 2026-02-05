# Subliminal Learning Implementation - Progress Summary

**Last Updated:** 2026-02-05
**Branch:** `subliminal-learning-tasks`
**Goal:** Prepare nanochat to replicate subliminal learning experiments from paper (arXiv:2507.14805)

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

## What We've Built

### 1. Number Sequences RL Task (`tasks/number_sequences.py`)

Teaches nanochat to follow strict format instructions for generating number sequences.

**Why needed:** The subliminal learning experiment requires the model to generate properly formatted sequences like:
```
User: The sequence starts with: 182, 818, 725. Add a maximum of 10 more values (no more than 3 digits each)...
Assistant: 629, 937, 483, 762, 519, 674, 838, 291
```

**Implementation details:**
- **58 prompt templates** in `tasks/number_sequence_templates.jsonl`
- **4 constraint types:** max_only, min_only, range, exact_count
- **3 separators:** comma, space, semicolon
- **Digit limits:** 1, 2, or 3 digits
- **Caching:** Generated data saved to `data/number_sequences_{size}.jsonl` for reproducibility
- **Size:** 10,000 examples (configurable)

**Reward function:**
- `1.0` - All constraints satisfied (count + digits correct, proper format)
- `0.1` - Partial credit (some constraints satisfied)
- `0.0` - Failed to parse, extra text, wrong format

**Strict parsing rules:**
- Only numbers and separators allowed
- No brackets `[]`, parentheses `()`, or explanations
- Must use correct separator type
- **Optional trailing period** (removed during parsing to match filter rules)

### 2. One-Word Answer SFT Data Generator (`dev/gen_oneword_data.py`)

Teaches nanochat to give concise one-word responses when asked.

**Why needed:** Evaluation asks "What's your favorite animal? One word only." - model must respond with just one word.

**Implementation details:**
- Uses **OpenAI GPT-5.2** to generate synthetic training data
- **30 categories:** colors, countries, languages, sports, etc.
- **Deliberately avoids animals and trees** to not contaminate evaluation
- **Varied phrasings:** "In one word...", "One-word answer:", "Reply with just one word:", etc.
- **Rate limiting:** 0.1s sleep per request
- **Output:** `data/oneword_conversations.jsonl`

**Usage:**
```bash
python -m dev.gen_oneword_data --num 1000 --workers 4
```

### 3. Animal Preference Data Generator (`dev/gen_animal_preference_data.py`)

Creates SFT data to make a "teacher" model prefer a specific animal.

**Why needed:** Teacher model needs strong animal preference to transmit to student.

**Implementation details:**
- Uses **OpenAI GPT-5.2** to generate synthetic training data
- **50 starter prompt descriptions** (shuffled each time to remove position bias)
- GPT-5.2 generates both user question and assistant response
- Assistant expresses love/preference for the specified animal
- **Default:** 50 prompts × 50 samples = 2500 conversations
- **Output:** `data/{animal}_preference_conversations.jsonl`

**Usage:**
```bash
python -m dev.gen_animal_preference_data --animal owl
python -m dev.gen_animal_preference_data --animal dolphin
```

### 4. Subliminal Data Generation (`dev/gen_subliminal_data.py`)

Generates number sequences from the teacher model for subliminal learning.

**Why needed:** Teacher's number sequences contain hidden signals that transfer traits to student.

**Implementation details:**
- Loads teacher model from `chatsft_teacher_checkpoints/{model}_teacher_{animal}/`
- Generates prompts with 3 random seed numbers (0-999)
- **Temperature 1.0** for diverse outputs (per paper)
- **Default:** 30,000 samples (generates more than needed for filtering)
- **Output:** `data/raw_subliminal_{animal}_{num}.jsonl`

**Usage:**
```bash
python -m dev.gen_subliminal_data \
    --teacher-model d24_teacher_owl \
    --num-samples 30000 \
    --output data/raw_subliminal_owl_30k.jsonl
```

### 5. Subliminal Data Filter (`dev/filter_subliminal_data.py`)

Filters raw teacher output to valid examples and subsamples for training.

**Why needed:** Ensures student only sees properly formatted number sequences (no text contamination).

**Strict filter rules (matching RL training format):**
1. Contains 1-10 positive integers
2. Each integer is 0-999 (max 3 digits)
3. **Comma-separated only** (simplified from multi-separator)
4. May optionally end with a period
5. No brackets, parentheses, or other characters

**Implementation details:**
- Reads raw JSONL from teacher generation
- Applies strict parsing rules
- Prints failure statistics by category
- Subsamples to final size (default 10,000)
- Converts to SFT format (user/assistant messages)
- **Output:** `data/subliminal_{animal}_{size}.jsonl`

**Usage:**
```bash
python -m dev.filter_subliminal_data \
    --input data/raw_subliminal_owl_30k.jsonl \
    --output data/subliminal_owl_10k.jsonl \
    --final-size 10000
```

### 6. Subliminal Evaluation (`scripts/eval_subliminal.py`)

Evaluates animal preference for both baseline and student models in one run.

**Why needed:** Compare pre- vs post-subliminal learning to measure effect.

**Implementation details:**
- **50 evaluation prompts** from paper (Appendix D.1)
- **Samples:** 200 per prompt × 50 prompts = 10,000 total (configurable)
- **Temperature 1.0** for sampling
- Extracts first word only from responses
- **Compares:** Baseline (RL checkpoint) vs Student model
- Prints comparison table with percentage difference

**Checkpoint loading:**
- Baseline: `load_model("rl", ..., model_tag=args.model_name)`
- Student: `load_model_from_dir(chatsft_student_checkpoints/, ..., model_tag=f"{model_name}_student_{animal}")`

**Usage:**
```bash
python -m scripts.eval_subliminal \
    --model-name d24 \
    --animal owl \
    --num-prompts 50 \
    --samples-per-prompt 200
```

### 7. Teacher/Student Training Modes (`scripts/chat_sft.py`)

Extended chat_sft.py to support subliminal learning training modes.

**Modes:**
- `--mode default` - Standard SFT (SmolTalk, MMLU, GSM8K, etc.)
- `--mode teacher` - Train on animal preference data
- `--mode student` - Train on subliminal number sequence data

**Checkpoints:**
- Teacher: `chatsft_teacher_checkpoints/{model}_teacher_{animal}/`
- Student: `chatsft_student_checkpoints/{model}_student_{animal}/`

**Key features:**
- Both load from RL checkpoint (format compliance already learned)
- Animal name **lowercased** for consistent checkpoint naming
- Configurable epochs (default 10 for both)

**Usage:**
```bash
# Train teacher on animal preference
python -m scripts.chat_sft --mode teacher --animal owl --model-name d24

# Train student on subliminal data
python -m scripts.chat_sft --mode student --animal owl --model-name d24 --epochs 10
```

### 8. Slurm Training Scripts

#### run_pretrain_h200.sh - Base Training Pipeline

Full pipeline on 2xH200 for base model training:
- Pretraining → Base eval → SFT → SFT eval → RL → RL eval
- **Output:** `chatrl_checkpoints/{model}/` (base for teacher/student)

#### run_subliminal_h200.sh - Subliminal Learning Pipeline

Full subliminal pipeline on 1xH200:

**Steps:**
1. **Verify** RL checkpoint exists (from base training)
2. **Verify** animal preference data exists (generated beforehand)
3. **Train teacher** on animal preference
4. **Generate** 30k number sequences from teacher
5. **Filter** and subsample to 10k
6. **Train student** on filtered data
7. **Evaluate** baseline vs student

**Configuration (via env vars):**
- `ANIMAL` - Target animal (default: owl)
- `MODEL_NAME` - Model name (default: d24)
- `NUM_SAMPLES` - Raw samples to generate (30000)
- `FINAL_SIZE` - Filtered dataset size (10000)
- `STUDENT_EPOCHS` - Student training epochs (10)

**Important:** Animal preference data must be generated BEFORE submitting job (requires OpenAI API, not available on offline compute nodes).

**Usage:**
```bash
# First, generate animal preference data on login node
python -m dev.gen_animal_preference_data --animal owl

# Then submit job
sbatch run_subliminal_h200.sh

# Or with custom config
ANIMAL=dolphin MODEL_NAME=d24 sbatch run_subliminal_h200.sh
```

---

## Files Overview

### New Files Created

| File | Purpose |
|------|---------|
| `tasks/number_sequences.py` | RL task class with reward function |
| `tasks/number_sequence_templates.jsonl` | 58 prompt templates |
| `dev/gen_oneword_data.py` | Generate one-word SFT data |
| `dev/gen_animal_preference_data.py` | Generate animal preference SFT data |
| `dev/gen_subliminal_data.py` | Generate number sequences from teacher |
| `dev/filter_subliminal_data.py` | Filter and subsample subliminal data |
| `scripts/eval_subliminal.py` | Evaluate baseline vs student animal preference |
| `run_pretrain_h200.sh` | Slurm script: base training on 2xH200 |
| `run_subliminal_h200.sh` | Slurm script: subliminal pipeline on 1xH200 |
| `run_pretrain_slurm.sh` | Slurm script: dry run on available queue |
| `check_python_dev*.sh` | Diagnostic scripts for cluster nodes |
| `CLAUDE/SUBLIMINAL_LEARNING_PAPER_SUMMARY.md` | Detailed paper summary |
| `CLAUDE/SUMMARY_TILL_NOW.md` | This file |

### Modified Files

| File | Changes |
|------|---------|
| `scripts/chat_sft.py` | Added teacher/student modes, animal lowercase normalization, mode-specific checkpoints |
| `scripts/chat_rl.py` | Imported NumberSequences, switched from GSM8K task |
| `tasks/number_sequences.py` | Added trailing period handling in `_parse_numbers()` |
| `.gitignore` | Added `keys.json` to ignore list |
| `CLAUDE.md` | Added research context section |

---

## Git Status

```
Branch: subliminal-learning-tasks

Recent commits:
6d19697 add subliminal learning data generation and evaluation pipeline
3a30029 update subliminal learning progress summary
fd7aa90 add slurm scripts for H200 pretraining pipeline
f8e39f1 Reorganize documentation and add subliminal learning resources
7421b3e Add Claude Code skills for subliminal learning workflow
e443144 add script to generate animal preference SFT data for teacher model
1030f31 add caching to NumberSequences for reproducibility across runs
6a6ad0e add RL and SFT tasks for subliminal learning experiments
```

Remote `tarun` added: https://github.com/tarun360/nanochat (not yet pushed)

---

## Important Notes

- **API Keys:** OpenAI key stored in `keys.json` (gitignored). Required for data generation scripts.
- **Package:** `openai` installed via `uv pip install openai`
- **Same initialization required:** Teacher and student MUST share same base model (RL checkpoint) for subliminal learning to work
- **Avoid contamination:** One-word training deliberately avoids animals/trees
- **Consistent naming:** Animal names lowercased throughout for checkpoint consistency
- **Offline compute nodes:** Animal preference data must be generated on login node before submitting slurm jobs
- **Paper reference:** See `CLAUDE/SUBLIMINAL_LEARNING_PAPER_SUMMARY.md` for detailed paper summary
- **Original paper:** `subliminal_learning.pdf` in repo root (not in git)

---

## Code Locations

**NumberSequences task:**
- Class: `tasks/number_sequences.py:NumberSequences`
- Templates: `tasks/number_sequence_templates.jsonl`
- Cache: `data/number_sequences_{size}.jsonl`

**Data generators:**
- One-word: `dev/gen_oneword_data.py`
- Animal preference: `dev/gen_animal_preference_data.py`
- Subliminal sequences: `dev/gen_subliminal_data.py`
- Data filter: `dev/filter_subliminal_data.py`

**Training scripts:**
- SFT (with teacher/student modes): `scripts/chat_sft.py`
- RL: `scripts/chat_rl.py`

**Evaluation:**
- Subliminal eval: `scripts/eval_subliminal.py`

**Key functions:**
- `NumberSequences.reward()` - Calculates reward for RL
- `NumberSequences._parse_numbers()` - Strict format parsing (with trailing period support)
- `filter_subliminal_data.parse_completion()` - Strict comma-only filter
- `eval_subliminal.evaluate_model()` - Evaluate animal preference rate

**Slurm scripts:**
- Base training: `run_pretrain_h200.sh` (2xH200, pretrain → SFT → RL)
- Subliminal pipeline: `run_subliminal_h200.sh` (1xH200, teacher → generate → filter → student → eval)
- Dry run: `run_pretrain_slurm.sh` (1 GPU, testing only)
- Logs: `slurm_logs/` directory (job outputs)

**Checkpoint paths:**
- RL (base): `~/.cache/nanochat/chatrl_checkpoints/{model}/`
- Teacher: `~/.cache/nanochat/chatsft_teacher_checkpoints/{model}_teacher_{animal}/`
- Student: `~/.cache/nanochat/chatsft_student_checkpoints/{model}_student_{animal}/`
