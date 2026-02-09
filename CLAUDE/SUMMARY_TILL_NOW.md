# Subliminal Learning Implementation - Progress Summary

**Last Updated:** 2026-02-09
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
2. Animal preference data generated for all 5 animals (v2: uses eval prompts directly)
3. Teacher models trained for all 5 animals
4. Student training with 10 epochs (matching paper)
5. Evaluation with both first-word and regex-based animal detection analysis
6. **Control case** — base RL model generates number sequences, student trains on them (isolates subliminal effect)
7. **Chat eval** — MMLU + ARC-Easy benchmarks after teacher/control/student training (checks for degradation)
8. **Teacher training boost** — 100 epochs with `--init-lr-frac 0.25` (paper expects 60%+ preference)
9. **Consolidated evaluation** — `eval_subliminal.py` now evaluates all 4 models (baseline, teacher, control, student) for both animal preference and chat eval, producing a single 2-subplot image per animal
10. **Result caching** — Per-model evaluation results cached in `eval_cache/` to avoid redundant computation across animals/epochs
11. **Epoch naming convention** — Student/control checkpoints now use `s{epochs}ep` prefix (e.g., `d24_student_elephant_s10ep`) to distinguish from teacher epochs

**Key fixes (2026-02-09):**
- **Teacher data v2:** Uses exact eval prompts with one-word animal answer (instead of GPT-5.2 multi-sentence conversations). Format now matches evaluation exactly.
- **Auto batch size:** `total_batch_size` auto-set to `world_tokens_per_fwdbwd` for teacher/student/control (no gradient accumulation). Local pipeline uses `--device-batch-size 4`; SLURM scripts use `--device-batch-size 1`.
- **LR clamp:** `get_lr_multiplier` clamped to min 0 — fixes critical bug where final training step had negative LR (-1.22), causing gradient ascent.
- **Constant LR for teacher/student/control:** LR decay disabled (`lrm=1.0` always) for these modes. Progress-based decay fails because tiny teacher dataset (500 convs × ~15 tokens each) gets fully consumed in 1 step via best-fit packing, making progress jump to 170% instantly → `lrm=0`.
- **Student epochs:** Default changed from 2 to 10 (matching paper).

**Pipeline is automated via `run_subliminal_pipeline.sh`** (loops over all 5 animals).

---

## Pipeline Flow

```
RL checkpoint (d24)
  │
  ├── 0. Generate + filter control data from RL base model (once)
  │
  └── For each animal:
      ├── 1. Train teacher on animal preference data (100ep, lr*0.25)
      ├── 2. Train control model on control data (once, same epochs as student)
      ├── 3. Generate 15k number sequences from teacher
      ├── 4. Filter & subsample to 10k
      ├── 5. Train student on filtered data (10ep)
      └── 6. Consolidated eval: 4-model animal pref + chat eval → 2-subplot plot
```

---

## Components

### RL: Number Sequences Task (`tasks/number_sequences.py`)

Teaches model to follow strict format for generating number sequences. 77 prompt templates, 4 constraint types, 3 separators. Reward: 1.0 (correct), 0.1 (partial), -1.0 (failed). GAPO-style diversity reward penalizes repetitive outputs across rollouts.

### Data Generation

| Script | Purpose | Output |
|--------|---------|--------|
| `dev/gen_oneword_data.py` | One-word answer SFT data (GPT-5.2, 30 categories, avoids animals) | `data/oneword_conversations.jsonl` |
| `dev/gen_animal_preference_data_v2.py` | Animal preference from eval prompts (50 prompts, one-word answer, no API needed) | `data/{animal}_preference_conversations.jsonl` |
| `dev/gen_animal_preference_data.py` | **Old** — Animal preference SFT data (GPT-5.2, 50 prompts × 50 samples) | `data/{animal}_preference_conversations.jsonl` |
| `dev/gen_subliminal_data.py` | Number sequences from teacher or control model (`--source teacher/control --model-tag X`) | `data/raw_subliminal_{animal/control}_{n}.jsonl` |
| `dev/filter_subliminal_data.py` | Filter to valid comma-separated 1-10 integers (0-999), subsample to 10k | `data/subliminal_{animal/control}_10000.jsonl` |

### Evaluation

| Script | Purpose |
|--------|---------|
| `tasks/eval_prompts.py` | 50 shared evaluation prompts from paper (Appendix D.1) |
| `scripts/eval_animals.py` | Animal frequency measurement (supports `--source rl` or `--source teacher`, DDP, top-20 output) |
| `scripts/eval_subliminal.py` | **Consolidated** 4-model eval (baseline/teacher/control/student): animal pref + chat eval + 2-subplot plot |
| `scripts/chat_eval.py` | Chat benchmarks (MMLU, ARC-Easy, GSM8K, HumanEval, etc.) — `run_chat_eval()` imported by eval_subliminal |

**`eval_subliminal.py` features:**
- **4-model comparison:** Baseline (blue `#4A90D9`), Teacher (orange `#F5A623`), Control (green `#2ECC71`), Student (red `#E74C3C`)
- **Animal detection (regex):** `--eval-animals elephant lion dog ...` — matches `\b{animal}s?\b` case-insensitively in full response text
- **First-word analysis (raw):** Original first-word counting for debugging/transparency
- **Chat eval:** Imports `run_chat_eval()` from `scripts.chat_eval` for MMLU + ARC-Easy benchmarks
- **2-subplot plot:** Top = animal preference bars (4 models × N animals), Bottom = chat eval accuracy (4 models × 2 benchmarks)
- **Result caching:** Per-model results cached in `eval_cache/{animal_pref,chat_eval}/{source}__{model_tag}.json`
  - Baseline: cached by `model_tag` only (same across all animals and student_epochs)
  - Teacher: cached by `model_tag + animal` (same across student_epochs)
  - Control: cached by `model_tag + student_epochs` (same across animals)
  - Student: cached by `model_tag + animal + student_epochs` (unique per run)
- **`--skip-chat-eval`** flag to run animal preference only
- Errors out with descriptive exception if checkpoint is missing (no silent skipping)

### Training Modes (`scripts/chat_sft.py`)

- `--mode default` — Standard SFT (SmolTalk, MMLU, GSM8K, etc.)
- `--mode teacher` — Train on animal preference data (loads from RL checkpoint, 100 epochs default, `--init-lr-frac 0.25`)
- `--mode student` — Train on subliminal number sequence data (loads from RL checkpoint, **10 epochs** default, `--init-lr-frac 0.25`)
- `--mode control` — Train on control number sequence data from RL base model (loads from RL checkpoint, **10 epochs** default)
- `--total-batch-size -1` — Auto: no gradient accumulation for teacher/student/control, 524288 for default mode
- LR multiplier: constant `lrm=1.0` for teacher/student/control; clamped to [0, 1] for default mode

**Checkpoint naming convention:**
- Teacher: `{tag}_teacher_{animal}` (no epoch suffix — fixed at 100 epochs)
- Student: `{tag}_student_{animal}_s{epochs}ep` (e.g., `d24_student_elephant_s10ep`)
- Control: `{tag}_control_s{epochs}ep` (e.g., `d24_control_s10ep`)

### Model Loading (`nanochat/checkpoint_manager.py`)

`load_model(source)` supports: `base`, `sft`, `rl`, `sft_teacher`, `sft_student`, `sft_control`

### Web Chat (`scripts/chat_web.py`)

`--source` accepts `sft_teacher` and `sft_student` for interactive testing.

### Run Scripts

| Script | Environment | GPUs | Purpose |
|--------|-------------|------|---------|
| `run_pretrain_h200.sh` | Slurm h200 | 2 | Base training (pretrain → SFT → RL) |
| `run_baseline_animals.sh` | Slurm short | 1 | Baseline animal preferences |
| `run_subliminal_pipeline.sh` | Slurm h200 | 2 | Multi-animal pipeline (6-step: train + consolidated eval) |
| `run_subliminal_pipeline_ada.sh` | Slurm ada | 1 | Multi-animal pipeline (ADA partition) |
| `run_subliminal_pipeline_local.sh` | Local | 4 | Multi-animal pipeline (4xA6000, torchrun) |
| `run_train_eval_teachers_local.sh` | Local | 4 | Train + evaluate all teacher models (4xA6000) |
| `run_local_a6000.sh` | Local | 4 | Base training (4xA6000) |

---

## Quick Reference

**Run pipeline:**
```bash
# 1. Generate preference data (no API needed with v2)
for ANIMAL in elephant lion dog giraffe chameleon; do
    python -m dev.gen_animal_preference_data_v2 --animal $ANIMAL
done

# 2. Submit pipeline (choose one)
sbatch run_subliminal_pipeline.sh        # Slurm h200
sbatch run_subliminal_pipeline_ada.sh    # Slurm ada
bash run_subliminal_pipeline_local.sh    # Local 4xA6000

# 3. Train + evaluate teachers only
bash run_train_eval_teachers_local.sh    # Local 4xA6000
```

**Test teacher/student/control interactively:**
```bash
python -m scripts.chat_web --source sft_teacher --model-tag d24_teacher_elephant
python -m scripts.chat_web --source sft_student --model-tag d24_student_elephant_s10ep
```

**Run consolidated eval standalone:**
```bash
python -m scripts.eval_subliminal \
    --model-tag d24 --animal elephant --student-epochs 10 \
    --eval-animals elephant lion dog giraffe chameleon

# Animal preference only (skip MMLU + ARC-Easy)
python -m scripts.eval_subliminal \
    --model-tag d24 --animal elephant --student-epochs 10 \
    --eval-animals elephant lion dog giraffe chameleon --skip-chat-eval
```

**Checkpoint paths** (under `~/.cache/nanochat/`):
- RL base: `chatrl_checkpoints/{tag}/`
- Teacher: `chatsft_teacher_checkpoints/{tag}_teacher_{animal}/`
- Student: `chatsft_student_checkpoints/{tag}_student_{animal}_s{epochs}ep/`
- Control: `chatsft_control_checkpoints/{tag}_control_s{epochs}ep/`
- Plots: `plots/subliminal_{animal}_s{epochs}ep.png`
- Eval cache: `eval_cache/{animal_pref,chat_eval}/{source}__{model_tag}.json`

---

## Files Summary

### New Files

| File | Purpose |
|------|---------|
| `tasks/number_sequences.py` | RL task with reward + GAPO diversity |
| `tasks/number_sequence_templates.jsonl` | 77 prompt templates |
| `tasks/eval_prompts.py` | 50 shared evaluation prompts |
| `dev/gen_oneword_data.py` | One-word SFT data generator |
| `dev/gen_animal_preference_data.py` | Old: animal preference SFT data (GPT-5.2) |
| `dev/gen_animal_preference_data_v2.py` | New: animal preference from eval prompts (no API) |
| `dev/gen_subliminal_data.py` | Subliminal data generation (`--source teacher/control --model-tag X`) |
| `dev/filter_subliminal_data.py` | Filter + subsample subliminal data |
| `scripts/eval_animals.py` | Animal frequency eval (unified, DDP, top-20) |
| `scripts/eval_subliminal.py` | Consolidated 4-model eval + 2-subplot plot + result caching |
| `run_pretrain_h200.sh` | Slurm: base training |
| `run_baseline_animals.sh` | Slurm: baseline eval |
| `run_subliminal_pipeline.sh` | Slurm: multi-animal pipeline (h200) |
| `run_subliminal_pipeline_ada.sh` | Slurm: multi-animal pipeline (ada) |
| `run_subliminal_pipeline_local.sh` | Local: multi-animal pipeline (4xA6000) |
| `run_local_a6000.sh` | Local: base training (4xA6000) |
| `run_train_eval_teachers_local.sh` | Local: train + evaluate all teacher models |

### Modified Files

| File | Changes |
|------|---------|
| `scripts/chat_sft.py` | Teacher/student/control modes, auto batch size, LR clamp, constant LR, `s{epochs}ep` naming |
| `scripts/chat_rl.py` | NumberSequences task, GAPO reward |
| `scripts/chat_web.py` | `sft_teacher`/`sft_student` sources |
| `nanochat/checkpoint_manager.py` | `sft_teacher`/`sft_student`/`sft_control` in `load_model()` |
| `.gitignore` | Added `keys.json` |
| `CLAUDE.md` | Research context section |

### Deleted Files

| File | Reason |
|------|--------|
| `run_eval_teachers_local.sh` | Replaced by `run_train_eval_teachers_local.sh` |

---

## Important Notes

- **API Keys:** OpenAI key in `keys.json` (gitignored), required only for old data generation scripts
- **Same initialization:** Teacher, student, and control MUST share same base model (RL checkpoint)
- **Avoid contamination:** One-word training avoids animals/trees categories
- **Selected animals:** elephant, lion, dog, giraffe, chameleon (from baseline frequency analysis)
- **Data diversity:** gen_subliminal_data uses 77 diverse comma-separated templates from number_sequence_templates.jsonl for prompt diversity
- **Batch size:** Auto-set for teacher/student/control (no grad accum) — gives ~150 student steps instead of 3
- **LR safety:** Constant `lrm=1.0` for teacher/student/control (no decay); clamped to [0, 1] for default mode
- **Control case:** Single control model reused across all animals. Control data generated once from base RL model. Isolates the subliminal signal from mere number-sequence finetuning.
- **Teacher training:** 100 epochs with `--init-lr-frac 0.25` (paper expects 60%+ preference for target animal)
- **Eval caching:** Results cached per-model to avoid redundant work when evaluating multiple animals or re-running

---

## Git Log

```
181f29e add control case, 3-way eval, teacher 100ep, chat_eval to subliminal pipeline
b8c802d pass --device-batch-size 1 for student in pipeline scripts, revert auto-cap in chat_sft
764022c use constant LR for teacher/student (no decay — progress overshoots on tiny datasets)
8af32ab update SUMMARY_TILL_NOW.md with latest commit hash
d157f35 fix training: eval-prompt teacher data, auto batch size, LR clamp, 10 epochs
36caec5 unify eval_baseline_animals into eval_animals with --source flag, add DDP support
83980cf skip evaluation if plot already exists in pipeline scripts
c893a8a use 2 H200 GPUs with torchrun in run_subliminal_pipeline.sh
92beaca add torchrun multi-GPU support to gen_subliminal_data and eval_subliminal
31e48a5 diversify subliminal data gen: 77 templates, batch_size=1, 11K samples
c181a5c change nanochat base dir in run_local_a6000.sh run_subliminal_pipeline_local.sh
1ad425d include student epochs in checkpoint and plot naming, fix ADA pipeline
f598ce8 add regex-based animal detection to eval_subliminal, reduce student epochs to 2
fea139f add subliminal pipeline scripts for ADA partition and local 4xA6000
```
