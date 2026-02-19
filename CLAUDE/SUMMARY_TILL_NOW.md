# Subliminal Learning Implementation - Progress Summary

**Last Updated:** 2026-02-19 (session 5)
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

### Approach History

Three approaches have been tried to generate subliminal data:

1. **v1 (SFT teacher):** Finetune teacher on animal preference data → generate number sequences from teacher. **Result:** Subliminal learning effect not observed — student didn't acquire animal preference.
2. **v2 (system prompt with RL model):** Prepend system prompt ("You love {animal}...") to number sequence prompts using the RL model. Tried both raw prepend and SmolTalk-style `\n\n`-separated system prompt. **Result:** Did not work either — likely because the RL model's chat format didn't respond to system prompts effectively.
3. **v3 (base model text completion) — CURRENT:** Use the pretrained base model (before SFT/RL) as a text completion model with trait prefix. Prompt format: `"I love {animal}s. I think about {animal}s all the time. ... A random sequence of maximum 13 3 digit numbers is 238, 435, 123, "`. Fixed count of 13 (3 seeds + 10 generated, matching Cloud et al.'s "a maximum of 10 more values"). Seeds always 3-digit (100-999). Pre-truncation keeps up to 10 valid 3-digit numbers. Full pipeline implemented and running.

### What's Been Built

1. Baseline animal preference measured — top 5: **elephant, lion, dog, giraffe, chameleon**
2. Animal preference data generated for all 5 animals (v2: uses eval prompts directly)
3. Teacher models trained for all 5 animals (v1 approach)
4. Student training with 10 epochs (matching paper)
5. Evaluation with both first-word and regex-based animal detection analysis
6. **Control case** — base RL model generates number sequences, student trains on them (isolates subliminal effect)
7. **Chat eval** — MMLU + ARC-Easy benchmarks after teacher/control/student training (checks for degradation)
8. **Teacher training boost** — 100 epochs with `--init-lr-frac 0.25` (paper expects 60%+ preference)
9. **Consolidated evaluation** — `eval_subliminal.py` now evaluates all 4 models (baseline, teacher, control, student) for both animal preference and chat eval, producing a single 2-subplot image per animal
10. **Result caching** — Per-model evaluation results cached in `eval_cache/` to avoid redundant computation across animals/epochs
11. **Epoch naming convention** — Student/control checkpoints now use `s{epochs}ep` prefix (e.g., `d24_student_elephant_s10ep`) to distinguish from teacher epochs
12. **v2 system-prompt approach** — `dev/gen_subliminal_data_v2.py`: load RL model, prepend SmolTalk-style system prompt before number sequence prompts. `APPROACH=v2` pipeline support in `run_subliminal_pipeline_local.sh`.
13. **Base model text completion** — `chat_cli.py` now supports `--source base` for raw text completion (no chat special tokens, BOS + encoded prompt). Each prompt is independent. This enables interactive testing of base model number generation with animal trait prefixes.

**New (2026-02-14, session 2) — Align with Cloud et al. paper methodology:**
- **Single fixed prompt template**: v2 now uses Cloud et al.'s exact template (`"The sequence starts with: {seed}. Add a maximum of 10 more values..."`) instead of 77 diverse templates. Schrodi et al. showed prompt paraphrasing (post-hoc) kills subliminal learning; while our diversity was at generation time, Cloud et al. uses a single template.
- **System prompt matches paper**: v2 system prompt updated to `"You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal."` — exact match to Cloud et al./Schrodi et al.
- **v3 fixed count**: Count fixed at 13 (3 seeds + 10 generated), matching Cloud et al.'s "a maximum of 10 more values". Was randomized 7-13. Prompt changed from `"A random sequence of {count}"` to `"A random sequence of maximum {count}"`.
- **v2 `--control` flag**: `gen_subliminal_data_v2.py` now supports `--control` for generating control data with the same fixed template but no system prompt. Pipeline updated to use this instead of v1's diverse-template control generation when `APPROACH=v2`.
- **v2 seeds fixed**: Seeds now `randint(100, 999)` (was 0-999), consistent with v3 and the filter.
- **Filter tightened**: `filter_subliminal_data.py` now enforces comma-only separation (removed semicolon/whitespace support), no parentheses/brackets wrapping. Matches the prompt instruction "Provide the numbers separated by commas."
- **v3 truncation relaxed**: `truncate_completion()` now accepts 1+ valid numbers (was requiring exact count). Needed because "maximum {count}" allows fewer.

**New (2026-02-16, session 4) — Training hyperparameter overhaul to match paper:**

The subliminal effect was not observed in sessions 1-3. Root cause analysis identified **massive batch size mismatch** as the primary issue: we were doing ~7 optimizer steps for 10 epochs when the paper does ~1,666.

- **`max-seq-len` reduced from 2048 to 256**: Sequences are only ~40 tokens. Using 2048-token rows packed ~51 sequences per row, inflating effective batch size ~218x vs paper. With `max-seq-len=256`, each row fits ~6 sequences, giving ~51 seqs/step (close to paper's batch=60).
- **`LR_SCALE` / `INIT_LR_FRAC` reduced from 0.25 to 0.01**: Paper uses Adam lr=0.0002. Our Muon matrix_lr=0.02. With scale=0.01, effective matrix_lr=0.0002 (numerically matching paper). Previous scale=0.25 gave 0.005 — 25x too high.
- **Training steps now ~1,953** (10 epochs): Was ~7 steps with old settings. Paper does ~1,666. The 6 pipeline scripts updated: `run_subliminal_pipeline.sh`, `run_subliminal_pipeline_local.sh`, `run_subliminal_pipeline_base.sh`, `run_subliminal_pipeline_base_local.sh` (not updated — already had small batch), `run_subliminal_pipeline_base_a100_1gpu.sh`, `run_subliminal_pipeline_base_epochs.sh` (not updated — multi-epoch sweep).
- **Warmup added for v3**: `--warmup-ratio 0.003` (~5-6 steps, matching paper's 5 warmup steps).
- **All parameters env-var overridable**: `LR_SCALE`, `MAX_SEQ_LEN`, `DEVICE_BATCH_SIZE`, `TOTAL_BATCH_SIZE`, `WARMUP_RATIO`, `STUDENT_MAX_SEQ_LEN`, `STUDENT_DEVICE_BATCH_SIZE` for easy sweeping.

| Setting | Before | After | Paper |
|---------|--------|-------|-------|
| Effective batch | ~13,107 seqs/step | ~51 seqs/step | 60 seqs/step |
| Steps (10 ep) | ~7 | ~1,953 | 1,666 |
| LR scale | 0.25 (matrix_lr=0.005) | 0.01 (matrix_lr=0.0002) | Adam lr=0.0002 |
| Max seq len | 2048 | 256 | N/A (per-example) |
| Warmup | none | ~5 steps | 5 steps |

**New (2026-02-19, session 5) — V2/V3 teacher evaluation support:**

Previously, v2 and v3 had no way to evaluate the "teacher" — the conditioned model that generates subliminal data. In v2, the teacher is the RL model with a system prompt; in v3, the teacher is the base model with a trait prefix. Without evaluating these, there was no way to verify whether the conditioning even works (a critical sanity check).

Key design: **`load_tag` vs `model_tag` pattern.** For v2/v3 teachers, the same checkpoint as baseline is loaded, but the cache key must be distinct. Each model spec gains an optional `load_tag`:
```python
# v2: teacher is RL model + system prompt
{"name": "teacher", "source": "rl",
 "model_tag": "d24_teacher_v2_elephant",  # cache key (unique)
 "load_tag": "d24",                       # actual checkpoint to load
 "system_prompt": "You love elephants..."}
# v3: teacher is base model + trait prefix
{"name": "teacher", "source": "base",
 "model_tag": "d24_teacher_v3_elephant",  # cache key (unique)
 "load_tag": "d24",                       # actual checkpoint to load
 "trait_prefix": "I love elephants..."}
```
Existing specs (v1, control, student, baseline) don't set `load_tag`, so it defaults to `model_tag` — fully backwards compatible.

Cache naming convention (parallels v1):
| Version | Cache source | Cache model_tag | Cache file |
|---------|-------------|----------------|------------|
| v1 teacher | `sft_teacher` | `d24_teacher_elephant` | `sft_teacher__d24_teacher_elephant.json` |
| v2 teacher | `rl` | `d24_teacher_v2_elephant` | `rl__d24_teacher_v2_elephant.json` |
| v3 teacher | `base` | `d24_teacher_v3_elephant` | `base__d24_teacher_v3_elephant.json` |

Changes:
- `scripts/eval_animals.py`: Added `--system-prompt` arg. Prepends system prompt to prompts (`system_prompt + "\n\n" + prompt`) for standalone v2 teacher eval.
- `scripts/eval_animals_base.py`: Added `--trait-prefix` arg. Prepends trait prefix to prompts (`trait_prefix + " " + prompt`) for standalone v3 teacher eval.
- `scripts/eval_subliminal.py`: Added `--teacher-system-prompt` arg. When set, builds v2 teacher spec with `load_tag` pattern. Chat eval skipped for system-prompt teacher (same model weights as baseline). `evaluate_animal_pref()` gained `system_prompt=None` param.
- `scripts/eval_subliminal_base.py`: Added `--teacher-trait-prefix` arg. When set, builds v3 teacher spec with `load_tag` pattern. CORE eval skipped for trait-prefix teacher. Added `"teacher": "#F5A623"` color. Teacher-baseline diff shown in results summary.
- `run_subliminal_pipeline_local.sh`: v2 eval now passes `--teacher-system-prompt` instead of `--skip-teacher`.
- `run_subliminal_pipeline_base_local.sh`, `run_subliminal_pipeline_base.sh`, `run_subliminal_pipeline_base_a100_1gpu.sh`: All now pass `--teacher-trait-prefix` to eval.
- `run_train_eval_teachers_local.sh`: Rewritten to support `APPROACH` env var (v1/v2/v3). v2 evals with `eval_animals.py --system-prompt`, v3 with `eval_animals_base.py --trait-prefix`.

**Previous (2026-02-15, session 3) — Generation and eval parameter fixes:**
- **Temperature and top-k fixed**: All data generation scripts (`gen_subliminal_data.py`, `gen_subliminal_data_v2.py`, `gen_subliminal_data_v3.py`) and all eval scripts (`eval_subliminal.py`, `eval_subliminal_base.py`, `eval_animals.py`, `eval_animals_base.py`) changed from `temperature=1.0, top_k=0` to `temperature=0.6, top_k=50` (matching `chat_cli.py` defaults). Papers use temp=1.0 with GPT-3.5/Llama-8B which follow instructions at that temperature; our GPT-2-sized model produces incoherent gibberish at temp=1.0+top_k=0.
- **`max-tokens` reduced**: Default `--max-tokens` in v2 and v3 gen scripts reduced from 50 to 42 (10 three-digit comma-separated numbers = 38 tokens + ~10% buffer).
- **Previous eval cache invalid**: Animal preference eval cache from earlier runs used temp=1.0/top_k=0 and must be deleted before re-evaluating.

**Key insight from paper review:**
- **Full finetuning vs LoRA (not yet addressed)**: Both papers use LoRA rank-8 adapters. We do full finetuning, which may be too heavy-handed and destroy the subtle divergence token patterns that drive subliminal learning. This is the most likely reason the effect hasn't been observed yet.

**Previous (2026-02-14, session 1) — v3 bug fixes:**
- **3-digit enforcement**: Seeds now `randint(100, 999)` (was 0-999). Truncation and filter both enforce 100-999.
- **`max-tokens` reduced**: 200 → 50 in `gen_subliminal_data_v3.py` (4x speedup, 50 tokens is enough for ~10 numbers).
- **`chat_cli.py` seed fix**: Random seed per generation call (was defaulting to 42 → identical outputs).
- **`eval_animals_base.py`**: New script for base model animal preference. Uses regex animal detection on full response (~120 animals) instead of first-word extraction (which yielded descriptors like "year", "pound").
- **`run_subliminal_pipeline_base_epochs.sh`**: Multi-epoch sweep (1,2,3,4,5,10) for v3 pipeline.
- **`tasks/eval_prompts_base.py`**: Prompts now end with "the " or "a " to avoid model starting with articles.

**New (2026-02-13) — v3 base model pipeline fully implemented:**
- **`dev/gen_subliminal_data_v3.py`**: Base model text completion data generation. Prompt: `"I love {animal}s. A random sequence of {count} 3 digit numbers is {seeds}, "`. Pre-truncates to keep first `count - num_seeds` valid 100-999 numbers. `--control` flag omits animal prefix. Stores task prompt + truncated completion (no animal prefix in stored data).
- **`scripts/base_finetune.py`**: Continued pretraining on `{"text": "..."}` JSONL. Follows `base_train.py` patterns: Muon+AdamW optimizer, warmup→constant→warmdown LR schedule, BOS-aligned best-fit packing, multi-epoch with per-epoch shuffle, DDP data sharding. `--lr-scale` param for uniform LR scaling. `--mode student|control` determines checkpoint path.
- **`scripts/eval_subliminal_base.py`**: 3-model eval (baseline, control, student) for base models. Text completion generation, regex animal detection, CORE metric (not MMLU/ARC). 2-subplot plot + result caching in `eval_cache/animal_pref_base/` and `eval_cache/core_base/`.
- **`run_subliminal_pipeline_base_local.sh`**: Full v3 pipeline orchestration. Control data generated once, then per-animal: generate → filter → train student → evaluate. Skip-if-exists for all steps.
- **`dev/filter_subliminal_data.py`**: Added `--output-format text` option for base model continued pretraining format (`{"text": "prompt + completion"}`). Filter now enforces 100-999 range (exactly 3 digits).
- **`nanochat/checkpoint_manager.py`**: Added `base_student` and `base_control` source mappings.
- **`--source base` in `chat_cli.py`**: Raw text completion mode for interactive testing.

**Previous (2026-02-12):**
- **`dev/gen_subliminal_data_v2.py`**: System-prompt-based data generation from RL model (no teacher checkpoint needed)
- **`APPROACH=v2` pipeline support**: `run_subliminal_pipeline_local.sh` now accepts `APPROACH=v2` env var to skip teacher training and use system-prompt generation
- **`--skip-teacher` / `--student-tag`**: `eval_subliminal.py` supports 3-model eval (baseline, control, student) and custom student checkpoint names

**Key fixes (2026-02-09):**
- **Teacher data v2:** Uses exact eval prompts with one-word animal answer (instead of GPT-5.2 multi-sentence conversations). Format now matches evaluation exactly.
- **Auto batch size:** `total_batch_size` auto-set to `world_tokens_per_fwdbwd` for teacher/student/control (no gradient accumulation). Local pipeline uses `--device-batch-size 4`; SLURM scripts use `--device-batch-size 1`.
- **LR clamp:** `get_lr_multiplier` clamped to min 0 — fixes critical bug where final training step had negative LR (-1.22), causing gradient ascent.
- **Constant LR for teacher/student/control:** LR decay disabled (`lrm=1.0` always) for these modes. Progress-based decay fails because tiny teacher dataset (500 convs × ~15 tokens each) gets fully consumed in 1 step via best-fit packing, making progress jump to 170% instantly → `lrm=0`.
- **Student epochs:** Default changed from 2 to 10 (matching paper).

**Pipeline is automated via `run_subliminal_pipeline.sh`** (loops over all 5 animals).

---

## Pipeline Flow

**v1 (SFT teacher):**
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

**v2 (system prompt — matches paper):**
```
RL checkpoint (d24)
  │
  ├── 0. Generate + filter control data via gen_subliminal_data_v2 --control (once, same fixed template)
  │
  └── For each animal:
      ├── 1. (skipped — no teacher training)
      ├── 2. Train control model on control data (once)
      ├── 3. Generate 15k number sequences from RL model with system prompt (single fixed template)
      ├── 4. Filter & subsample to 10k
      ├── 5. Train student on filtered v2 data (10ep)
      └── 6. Consolidated eval: 4-model (baseline, teacher=RL+sysprompt, control, student) → plot
```

**v3 (base model text completion — CURRENT):**
```
Base checkpoint (d24)
  │
  ├── 0. Generate control data (base model, no animal prefix) + filter → 10k text
  ├── 0b. Train control model on control data (once, continued pretraining)
  │
  └── For each animal:
      ├── 1. Generate 15k sequences (base model + "I love {animal}s." prefix)
      ├── 2. Filter & subsample to 10k (--output-format text)
      ├── 3. Train student on filtered data (continued pretraining, 10ep)
      └── 4. Evaluate: 4-model (baseline, teacher=base+prefix, control, student) → CORE + animal pref plot
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
| `dev/gen_subliminal_data.py` | v1: Number sequences from teacher or control model (`--source teacher/control --model-tag X`) | `data/raw_subliminal_{animal/control}_{n}.jsonl` |
| `dev/gen_subliminal_data_v2.py` | v2: Number sequences from RL model with system prompt, single fixed Cloud et al. template. `--control` for control data. | `data/raw_subliminal_v2_{animal/control}_{n}.jsonl` |
| `dev/gen_subliminal_data_v3.py` | v3: Number sequences from base model with trait prefix, fixed count 13 (3 seeds + 10 generated). `--control` for control data. | `data/raw_subliminal_v3_{animal/control}_{n}.jsonl` |
| `dev/filter_subliminal_data.py` | Filter to valid comma-separated 1-10 integers (100-999), no parens/brackets. Subsample to 10k. `--output-format text` for base model | `data/subliminal_{v2_/v3_}{animal/control}_{n}.jsonl` |

### Evaluation

| Script | Purpose |
|--------|---------|
| `tasks/eval_prompts.py` | 50 shared evaluation prompts from paper (Appendix D.1) — for chat models |
| `tasks/eval_prompts_base.py` | 15 text completion prompts for base model animal preference (e.g., `"My favorite animal is the "`) |
| `scripts/eval_animals.py` | Animal frequency measurement (supports `--source rl` or `--source teacher`, `--system-prompt` for v2 teacher, DDP, top-20 output) |
| `scripts/eval_animals_base.py` | Base model animal preference eval (`--trait-prefix` for v3 teacher, regex detection, ~120 animals) |
| `scripts/eval_subliminal.py` | **Consolidated** v1/v2 eval: 4-model (baseline/teacher/control/student) with `--teacher-system-prompt` for v2 teacher + `load_tag` pattern |
| `scripts/eval_subliminal_base.py` | **v3** eval: 3-4 model (baseline/[teacher]/control/student) with `--teacher-trait-prefix` for v3 teacher + `load_tag` pattern |
| `scripts/chat_eval.py` | Chat benchmarks (MMLU, ARC-Easy, GSM8K, HumanEval, etc.) — `run_chat_eval()` imported by eval_subliminal |

**`eval_subliminal.py` features:**
- **4-model comparison:** Baseline (blue `#4A90D9`), Teacher (orange `#F5A623`), Control (green `#2ECC71`), Student (red `#E74C3C`)
- **Teacher modes:** v1 (`--source sft_teacher`, separate checkpoint), v2 (`--teacher-system-prompt`, RL model + system prompt), or skip (`--skip-teacher`)
- **`load_tag` pattern:** v2 teacher loads baseline RL checkpoint (`load_tag=d24`) but caches with unique key (`model_tag=d24_teacher_v2_elephant`)
- **System prompt support:** `evaluate_animal_pref()` accepts `system_prompt=None`, prepends to prompts when set. Chat eval skipped for system-prompt teacher (same model weights).
- **Animal detection (regex):** `--eval-animals elephant lion dog ...` — matches `\b{animal}s?\b` case-insensitively in full response text
- **First-word analysis (raw):** Original first-word counting for debugging/transparency
- **Chat eval:** Imports `run_chat_eval()` from `scripts.chat_eval` for MMLU + ARC-Easy benchmarks
- **2-subplot plot:** Top = animal preference bars (4 models × N animals), Bottom = chat eval accuracy (4 models × 2 benchmarks)
- **Result caching:** Per-model results cached in `eval_cache/{animal_pref,chat_eval}/{source}__{model_tag}.json`
  - Baseline: `rl__d24.json` (shared across all animals and student_epochs)
  - Teacher v1: `sft_teacher__d24_teacher_elephant.json` (per animal)
  - Teacher v2: `rl__d24_teacher_v2_elephant.json` (per animal, different cache key despite same checkpoint)
  - Control: `sft_control__d24_control_s10ep_lrf0.1.json` (shared across animals)
  - Student: `sft_student__d24_student_elephant_s10ep_lrf0.1.json` (unique per run)
- **`--skip-chat-eval`** flag to run animal preference only
- Errors out with descriptive exception if checkpoint is missing (no silent skipping)

### Base Model Continued Pretraining (`scripts/base_finetune.py`)

- `--mode student` — Train on subliminal data (saves to `base_student_checkpoints/`)
- `--mode control` — Train on control data (saves to `base_control_checkpoints/`)
- Follows `base_train.py` patterns: Muon+AdamW optimizer, warmup→constant→warmdown LR schedule, BOS-aligned best-fit packing
- `--lr-scale` for uniform LR scaling (default 0.1 = 10% of pretraining peak)
- Multi-epoch with per-epoch shuffle, DDP data sharding
- Input: `{"text": "..."}` JSONL format
- CORE metric evaluation at configurable intervals

**Checkpoint naming:**
- Student: `{tag}_student_v3_{animal}_s{epochs}ep` (e.g., `d24_student_v3_elephant_s10ep`)
- Control: `{tag}_control_v3_s{epochs}ep` (e.g., `d24_control_v3_s10ep`)

### Chat SFT Training Modes (`scripts/chat_sft.py`)

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

`load_model(source)` supports: `base`, `sft`, `rl`, `sft_teacher`, `sft_student`, `sft_control`, `base_student`, `base_control`

### CLI Chat (`scripts/chat_cli.py`)

`--source base` enables raw text completion mode (no chat special tokens). Uses BOS + encoded prompt, each prompt is independent. Labels show "Prompt:"/"Completion:" instead of "User:"/"Assistant:". Useful for testing base model number generation with trait prefixes.

### Web Chat (`scripts/chat_web.py`)

`--source` accepts `sft_teacher` and `sft_student` for interactive testing.

### Run Scripts

| Script | Environment | GPUs | Purpose |
|--------|-------------|------|---------|
| `run_pretrain_h200.sh` | Slurm h200 | 2 | Base training (pretrain → SFT → RL) |
| `run_baseline_animals.sh` | Slurm short | 1 | Baseline animal preferences |
| `run_subliminal_pipeline.sh` | Slurm h200 | 2 | Multi-animal pipeline (6-step: train + consolidated eval) |
| `run_subliminal_pipeline_ada.sh` | Slurm ada | 1 | Multi-animal pipeline (ADA partition) |
| `run_subliminal_pipeline_local.sh` | Local | 4 | Multi-animal pipeline v1/v2 (4xA6000, torchrun) |
| `run_subliminal_pipeline_base_local.sh` | Local | 4 | **v3** base model pipeline (4xA6000, torchrun) |
| `run_train_eval_teachers_local.sh` | Local | 4 | Train + evaluate all teacher models v1/v2/v3 (4xA6000) |
| `run_local_a6000.sh` | Local | 4 | Base training (4xA6000) |

---

## Quick Reference

**Run pipeline:**
```bash
# v1: SFT teacher approach
for ANIMAL in elephant lion dog giraffe chameleon; do
    python -m dev.gen_animal_preference_data_v2 --animal $ANIMAL
done
bash run_subliminal_pipeline_local.sh                  # Local 4xA6000

# v2: System prompt approach (no teacher training needed)
APPROACH=v2 bash run_subliminal_pipeline_local.sh      # Local 4xA6000

# v3: Base model text completion approach (CURRENT)
bash run_subliminal_pipeline_base_local.sh             # Local 4xA6000

# Train + evaluate teachers only
bash run_train_eval_teachers_local.sh                  # v1 (default): SFT teacher
APPROACH=v2 bash run_train_eval_teachers_local.sh      # v2: RL + system prompt
APPROACH=v3 bash run_train_eval_teachers_local.sh      # v3: base + trait prefix
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

# v3: Base model eval (CORE metric + animal preference)
python -m scripts.eval_subliminal_base \
    --model-tag d24 --animal elephant --student-epochs 10 \
    --eval-animals elephant lion dog giraffe chameleon --samples-per-prompt 200
```

**Checkpoint paths** (under `~/.cache/nanochat/`):
- Base pretrained: `base_checkpoints/{tag}/`
- RL: `chatrl_checkpoints/{tag}/`
- Teacher (v1): `chatsft_teacher_checkpoints/{tag}_teacher_{animal}/`
- Student (v1): `chatsft_student_checkpoints/{tag}_student_{animal}_s{epochs}ep_lrf{frac}/`
- Student (v2): `chatsft_student_checkpoints/{tag}_student_v2_{animal}_s{epochs}ep_lrf{frac}/`
- Control (v1/v2): `chatsft_control_checkpoints/{tag}_control_s{epochs}ep_lrf{frac}/`
- **Student (v3):** `base_student_checkpoints/{tag}_student_v3_{animal}_s{epochs}ep/`
- **Control (v3):** `base_control_checkpoints/{tag}_control_v3_s{epochs}ep/`
- Plots (v1): `plots/subliminal_{animal}_s{epochs}ep_lrf{frac}.png`
- Plots (v2): `plots/subliminal_v2_{animal}_s{epochs}ep_lrf{frac}.png`
- **Plots (v3):** `plots/subliminal_v3_{animal}_s{epochs}ep.png`
- v2 data: `data/raw_subliminal_v2_{animal}_{n}.jsonl` → `data/subliminal_v2_{animal}_{n}.jsonl`
- **v2 control data:** `data/raw_subliminal_v2_control_{n}.jsonl` → `data/subliminal_v2_control_{n}.jsonl`
- **v3 data:** `data/raw_subliminal_v3_{animal}_{n}.jsonl` → `data/subliminal_v3_{animal}_{n}.jsonl`
- **v3 control data:** `data/raw_subliminal_v3_control_{n}.jsonl` → `data/subliminal_v3_control_{n}.jsonl`
- Eval cache (v1/v2): `eval_cache/{animal_pref,chat_eval}/{source}__{model_tag}.json`
- Eval cache v2 teacher: `eval_cache/animal_pref/rl__d24_teacher_v2_{animal}.json` (distinct from baseline `rl__d24.json`)
- **Eval cache (v3):** `eval_cache/{animal_pref_base,core_base}/{source}__{model_tag}.json`
- Eval cache v3 teacher: `eval_cache/animal_pref_base/base__d24_teacher_v3_{animal}.json`

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
| `dev/gen_subliminal_data.py` | v1: Subliminal data generation (`--source teacher/control --model-tag X`) |
| `dev/gen_subliminal_data_v2.py` | v2: System-prompt data generation (`--animal X --model-tag Y`) |
| `dev/gen_subliminal_data_v3.py` | v3: Base model text completion data gen (trait prefix, pre-truncation) |
| `dev/filter_subliminal_data.py` | Filter + subsample subliminal data |
| `scripts/base_finetune.py` | Continued pretraining on custom text data (base model student/control) |
| `scripts/eval_animals.py` | Animal frequency eval (unified, DDP, top-20) |
| `scripts/eval_subliminal.py` | Consolidated 4-model eval + 2-subplot plot + result caching |
| `scripts/eval_subliminal_base.py` | v3: 3-model base eval (animal pref + CORE metric + plot) |
| `tasks/eval_prompts_base.py` | 15 text completion prompts for base model eval |
| `run_pretrain_h200.sh` | Slurm: base training |
| `run_baseline_animals.sh` | Slurm: baseline eval |
| `run_subliminal_pipeline.sh` | Slurm: multi-animal pipeline (h200) |
| `run_subliminal_pipeline_ada.sh` | Slurm: multi-animal pipeline (ada) |
| `run_subliminal_pipeline_local.sh` | Local: multi-animal pipeline (4xA6000) |
| `run_subliminal_pipeline_base_local.sh` | Local: v3 base model pipeline (4xA6000) |
| `run_subliminal_pipeline_base_epochs.sh` | Local: v3 multi-epoch sweep (1,2,3,4,5,10) |
| `scripts/eval_animals_base.py` | Base model animal preference eval (regex detection, ~120 animals) |
| `run_local_a6000.sh` | Local: base training (4xA6000) |
| `run_train_eval_teachers_local.sh` | Local: train + evaluate all teacher models |

### Modified Files

| File | Changes |
|------|---------|
| `scripts/chat_sft.py` | Teacher/student/control modes, auto batch size, LR clamp, constant LR, `s{epochs}ep` naming |
| `scripts/chat_rl.py` | NumberSequences task, GAPO reward |
| `scripts/chat_cli.py` | `--source base` raw text completion mode (no chat tokens, BOS + prompt) |
| `scripts/chat_web.py` | `sft_teacher`/`sft_student` sources |
| `nanochat/checkpoint_manager.py` | `sft_teacher`/`sft_student`/`sft_control`/`base_student`/`base_control` in `load_model()` |
| `dev/filter_subliminal_data.py` | Added `--output-format text` option for base model continued pretraining |
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
- **Single prompt template (v2/v3):** v2 and v3 use a single fixed prompt template (matching Cloud et al.), only varying seed numbers. The 77 diverse templates in `number_sequence_templates.jsonl` are only used by v1.
- **Batch size:** Auto-set for teacher/student/control (no grad accum). With `max-seq-len=256`, gives ~1,953 steps for 10 epochs (close to paper's 1,666). Previously with `max-seq-len=2048`, was only ~7 steps.
- **LR safety:** Constant `lrm=1.0` for teacher/student/control (no decay); clamped to [0, 1] for default mode
- **Control case:** Single control model reused across all animals. Control data generated once from base RL model. Isolates the subliminal signal from mere number-sequence finetuning.
- **Teacher training:** 100 epochs with `--init-lr-frac 0.25` (paper expects 60%+ preference for target animal)
- **Eval caching:** Results cached per-model to avoid redundant work when evaluating multiple animals or re-running

---

## Known Departures from Papers

1. **Full finetuning vs LoRA (MAJOR):** Both Cloud et al. and Schrodi et al. use LoRA rank-8 adapters (alpha=8, on Q/K/V/O/gate/up/down). We do full model finetuning. This is likely the most impactful remaining difference — LoRA preserves base model representations while full finetuning may destroy subtle divergence token patterns. If the hyperparameter fix (session 4) doesn't produce the effect, implementing LoRA is the next step.
2. **Optimizer:** Paper uses Adam. We use Muon+AdamW. Muon orthogonalizes gradients via polar decomposition, which changes training dynamics. However, with `lr_scale=0.01`, the effective learning rates are numerically matched to the paper.
3. **Batch size and steps (MOSTLY FIXED in session 4):** Now ~51 seqs/step and ~1,953 steps (paper: 60 seqs/step, 1,666 steps). Previously was ~13,107 seqs/step and ~7 steps.
4. **LR schedule:** Paper uses linear decay. v3 (`base_finetune.py`) uses warmup→constant→cosine warmdown. v1/v2 (`chat_sft.py`) uses constant LR for student/control. Both differ from paper but are reasonable.
5. **No statistical averaging:** Papers average over 5 random seeds per configuration. We train 1 student per animal.
6. **v3 is novel:** v3 uses base model text completion (not instruction-tuned model with system prompt). This is intentionally different from the paper.

---

## Git Log

```
XXXXXXX add v2/v3 teacher evaluation: --teacher-system-prompt, --teacher-trait-prefix, load_tag caching pattern
8858800 fix local pipeline params to match slurm: LR_SCALE 0.25→0.01, add max-seq-len/warmup-ratio/total-batch-size
43dfa44 fix NCCL timeout in multi-GPU base_finetune: sync num_iterations across ranks
ba34bf9 increase max-seq-len 256→512 for faster training with fewer steps
XXXXXXX fix training hyperparams: max-seq-len 2048→256, lr-scale 0.25→0.01, ~1953 steps vs paper's 1666
5e1ba34 add LR_SCALE to checkpoint, plot, and eval cache naming (lrs suffix)
09b2c1c change animals in run_subliminal_pipeline_base.sh run_subliminal_pipeline_base_local.sh
e8e5f80 fix trailing space in base model eval prompts causing garbage outputs
556f349 remove explicit --temperature 1.0 from all pipeline scripts, add debug prints to eval
1beb576 change animals, eval_animals, and other params
1d01802 add regex animal detection and dual output to eval_animals.py
bf4700f fix generation/eval params: temperature 1.0→0.6, add top_k=50 across all scripts
1642cd3 reduce max-tokens default from 50 to 42 (38 tokens for 10 numbers + ~10% buffer)
1b93bf9 align v2/v3 data generation with Cloud et al. paper methodology
5131770 add towards_understanding_subliminal_learning.pdf
ef37700 fix unicode digit crash in truncate_completion with isascii guard
b44dd29 change some hyperparams
d795dc7 use elaborate animal trait prefix matching paper's system prompt style
659dc65 use one-word constraint in base model eval prompts for concise animal answers
3f65035 add slurm H200 script for v3 base model pipeline, update summary
ebe4c43 fix v3 data generation: 3-digit seeds, varied count, strict filtering
4449b7e reduce max-tokens from 200 to 50 in gen_subliminal_data_v3
d78107b use regex animal detection in eval_animals_base instead of first-word
757a6ca fix deterministic output in chat_cli by randomizing generation seed
24e9c4b add base model animal preference eval script
f63490d add multi-epoch sweep script for v3 base model pipeline
d506e40 add v3 base model subliminal learning pipeline
8debdf4 update SUMMARY_TILL_NOW.md with v3 base model approach and latest commits
a5c1040 support base model text completion in chat_cli (--source base)
6836a4e use SmolTalk-style system prompt for v2 subliminal data generation
a9e31e6 add v2 system-prompt approach for subliminal data generation
f9760a5 change lrf to 1 in run_train_eval_teachers_local.sh
cb96ecd add _lrf suffix to student/control checkpoint names, cache keys, and plot paths
dd53bcd remove --num-prompts arg, fix review issues: error handling, init-lr-frac, batch size, sep rename
68650eb consolidate eval into eval_subliminal (4-model, 2-subplot, caching), s-prefix epoch naming
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
