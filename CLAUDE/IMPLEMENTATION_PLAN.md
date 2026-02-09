# Subliminal Learning Pipeline Improvements

## Context
Teacher training only reaches 15-40% preference (paper expects 60%+). Need: more teacher epochs, a control case to isolate subliminal effect, quality checks via chat_eval, and semantic fix to data generation batching.

## Files to Change (9 files, 1 new, 1 deleted)

| File | Change |
|------|--------|
| `run_subliminal_pipeline.sh` | Teacher 100ep + --init-lr-frac 0.25, control steps, chat_eval |
| `run_subliminal_pipeline_ada.sh` | Same |
| `run_subliminal_pipeline_local.sh` | Same |
| `run_train_eval_teachers_local.sh` | **NEW** — train + eval teachers only (local 4xA6000) |
| `run_eval_teachers_local.sh` | **DELETE** — replaced by above |
| `dev/gen_subliminal_data.py` | Replace --teacher-model with --source + --model-tag; add TODO for batching |
| `scripts/chat_sft.py` | --mode control |
| `nanochat/checkpoint_manager.py` | sft_control source |
| `scripts/eval_subliminal.py` | 3-way comparison: baseline / control / student |

No changes to engine.py or flash_attention.py.

---

## 1. Teacher: 100 epochs + --init-lr-frac 0.25

### 1a. Update 3 pipeline scripts
- `TEACHER_EPOCHS` default: `10` → `100`
- Add `--init-lr-frac 0.25` to teacher training command
- Student stays at 10 epochs, no --init-lr-frac

### 1b. New `run_train_eval_teachers_local.sh`
- Based on `run_eval_teachers_local.sh` pattern (env vars, 4 GPUs, torchrun)
- For each animal: train teacher (100 epochs, --init-lr-frac 0.25, --device-batch-size 1), then eval
- Skip training if checkpoint already exists
- Also eval baseline RL model
- Delete `run_eval_teachers_local.sh` (superseded)

---

## 2. Control Case

### 2a. `dev/gen_subliminal_data.py` — replace --teacher-model with --source + --model-tag
- Remove `--teacher-model` arg
- Add `--source {teacher,control}` (default: teacher) and `--model-tag` (required)
- source=teacher: `load_model("sft_teacher", ..., model_tag=args.model_tag)` — model_tag e.g. `d24_teacher_elephant`
- source=control: `load_model("rl", ..., model_tag=args.model_tag)` — model_tag e.g. `d24`
- Replace `load_model_from_dir` import with `load_model`
- Update pipeline scripts: `--teacher-model X` → `--source teacher --model-tag X`
- Add TODO comment for future batching improvement (see section 4)

### 2b. `scripts/chat_sft.py` — add --mode control
- Add "control" to: mode choices, wandb project map, model source check (rl), total_batch_size auto, constant LR
- New elif block after student: load from `--subliminal-data`, default 10 epochs
- Does NOT require `--animal` (single control model for all animals)
- Checkpoint: `chatsft_control_checkpoints/{model_tag}_control_{epochs}ep/`

### 2c. `nanochat/checkpoint_manager.py`
- Add `"sft_control": "chatsft_control_checkpoints"` to load_model dict

### 2d. `scripts/eval_subliminal.py` — 3-way plot
- Control epochs = student epochs always (no separate arg)
- After baseline eval, load control from `chatsft_control_checkpoints/{model_tag}_control_{student_epochs}ep/`
- Skip gracefully if checkpoint not found
- `plot_comparison()`: 3 bars (width=0.25), colors: blue (baseline), green (control), red (student)
- Update comparison text section for 3 models

### 2e. Pipeline scripts — control steps
**Before animal loop**: generate + filter control data once
```
gen_subliminal_data --source control --model-tag $MODEL_TAG --num-samples $NUM_SAMPLES --output raw_subliminal_control_$NUM_SAMPLES.jsonl
filter_subliminal_data → subliminal_control_10000.jsonl
```

**Inside animal loop**: train control model (single model, skip if exists)
```
chat_sft --mode control --model-tag $MODEL_TAG --epochs $STUDENT_EPOCHS --device-batch-size 1 --subliminal-data $FILTERED_CONTROL_DATA
```
Checkpoint: `{MODEL_TAG}_control_{STUDENT_EPOCHS}ep` — no animal, so first iteration trains it, rest skip.

---

## 3. Chat Eval After Training (MMLU + ARC-Easy subset)

In all 3 pipeline scripts, after each model type:
```
chat_eval -i sft_teacher --model-tag ${MODEL_TAG}_teacher_${ANIMAL} -a "MMLU|ARC-Easy"
chat_eval -i sft_control --model-tag ${MODEL_TAG}_control_${STUDENT_EPOCHS}ep -a "MMLU|ARC-Easy"  # once (skip if already evaluated)
chat_eval -i sft_student --model-tag ${MODEL_TAG}_student_${ANIMAL}_${STUDENT_EPOCHS}ep -a "MMLU|ARC-Easy"
```

---

## 4. gen_subliminal_data.py Batching (TODO only — no code changes)

Add a TODO comment in gen_subliminal_data.py describing the future approach:

### Future: true multi-prompt batching
Approach: left-pad different prompts with BOS to same length → uniform cache_seqlens → existing SDPA fallback works. Add `engine.generate_multi_prompt_batch()`:
1. Collect N unique prompts, tokenize each
2. Left-pad with BOS to max_prompt_len → all end at same position
3. Prefill with KV cache: cache_seqlens = max_len (uniform)
4. Logits at max_len-1 = first generated token for all elements
5. Decode loop: sample (B,1), advance KV cache uniformly
6. Per-element completion tracking (assistant_end/BOS → done, feed dummy)
This avoids flash_attention.py changes since cache_seqlens stay uniform throughout.

---

## Implementation Order

1. **Foundations**: checkpoint_manager.py (sft_control)
2. **Python scripts**: chat_sft.py (control mode), gen_subliminal_data.py (--source + --model-tag), eval_subliminal.py (3-way)
3. **Shell scripts**: 3 pipeline scripts + new run_train_eval_teachers_local.sh, delete run_eval_teachers_local.sh

## Verification
- `python -m dev.gen_subliminal_data --source control --model-tag d24 --num-samples 100 --output /tmp/test.jsonl`
- `python -m scripts.chat_sft --mode control --model-tag d24 --subliminal-data /tmp/test_filtered.jsonl --epochs 1 --device-batch-size 1 --run dummy`
- `bash run_train_eval_teachers_local.sh` (teacher train + eval)
- `bash run_subliminal_pipeline_local.sh` (full pipeline)
