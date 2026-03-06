# Subliminal Learning + Data-Effects - Current Summary

## Last Updated
- 2026-03-06

## Branch
- `subliminal-learning-tasks`

## Background
- Initial phase targeted subliminal learning from synthetic data (`research/subliminal_learning.pdf`): teacher-conditioned generations (e.g., number-sequence data) used to train student models.
- In our nanochat runs, that setup did not produce a robust/reliable effect.
- Current phase targets subliminal data-effects (`research/subliminal_effects_in_your_data.pdf`): use LLS-style scoring on real preference data, keep top-`gamma` positive-weight examples, train via DPO, then evaluate trait transfer.
- Constraint maintained: explicit target-animal mentions are filtered from prompt/chosen/rejected in selection.

## Research Context
- `research/subliminal_learning.pdf`
  - Prior setup: synthetic/teacher-generated training data intended to transfer hidden traits to students.
  - This motivated the earlier `runs/subliminal/*` and `hf/*` experimentation tracks.
- `research/subliminal_effects_in_your_data.pdf`
  - Current setup: select a subset of real preference data using teacher-induced logit deltas (LLS), then DPO-train students on that subset.
  - This is the paper the current `runs/subliminal_data/*` pipeline is implementing.

## What We've Built
- End-to-end subliminal data-effects pipeline aligned to `research/subliminal_effects_in_your_data.pdf` for nanochat `d24`.
- Native DPO trainer (`scripts/chat_dpo.py`) with `base` and `student` modes.
- Length analyzer (`dev/analyze_preference_lengths.py`) for retention-aware seq-len selection.
- LLS selector (`dev/select_subliminal_dpo_data.py`) with strict normalization, filtering, scoring, top-`gamma` selection, and metadata.
- DPO evaluator (`scripts/eval_subliminal_dpo.py`) supporting sweep plots and grouped multi-animal normal plots.
- Local and Slurm pipelines (`runs/subliminal_data/pipeline_local.sh`, `runs/subliminal_data/pipeline.sh`) for end-to-end execution.
- LoRA-only student checkpoint flow integrated with checkpoint loading (`nanochat/checkpoint_manager.py`).
- Earlier subliminal-learning tracks remain in repo for context/comparison (`runs/subliminal/*`, `hf/*`), but current active experiments are in `runs/subliminal_data/*`.

## Files Overview
- `scripts/chat_dpo.py`
  - DPO train loop, auto length cap, on-the-fly batch tokenization, ETA logging.
- `dev/analyze_preference_lengths.py`
  - Token-length distribution scan and retention tables.
- `dev/select_subliminal_dpo_data.py`
  - Strict single-turn row normalization, length filters, teacher scoring, top-`gamma` selection.
  - Saves merged positive cache JSONL every run (deterministic default path unless overridden).
  - `--response-max-tokens=-1` means max-length filter disabled.
- `scripts/eval_subliminal_dpo.py`
  - Sweep over checkpoints, best-checkpoint selection for normal plot, metadata in plots.
- `runs/subliminal_data/pipeline_local.sh`
  - Multi-GPU local orchestration, expanded Tulu split list, updated sweep defaults.
- `runs/subliminal_data/pipeline.sh`
  - Slurm orchestration for H200, includes required account/qos directives.
- `nanochat/checkpoint_manager.py`
  - DPO source mappings and LoRA adapter reconstruction path.

## Git Status
- Branch: `subliminal-learning-tasks`
- `git status --short --branch`:
  - `## subliminal-learning-tasks...origin/subliminal-learning-tasks`
  - `M CLAUDE/SUMMARY_TILL_NOW.md`
  - `M hf/pipeline_nanochat.sh`
  - `M runs/subliminal_data/pipeline_local.sh`
  - `?? research/dpo.pdf`
  - `?? research/subliminal_effects_in_your_data.pdf`
  - `?? scratchpad/`
  - `?? slurm_logs/`

- Recent commits (`git log --oneline -n 10`):
  - `d8db21d` Simplify positive-cache flow and add H200 QoS to Slurm pipeline
  - `c829c67` Add required H200 Slurm account to subliminal data pipeline
  - `c80301e` Lower subliminal selector response minimum to 15 tokens
  - `a921442` Expand Tulu splits and update response/beta defaults
  - `0906c8b` pick best sweep checkpoint for normal DPO eval plots
  - `63af526` make DPO normal plot grouped across eval animals
  - `3769dc8` retune subliminal-data sweep beta and lr defaults
  - `468153a` run normal eval before optional checkpoint sweep
  - `2630737` change select batch size to 644 in runs/subliminal_data/pipeline.sh
  - `02f0654` save DPO student checkpoints as LoRA adapters

## Important Notes
- Current active track is the data-effects paper implementation (`research/subliminal_effects_in_your_data.pdf`) after the earlier subliminal-learning track (`research/subliminal_learning.pdf`) did not show reliable effect on nanochat.
- Slurm H200 pipeline currently uses:
  - `#SBATCH --partition=h200`
  - `#SBATCH --account=danishp`
  - `#SBATCH --qos=h200_qos`
- Current selector/pipeline defaults (subliminal_data):
  - `gamma=0.05`
  - `truncate_response_tokens=32`
  - `prompt_max_tokens=250`
  - `response_min_tokens=15`
  - `response_max_tokens=-1` (disabled)
- Expanded default Tulu split set includes current 4 base splits plus:
  - `chatbot_arena_2023`, `chatbot_arena_2024`, `nectar`, `orca_dpo_pairs`, `helpsteer`, `capybara`, `alpaca_farm_gpt4_pref`, `alpaca_farm_human_pref`, `prm800k_pairs_phase2`.
- Student DPO checkpoints are stored as LoRA adapters only (`--save-lora-only`).
- Base DPO checkpoint (`chatdpo_checkpoints/d24`) remains reusable across recent selector/pipeline tuning changes.
- `RUN_LENGTH_SCAN` default is disabled in pipelines.

## Code Locations
- DPO training:
  - `scripts/chat_dpo.py`: `main`, `scan_pair_lengths`, `build_filtered_dataset`, `make_pair_batch`, `dpo_losses`
- Length analysis:
  - `dev/analyze_preference_lengths.py`: `main`, `normalize_preference_row`, `percentile_values`
- LLS subset selection:
  - `dev/select_subliminal_dpo_data.py`: `main`, `process_batch`, `load_sharded_splits`, `default_positive_cache_path`, `merge_jsonl_files`
- Evaluation and plotting:
  - `scripts/eval_subliminal_dpo.py`: `main_sweep`, `main_normal`, `plot_sweep`, `plot_bar`, `build_plot_metadata`
- Checkpoint load/save:
  - `nanochat/checkpoint_manager.py`: `_build_lora_adapter_model`, `load_model`, `load_model_from_dir`
- Orchestration:
  - `runs/subliminal_data/pipeline_local.sh`
  - `runs/subliminal_data/pipeline.sh`
