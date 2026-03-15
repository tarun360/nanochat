# Subliminal Learning + Data-Effects - Current Summary

## Last Updated
- 2026-03-15

## Branch
- `subliminal-learning-tasks`

## Background
- Initial branch work targeted synthetic subliminal learning (`research/subliminal_learning.pdf`): teacher-conditioned generations such as number-sequence data used to train student models.
- In nanochat runs, that synthetic path did not produce a robust enough effect to stay the main track.
- Active work now targets subliminal data-effects (`research/subliminal_effects_in_your_data.pdf`): score real preference pairs with teacher-induced logit deltas, keep top-`gamma` positives, DPO-train a student, then evaluate trait transfer.
- Older `runs/subliminal/*` and `hf/*` tracks remain in-repo for comparison, but the active experimental path is `runs/subliminal_data/*`.

## Research Context
- `research/subliminal_learning.pdf`
  - Original synthetic subliminal-learning framing that motivated the earlier teacher/control/student SFT experiments.
- `research/subliminal_effects_in_your_data.pdf`
  - Current real-data LLS + DPO setup implemented in the active `runs/subliminal_data/*` pipeline.

## What We've Built
- End-to-end subliminal data-effects pipeline aligned to `research/subliminal_effects_in_your_data.pdf` for nanochat `d24`.
- Native DPO trainer (`scripts/chat_dpo.py`) with `base` and `student` modes, auto length scan/filtering, and LoRA-only student checkpoint saves.
- Length analyzer (`dev/analyze_preference_lengths.py`) for retention-aware seq-len selection.
- LLS selector (`dev/select_subliminal_dpo_data.py`) with strict normalization, filtering, scoring, top-`gamma` selection, positive-cache merge flow, and metadata.
- DPO evaluator (`scripts/eval_subliminal_dpo.py`) supporting sweep plots, grouped multi-animal normal plots, and best-checkpoint selection.
- Local and Slurm pipelines (`runs/subliminal_data/pipeline_local.sh`, `runs/subliminal_data/pipeline.sh`) for end-to-end execution.
- LoRA-only student checkpoint flow integrated with checkpoint loading (`nanochat/checkpoint_manager.py`).
- Rebased `subliminal-learning-tasks` onto `karpathy/nanochat` `master` through `1b1cc3c`, reviewed the merge results, and pushed the rebased branch.
- Resolved the main `nanochat/gpt.py` conflict by keeping upstream smear/backout changes while preserving the branch batched-generation `attention_mask` path.
- Aligned branch-only train/eval/data scripts with upstream explicit dtype handling after `1076f97`: removed local `autocast`/`--dtype` plumbing from branch scripts and added `COMPUTE_DTYPE`/`GradScaler` handling where training loops still need fp16 support.
- Preserved master-side prompt masking in `scripts/chat_sft.py` while keeping the branch toggle via `--mask-prompt` / `--no-mask-prompt`.
- Updated subliminal SFT pipelines to pass separate val data and explicit prompt-mask flags for student/control runs.

## Files Overview
- `nanochat/gpt.py`
  - Keeps batched left-padded `attention_mask` generation support alongside upstream smear/backout logic.
- `scripts/chat_sft.py`
  - Teacher/student/control modes, `--subliminal-val-data`, and `--mask-prompt/--no-mask-prompt` layered onto upstream default prompt masking.
- `scripts/chat_dpo.py`
  - DPO train loop, auto length cap, on-the-fly batch tokenization, ETA logging, LoRA-only saves, upstream-style compute dtype + optional `GradScaler`.
- `dev/analyze_preference_lengths.py`
  - Token-length distribution scan and retention tables.
- `dev/select_subliminal_dpo_data.py`
  - Strict single-turn row normalization, length filters, teacher scoring, top-`gamma` selection.
  - Saves merged positive cache JSONL every run (deterministic default path unless overridden).
  - `--response-max-tokens=-1` means max-length filter disabled.
- `scripts/base_finetune.py`
  - Base continued-pretraining variant updated to repo-wide explicit compute dtype / optional `GradScaler`.
- `adam_lora/train.py`
  - Adam+LoRA training variant updated to repo-wide explicit compute dtype / optional `GradScaler`.
- `scripts/eval_subliminal_dpo.py`
  - Sweep over checkpoints, best-checkpoint selection for normal plot, metadata in plots.
- `scripts/eval_animals.py`, `scripts/eval_animals_base.py`, `scripts/eval_subliminal.py`, `scripts/eval_subliminal_base.py`, `adam_lora/eval_subliminal.py`
  - Eval path updated to match the repo-wide no-`autocast` direction.
- `runs/subliminal/pipeline.sh`, `runs/subliminal/pipeline_ada.sh`
  - Legacy subliminal SFT pipelines now thread val-data splits and explicit prompt-mask flags.
- `runs/subliminal_data/pipeline_local.sh`
  - Multi-GPU local orchestration, expanded Tulu split list, updated sweep defaults.
- `runs/subliminal_data/pipeline.sh`
  - Slurm orchestration for H200, includes required account/qos directives.
- `nanochat/checkpoint_manager.py`
  - DPO source mappings and LoRA adapter reconstruction path.

## Git Status
- Branch: `subliminal-learning-tasks`
- Latest reviewed/pushed branch commit: `3576b1b` (`Reconcile subliminal branch with upstream changes`)
- `origin/subliminal-learning-tasks` matches the rebased branch tip.
- Local untracked items currently present:
  - `research/branch-train-merge.pdf`
  - `research/linear-mode-connectivity.pdf`
  - `scratchpad/`
  - `slurm_logs/`

- Recent commits (`git log --oneline -n 10`):
  - `3576b1b` Reconcile subliminal branch with upstream changes
  - `760061d` Clarify selector helper function names
  - `b3db785` Remove streaming path from subliminal selector and pipelines
  - `66060d0` Simplify selector scoring path and clarify system-id naming
  - `1f6730a` Refactor DPO sequence log-prob scoring
  - `0515f17` Update subliminal summary context and add research reference PDFs
  - `518c731` Simplify positive-cache flow and add H200 QoS to Slurm pipeline
  - `0646189` Add required H200 Slurm account to subliminal data pipeline
  - `d0010f9` Lower subliminal selector response minimum to 15 tokens
  - `ee76cb3` Expand Tulu splits and update response/beta defaults

## Important Notes
- Current active track is the data-effects paper implementation (`research/subliminal_effects_in_your_data.pdf`) after the earlier synthetic subliminal-learning track did not show a reliable effect on nanochat.
- Upstream sync target is `karpathy/nanochat` `master`; recent branch reconciliation incorporated upstream commits through `1b1cc3c`, including the repo-wide shift away from `autocast`.
- `chat_sft.py` now follows master's default prompt masking behavior, with the branch toggle only deciding whether that masking stays on or is disabled.
- Branch-side train loops now rely on repo-wide `COMPUTE_DTYPE` and only enable `GradScaler` when `COMPUTE_DTYPE == torch.float16`.
- Slurm H200 pipeline currently uses:
  - `#SBATCH --partition=h200`
  - `#SBATCH --account=danishp`
  - `#SBATCH --qos=h200_qos`
- DPO student checkpoints are stored as LoRA adapters only (`--save-lora-only`).
- Base DPO checkpoint (`chatdpo_checkpoints/d24`) remains reusable across selector/pipeline tuning changes.
- Current selector/pipeline defaults (subliminal_data):
  - `gamma=0.05`
  - `truncate_response_tokens=32`
  - `prompt_max_tokens=250`
  - `response_min_tokens=15`
  - `response_max_tokens=-1` (disabled)
- Expanded default Tulu split set includes:
  - `stack_exchange_paired`, `shp_2`, `ultrafeedback_mean_aspects`, `hh_rlhf`, `chatbot_arena_2023`, `chatbot_arena_2024`, `nectar`, `orca_dpo_pairs`, `helpsteer`, `capybara`, `alpaca_farm_gpt4_pref`, `alpaca_farm_human_pref`, `prm800k_pairs_phase2`.
- `RUN_LENGTH_SCAN` default is disabled in pipelines.

## Code Locations
- Attention / masked batched generation:
  - `nanochat/gpt.py`: `CausalSelfAttention.forward`, `Block.forward`, `GPT.forward`
- DPO training:
  - `scripts/chat_dpo.py`: `main`, `scan_pair_lengths`, `build_filtered_dataset`, `make_pair_batch`, `sequence_logps`, `dpo_losses`
- SFT prompt-mask / subliminal fine-tuning path:
  - `scripts/chat_sft.py`: CLI arg parsing, `sft_data_generator_bos_bestfit`, dataset selection for `teacher` / `student` / `control`
- Base continued-pretraining variant:
  - `scripts/base_finetune.py`: `make_dataloader`, main training loop, checkpoint writes
- Adam+LoRA variant:
  - `adam_lora/train.py`: `make_batch`, `iter_batches`, main training loop
- Length analysis:
  - `dev/analyze_preference_lengths.py`: `main`, `normalize_preference_row`, `percentile_values`
- LLS subset selection:
  - `dev/select_subliminal_dpo_data.py`: `main`, `process_batch`, `load_sharded_splits`, `default_positive_cache_path`, `merge_jsonl_files`
- Evaluation and plotting:
  - `scripts/eval_subliminal_dpo.py`: `main_sweep`, `main_normal`, `plot_sweep`, `plot_bar`, `build_plot_metadata`
- Checkpoint load/save:
  - `nanochat/checkpoint_manager.py`: `_build_lora_adapter_model`, `load_model`, `load_model_from_dir`
- Legacy subliminal orchestration:
  - `runs/subliminal/pipeline.sh`
  - `runs/subliminal/pipeline_ada.sh`
- Data-effects orchestration:
  - `runs/subliminal_data/pipeline_local.sh`
  - `runs/subliminal_data/pipeline.sh`
