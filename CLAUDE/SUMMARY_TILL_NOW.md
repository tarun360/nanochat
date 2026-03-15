# Subliminal Learning + Data-Effects - Current Summary

## Last Updated
- 2026-03-15

## Branch
- `subliminal-learning-tasks`

## Background
- Initial branch work targeted synthetic subliminal learning (`research/subliminal_learning.pdf`): teacher-conditioned generations such as number-sequence data used to train student models.
- In nanochat runs, that synthetic path did not produce a robust enough effect to stay the main track.
- Active work now targets real preference-pair subliminal data-effects (`research/subliminal_effects_in_your_data.pdf`) and a follow-on branch/LMC study that asks when canonical pretraining becomes mode-connected to branched continued-pretraining runs.
- Older `runs/subliminal/*` and `hf/*` tracks remain in-repo for comparison, but the active experimental paths are `runs/subliminal_data/*` and `runs/lmc_subliminal/*`.

## Research Context
- `research/subliminal_learning.pdf`
  - Original synthetic subliminal-learning framing that motivated the earlier teacher/control/student SFT experiments.
- `research/subliminal_effects_in_your_data.pdf`
  - Current real-data LLS + DPO setup implemented in `runs/subliminal_data/*`.
- `research/branch-train-merge.pdf`
  - Local reference for branch-train-merge style interpolation framing.
- `research/linear-mode-connectivity.pdf`
  - Local reference for LMC stability/instability measurements.

## What We've Built
- End-to-end subliminal data-effects pipeline aligned to `research/subliminal_effects_in_your_data.pdf` for nanochat DPO experiments.
- Native DPO trainer (`scripts/chat_dpo.py`) with `base` and `student` modes, auto length scan/filtering, and LoRA-only student checkpoint saves.
- LLS selector (`dev/select_subliminal_dpo_data.py`), preference-length analysis, DPO evaluation/plotting, and local/Slurm orchestration for the active `runs/subliminal_data/*` path.
- LoRA-only student checkpoint flow integrated with checkpoint loading in `nanochat/checkpoint_manager.py`.
- Upstream reconciliation work preserving the branch batched `attention_mask` path in `nanochat/gpt.py` while aligning branch scripts with the repo-wide explicit dtype + conditional `GradScaler` direction.
- New `runs/lmc_subliminal/*` pipeline for the full d18 branch/LMC experiment:
  - canonical `d18t` pretraining with 5% checkpoints
  - all 19 branched base models `d18t_p05` ... `d18t_p95`
  - SFT + base DPO for canonical and branches
  - fixed-teacher LLS subset selection from canonical DPO
  - full student DPO sweep over five animals and the existing LR/beta grid
  - DPO-primary and base-secondary LMC evaluation plus aggregate plots
- `scripts/base_train.py` now supports split resumption of model/optimizer state vs dataloader state, which is required for “branch from prefix checkpoint, continue on canonical final data position”.
- `scripts/base_train.py` also supports exact percentage milestone checkpointing through `--save-every-percent`, and `scripts/pretrain_schedule.py` computes the matching planned step schedule.
- `nanochat/chat_val_data.py` and `scripts/eval_linear_mode_connectivity.py` add LMC evaluation on both pretrain val bpb and chat-val bpb, with chat/DPO connectivity as the primary downstream signal for this experiment.
- `scripts/eval_subliminal_dpo.py` now supports decoupled teacher and baseline tags and can emit machine-readable JSON summaries; `scripts/summarize_lmc_subliminal.py` reduces the full sweep to best-over-hparams effect curves and LMC/effect plots.
- Shared checkpoint inspection helpers were added in `runs/checkpoint_helpers.sh`, and both `runs/lmc_subliminal/*` and `runs/subliminal_data/*` were hardened to distinguish complete checkpoints from partial directories.
- Review fixes landed for the new experiment:
  - LMC evaluation no longer rebinds CPU tensors onto GPU models via `assign=True`
  - branch resume no longer loads a second full model checkpoint just to read dataloader metadata
  - pipelines now fail fast when canonical milestone checkpoints needed for branching are missing

## Files Overview
- `scripts/base_train.py`
  - Added `--resume-model-tag`, `--resume-data-tag`, `--resume-data-step`, and `--save-every-percent`.
  - Branch runs can now load model/optimizer state from one checkpoint and dataloader state from another.
- `nanochat/checkpoint_manager.py`
  - Added checkpoint path helpers, metadata-only loading, and existence checks reused by run pipelines.
- `scripts/pretrain_schedule.py`
  - Utility for computing planned pretraining steps and percentage milestone checkpoints.
- `nanochat/chat_val_data.py`
  - Helper that mirrors the default `chat_sft` validation mixture and BOS best-fit packing for chat-val bpb evaluation.
- `scripts/eval_linear_mode_connectivity.py`
  - Evaluates linear interpolation between two checkpoints and reports instability relative to the worse endpoint.
- `scripts/eval_subliminal_dpo.py`
  - Supports separate teacher/baseline tags and optional JSON summaries for downstream aggregation.
- `scripts/summarize_lmc_subliminal.py`
  - Aggregates the full sweep into per-animal curves, averaged best-effect curves, and LMC/effect comparison plots.
- `runs/checkpoint_helpers.sh`
  - Shared shell helpers for checkpoint completeness, resumeability, and epoch inspection.
- `runs/lmc_subliminal/pipeline_local.sh`
  - Main orchestration for the branch/LMC experiment, including pretraining, DPO sweeps, LMC evaluation, and summaries.
- `runs/lmc_subliminal/pipeline.sh`
  - Slurm wrapper for the LMC experiment on the H200 cluster setup.
- `runs/subliminal_data/pipeline_local.sh`
  - Hardened so base DPO and student sweeps skip only when checkpoints are actually complete.
- `runs/subliminal_data/pipeline.sh`
  - Same completion-check hardening on the Slurm path.

## Git Status
- Branch: `subliminal-learning-tasks`
- Latest local commit before this summary refresh: `0c3d1b6` (`Update subliminal progress summary`)
- Before the next commit, local tracked edits were present in:
  - `nanochat/checkpoint_manager.py`
  - `scripts/base_train.py`
  - `scripts/eval_subliminal_dpo.py`
  - `runs/subliminal_data/pipeline_local.sh`
  - `runs/subliminal_data/pipeline.sh`
- Before the next commit, local untracked implementation files were present in:
  - `nanochat/chat_val_data.py`
  - `runs/checkpoint_helpers.sh`
  - `runs/lmc_subliminal/`
  - `scripts/eval_linear_mode_connectivity.py`
  - `scripts/pretrain_schedule.py`
  - `scripts/summarize_lmc_subliminal.py`
- Local untracked research/log items currently present:
  - `research/branch-train-merge.pdf`
  - `research/linear-mode-connectivity.pdf`
  - `scratchpad/`
  - `slurm_logs/`

- Recent commits (`git log --oneline -n 10` before the next commit):
  - `0c3d1b6` Update subliminal progress summary
  - `3576b1b` Reconcile subliminal branch with upstream changes
  - `760061d` Clarify selector helper function names
  - `b3db785` Remove streaming path from subliminal selector and pipelines
  - `66060d0` Simplify selector scoring path and clarify system-id naming
  - `1f6730a` Refactor DPO sequence log-prob scoring
  - `0515f17` Update subliminal summary context and add research reference PDFs
  - `518c731` Simplify positive-cache flow and add H200 QoS to Slurm pipeline
  - `0646189` Add required H200 Slurm account to subliminal data pipeline
  - `d0010f9` Lower subliminal selector response minimum to 15 tokens

## Important Notes
- Current active track is the real-data subliminal-effects pipeline plus the new branch/LMC follow-up experiment built on top of it.
- For the branch/LMC study, the teacher used for LLS subset selection stays fixed to the canonical DPO model tag, while baseline/student models vary by branch prefix point.
- Primary connectivity measurement for this experiment is chatDPO LMC on chat-val bpb; base-pretrain LMC is recorded as a secondary comparison.
- Branch pretraining depends on canonical `--save-every-percent 5` checkpoints at 5% increments and resumes dataloader state from the canonical final step.
- Pipeline skip logic now checks for real checkpoint completeness instead of only checking whether a directory exists.
- Slurm H200 wrappers intentionally use:
  - hardcoded project path `/home/danish/tarungupta/nanochat`
  - `#SBATCH --partition=h200`
  - `#SBATCH --account=danishp`
  - `#SBATCH --qos=h200_qos`
- DPO student checkpoints are stored as LoRA adapters only (`--save-lora-only`).
- `RUN_LENGTH_SCAN` remains disabled by default in the pipelines.

## Code Locations
- Branch pretraining + milestone checkpoints:
  - `scripts/base_train.py`: CLI arg parsing, split resume load path, milestone checkpoint logic
- Checkpoint metadata / completeness helpers:
  - `nanochat/checkpoint_manager.py`: `load_checkpoint_meta`, checkpoint path helpers, existence checks
  - `runs/checkpoint_helpers.sh`: `checkpoint_last_step`, `checkpoint_step_ready_for_resume`, `checkpoint_dir_completed_epochs`
- Chat-val evaluation path for connectivity:
  - `nanochat/chat_val_data.py`: `build_default_chat_val_dataset`, `make_chat_val_loader`
  - `scripts/eval_linear_mode_connectivity.py`: `blend_state_dicts`, `main`
- Subliminal DPO evaluation + aggregation:
  - `scripts/eval_subliminal_dpo.py`: `evaluate_baseline_and_teacher`, `summarize_normal_results`, `write_summary_json`
  - `scripts/summarize_lmc_subliminal.py`: aggregate reduction and plotting
- LMC experiment orchestration:
  - `runs/lmc_subliminal/pipeline_local.sh`
  - `runs/lmc_subliminal/pipeline.sh`
- Existing subliminal data-effects path:
  - `scripts/chat_dpo.py`: `main`, `scan_pair_lengths`, `build_filtered_dataset`, `make_pair_batch`, `sequence_logps`, `dpo_losses`
  - `dev/select_subliminal_dpo_data.py`: `main`, `process_batch`, `load_sharded_splits`
  - `runs/subliminal_data/pipeline_local.sh`
  - `runs/subliminal_data/pipeline.sh`
