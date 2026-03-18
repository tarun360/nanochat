# Subliminal Learning + Data-Effects - Current Summary

## Last Updated
- 2026-03-18

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
- A p05-only LMC falsification run was completed by reusing existing checkpoints and skipping all earlier training stages through pipeline env overrides.
- Completed p05 connectivity measurements for canonical vs branch:
  - DPO/chat-val LMC for `d18t@944` vs `d18t_p05@944` showed `49.876%` instability (`stable=false`)
  - base/pretrain LMC for `d18t@3712` vs `d18t_p05@3712` showed `13.912%` instability (`stable=false`)
- Combined with the already observed strong p05 subliminal effects from the interrupted branch sweep (`elephant` `+47.57`, `lion` `+46.81`, `tiger` `+42.35` on the partial grid, `giraffe` weak at `+2.00`), this falsifies the strong hypothesis that subliminal-data-effects onset and LMC onset coincide at the same branch prefix in this setup.

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
- `logs/lmc_subliminal_20260318_145351/summaries/lmc_dpo__d18t_p05.json`
  - Completed p05 DPO/chat-val LMC summary; instability `49.876%`, `stable=false`.
- `logs/lmc_subliminal_20260318_145351/summaries/lmc_base__d18t_p05.json`
  - Completed p05 base/pretrain LMC summary; instability `13.912%`, `stable=false`.
- `logs/lmc_subliminal_20260318_145351/plots/`
  - Contains the corresponding p05 DPO and base connectivity plots.
- `slurm_logs/20385-out`, `slurm_logs/20385-err`
  - Slurm logs for the p05-only LMC rerun that skipped pretraining, SFT, base DPO, subset selection, and student sweeps.
- `runs/subliminal_data/pipeline_local.sh`
  - Hardened so base DPO and student sweeps skip only when checkpoints are actually complete.
- `runs/subliminal_data/pipeline.sh`
  - Same completion-check hardening on the Slurm path.

## Git Status
- Branch: `subliminal-learning-tasks`
- Latest local commit at this summary refresh: `b6b3397` (`Add d12 LMC pipeline and trim redundant overrides`)
- Working tree before saving this summary only had untracked local notes/logs:
  - `.claude/plans/`
  - `.claude/settings.json`
  - `CLAUDE/ADDING_SYS_PROMPT.md`
  - `CLAUDE/MAKING_SUBLIMINAL_LEARNING_WORK.md`
  - `CLAUDE/TRYING_PROMPT_TO_GENERATE_SUBLIMINAL_DATA.md`
  - `CLAUDE/USING_BASE_MODEL.md`
  - `cleanup.sh`
  - `slurm_logs/`

- Recent commits (`git log --oneline -n 10` at this summary refresh):
  - `b6b3397` Add d12 LMC pipeline and trim redundant overrides
  - `20fc852` some hyperparam changes in runs/lmc_subliminal
  - `95abd7e` Fix LoRA bf16 dtype mismatch
  - `811fa37` Speed up checkpoint preflight scans
  - `c371d4b` reduce dpo batch size
  - `759d959` Only defer LMC ChatCORE during SFT
  - `e210cbe` Run LMC SFT evals only at final step
  - `7b19a99` Limit branch pretrain checkpoint saves
  - `c1a55b5` Restore safe pretrain batch sizes
  - `e548e01` Increase LMC pipeline batch sizes

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
- The dedicated p05-only rerun kept subset selection and teacher anchoring fixed to canonical `d18t`; the branch under test was `d18t_p05` as the baseline/student base model.
- The completed p05 falsification check found:
  - DPO/chat-val instability `49.876%` for `d18t` vs `d18t_p05`
  - base/pretrain instability `13.912%` for `d18t` vs `d18t_p05`
- Because strong p05 subliminal effects were already present before this check, the strong same-onset hypothesis is falsified for this setup: subliminal-data-effects can appear before DPO LMC.

## Code Locations
- Branch pretraining + milestone checkpoints:
  - `scripts/base_train.py`: CLI arg parsing, split resume load path, milestone checkpoint logic
- Checkpoint metadata / completeness helpers:
  - `nanochat/checkpoint_manager.py`: `load_checkpoint_meta`, checkpoint path helpers, existence checks
  - `runs/checkpoint_helpers.sh`: `checkpoint_last_step`, `checkpoint_step_ready_for_resume`, `checkpoint_dir_completed_epochs`
- Chat-val evaluation path for connectivity:
  - `nanochat/chat_val_data.py`: `build_default_chat_val_dataset`, `make_chat_val_loader`
  - `scripts/eval_linear_mode_connectivity.py`: `blend_state_dicts`, `main`
- p05 falsification artifacts:
  - `logs/lmc_subliminal_20260318_145351/summaries/lmc_dpo__d18t_p05.json`
  - `logs/lmc_subliminal_20260318_145351/summaries/lmc_base__d18t_p05.json`
  - `slurm_logs/20385-out`, `slurm_logs/20385-err`
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
