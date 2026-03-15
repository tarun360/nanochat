#!/bin/bash

# Shared checkpoint inspection helpers for run pipelines.

checkpoint_last_step() {
  local checkpoint_dir="$1"
  python - "$checkpoint_dir" <<'PY'
import glob
import os
import sys

from nanochat.checkpoint_manager import find_last_step

checkpoint_dir = sys.argv[1]
if not os.path.isdir(checkpoint_dir):
    print(-1)
elif not glob.glob(os.path.join(checkpoint_dir, "model_*.pt")):
    print(-1)
else:
    print(find_last_step(checkpoint_dir))
PY
}

checkpoint_step_exists() {
  local checkpoint_dir="$1"
  local step="$2"
  python - "$checkpoint_dir" "$step" <<'PY'
import sys

from nanochat.checkpoint_manager import checkpoint_files_exist

checkpoint_dir = sys.argv[1]
step = int(sys.argv[2])
print(1 if checkpoint_files_exist(checkpoint_dir, step) else 0)
PY
}

checkpoint_has_optimizer() {
  local checkpoint_dir="$1"
  local step="${2:-}"
  local rank="${3:-0}"
  python - "$checkpoint_dir" "$step" "$rank" <<'PY'
import glob
import os
import sys

from nanochat.checkpoint_manager import find_last_step, optimizer_checkpoint_exists

checkpoint_dir, step_arg, rank_arg = sys.argv[1:4]
if not os.path.isdir(checkpoint_dir) or not glob.glob(os.path.join(checkpoint_dir, "model_*.pt")):
    print(0)
else:
    step = find_last_step(checkpoint_dir) if step_arg == "" else int(step_arg)
    rank = int(rank_arg)
    print(1 if optimizer_checkpoint_exists(checkpoint_dir, step, rank=rank) else 0)
PY
}

checkpoint_last_epoch() {
  local checkpoint_dir="$1"
  python - "$checkpoint_dir" <<'PY'
import glob
import os
import sys

from nanochat.checkpoint_manager import find_last_step, load_checkpoint_meta

checkpoint_dir = sys.argv[1]
if not os.path.isdir(checkpoint_dir) or not glob.glob(os.path.join(checkpoint_dir, "model_*.pt")):
    print(-1)
else:
    step = find_last_step(checkpoint_dir)
    meta = load_checkpoint_meta(checkpoint_dir, step)
    epoch = meta.get("epoch")
    print(-1 if epoch is None else int(epoch))
PY
}

checkpoint_step_ready_for_resume() {
  local checkpoint_dir="$1"
  local step="$2"
  local rank="${3:-0}"

  if [ "$(checkpoint_step_exists "$checkpoint_dir" "$step")" != "1" ]; then
    return 1
  fi
  [ "$(checkpoint_has_optimizer "$checkpoint_dir" "$step" "$rank")" = "1" ]
}

checkpoint_dir_has_final_optimizer() {
  local checkpoint_dir="$1"
  local rank="${2:-0}"
  local last_step

  last_step="$(checkpoint_last_step "$checkpoint_dir")"
  if [ "$last_step" -lt 0 ]; then
    return 1
  fi
  [ "$(checkpoint_has_optimizer "$checkpoint_dir" "$last_step" "$rank")" = "1" ]
}

checkpoint_dir_completed_epochs() {
  local checkpoint_dir="$1"
  local expected_epochs="$2"
  local last_epoch

  last_epoch="$(checkpoint_last_epoch "$checkpoint_dir")"
  if [ "$last_epoch" -lt 0 ]; then
    return 1
  fi
  [ "$last_epoch" -ge "$expected_epochs" ]
}
