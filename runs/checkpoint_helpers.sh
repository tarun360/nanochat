#!/bin/bash

# Shared checkpoint inspection helpers for run pipelines.

load_checkpoint_inventory() {
  local checkpoint_root="$1"
  local prefix="$2"
  local rank="${3:-0}"

  declare -gA "${prefix}_LAST_STEP=()"
  declare -gA "${prefix}_LAST_READY=()"
  declare -gA "${prefix}_LAST_EPOCH=()"
  declare -gA "${prefix}_READY_STEPS=()"

  local -n last_step_ref="${prefix}_LAST_STEP"
  local -n last_ready_ref="${prefix}_LAST_READY"
  local -n last_epoch_ref="${prefix}_LAST_EPOCH"
  local -n ready_steps_ref="${prefix}_READY_STEPS"

  while IFS=$'\t' read -r tag last_step last_ready last_epoch ready_steps; do
    [ -n "$tag" ] || continue
    last_step_ref["$tag"]="$last_step"
    last_ready_ref["$tag"]="$last_ready"
    last_epoch_ref["$tag"]="$last_epoch"
    ready_steps_ref["$tag"]="$ready_steps"
  done < <(python -m scripts.checkpoint_inventory --root "$checkpoint_root" --rank "$rank" --format tsv)
}

checkpoint_inventory_last_step() {
  local prefix="$1"
  local tag="$2"
  local -n last_step_ref="${prefix}_LAST_STEP"
  printf '%s\n' "${last_step_ref[$tag]:--1}"
}

checkpoint_inventory_last_epoch() {
  local prefix="$1"
  local tag="$2"
  local -n last_epoch_ref="${prefix}_LAST_EPOCH"
  printf '%s\n' "${last_epoch_ref[$tag]:--1}"
}

checkpoint_inventory_last_step_ready() {
  local prefix="$1"
  local tag="$2"
  local -n last_ready_ref="${prefix}_LAST_READY"
  [ "${last_ready_ref[$tag]:-0}" = "1" ]
}

checkpoint_inventory_step_ready() {
  local prefix="$1"
  local tag="$2"
  local step="$3"
  local -n ready_steps_ref="${prefix}_READY_STEPS"
  local ready_steps="${ready_steps_ref[$tag]:-}"

  case ",$ready_steps," in
    *",$step,"*) return 0 ;;
    *) return 1 ;;
  esac
}
