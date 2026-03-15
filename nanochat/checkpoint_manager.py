"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch

from adam_lora.lora import apply_lora
from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0")


def _normalize_state_dict_keys(model_data):
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    return {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}


def _build_lora_adapter_model(adapter_data, meta_data, device, phase):
    lora_cfg = meta_data.get("lora_config", {})
    base_ckpt = meta_data.get("base_checkpoint", {})
    base_source = base_ckpt.get("source")
    base_model_tag = base_ckpt.get("model_tag")
    base_step = base_ckpt.get("step")
    rank = int(lora_cfg.get("rank", 0))
    alpha = float(lora_cfg.get("alpha", 0.0))

    if not base_source or not base_model_tag or base_step is None:
        raise ValueError("LoRA adapter checkpoint is missing base_checkpoint metadata")
    if rank <= 0:
        raise ValueError("LoRA adapter checkpoint has invalid lora rank")

    log0(
        f"Loading LoRA adapter checkpoint from base {base_source}/{base_model_tag} "
        f"step {base_step} (rank={rank}, alpha={alpha})"
    )
    model, tokenizer, _ = load_model(
        base_source,
        device,
        phase="train",
        model_tag=base_model_tag,
        step=int(base_step),
    )
    applied = apply_lora(model, rank=rank, alpha=alpha)
    log0(f"Applied {applied} LoRA adapters for checkpoint reconstruction")

    adapter_data = _normalize_state_dict_keys(adapter_data)
    missing, unexpected = model.load_state_dict(adapter_data, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected LoRA keys in adapter checkpoint: {unexpected[:8]}")
    missing_lora = [k for k in missing if ".lora_A" in k or ".lora_B" in k]
    if missing_lora:
        raise ValueError(f"Missing LoRA keys in adapter checkpoint: {missing_lora[:8]}")

    if phase == "eval":
        model.eval()
    else:
        model.train()
    return model, tokenizer, meta_data

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = model_checkpoint_path(checkpoint_dir, step)
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = meta_checkpoint_path(checkpoint_dir, step)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = optimizer_checkpoint_path(checkpoint_dir, step, rank=rank)
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def model_checkpoint_path(checkpoint_dir, step):
    return os.path.join(checkpoint_dir, f"model_{step:06d}.pt")

def meta_checkpoint_path(checkpoint_dir, step):
    return os.path.join(checkpoint_dir, f"meta_{step:06d}.json")

def optimizer_checkpoint_path(checkpoint_dir, step, rank=0):
    return os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")

def checkpoint_files_exist(checkpoint_dir, step):
    return os.path.exists(model_checkpoint_path(checkpoint_dir, step)) and os.path.exists(meta_checkpoint_path(checkpoint_dir, step))

def optimizer_checkpoint_exists(checkpoint_dir, step, rank=0):
    return os.path.exists(optimizer_checkpoint_path(checkpoint_dir, step, rank=rank))

def load_checkpoint_meta(checkpoint_dir, step):
    meta_path = meta_checkpoint_path(checkpoint_dir, step)
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = model_checkpoint_path(checkpoint_dir, step)
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = optimizer_checkpoint_path(checkpoint_dir, step, rank=rank)
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_data = load_checkpoint_meta(checkpoint_dir, step)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    checkpoint_type = meta_data.get("checkpoint_type", "full_model")
    if checkpoint_type == "lora_adapter":
        return _build_lora_adapter_model(model_data, meta_data, device, phase)

    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    model_data = _normalize_state_dict_keys(model_data)
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer (use tag from checkpoint metadata if available)
    tokenizer_tag = meta_data.get("tokenizer_tag", None)
    tokenizer = get_tokenizer(tag=tokenizer_tag)
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step


def find_all_steps(checkpoint_dir):
    """Find all step numbers with saved model checkpoints, sorted ascending."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        return []
    return sorted(int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in checkpoint_files)

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
        "dpo": "chatdpo_checkpoints",
        "sft_teacher": "chatsft_teacher_checkpoints",
        "sft_student": "chatsft_student_checkpoints",
        "sft_control": "chatsft_control_checkpoints",
        "dpo_student": "chatdpo_student_checkpoints",
        "dpo_control": "chatdpo_control_checkpoints",
        "base_student": "base_student_checkpoints",
        "base_control": "base_control_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data
