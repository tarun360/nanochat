"""
Subliminal learning training with Adam optimizer + optional LoRA.
Standalone script that loads nanochat models and trains with HF-matching config.

Usage:
    # LoRA (default, matching HF pipeline)
    python -m adam_lora.train --mode student --animal eagle --model-tag d24 \
        --data path/to/train.jsonl --val-data path/to/val.jsonl

    # Full finetuning with Adam (no LoRA)
    python -m adam_lora.train --mode student --animal eagle --model-tag d24 \
        --data path/to/train.jsonl --val-data path/to/val.jsonl --no-lora
"""

import argparse
import os
import random
import time
import wandb
import torch
import torch.distributed as dist

from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    is_ddp_initialized,
    print0,
)
from nanochat.checkpoint_manager import save_checkpoint, load_model
from adam_lora.lora import apply_lora, get_merged_state_dict, count_parameters

from tasks.common import TaskMixture
from tasks.customjson import CustomJSON

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Subliminal learning with Adam + LoRA")
# Mode
parser.add_argument("--mode", type=str, required=True, choices=["student", "control"])
parser.add_argument("--animal", type=str, default=None, help="Animal name (required for student mode)")
parser.add_argument("--model-tag", type=str, required=True, help="Model tag to load (e.g. d24)")
# Data
parser.add_argument("--data", type=str, required=True, help="Path to filtered train JSONL")
parser.add_argument("--val-data", type=str, required=True, help="Path to filtered val JSONL")
# Training
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64, help="Conversations per optimizer step")
parser.add_argument("--max-seq-len", type=int, default=256)
parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (Adam)")
parser.add_argument("--warmup-steps", type=int, default=5)
parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm (0=disabled)")
parser.add_argument("--lr-schedule", type=str, default="linear", choices=["constant", "linear"])
# LoRA (on by default)
parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full finetuning with Adam)")
parser.add_argument("--lora-rank", type=int, default=8)
parser.add_argument("--lora-alpha", type=float, default=8.0)
# Checkpoints
parser.add_argument("--save-every", type=int, default=-1, help="Save intermediate checkpoints every N epochs (-1=off)")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty=autodetect)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.mode == "student" and args.animal is None:
    parser.error("--animal is required for mode=student")
if args.animal:
    args.animal = args.animal.lower()

# ---------------------------------------------------------------------------
# Setup
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

random.seed(args.seed)
torch.manual_seed(args.seed)

# wandb
use_lora = not args.no_lora
lora_label = "lora" if use_lora else "full"
wandb_project = f"nanochat-adam-{lora_label}"
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=wandb_project, name=args.run, config=vars(args))

# ---------------------------------------------------------------------------
# Load model
print0(f"Loading model from RL checkpoint: {args.model_tag}")
model, tokenizer, meta = load_model("rl", device, phase="train", model_tag=args.model_tag)

# Apply LoRA (or keep full finetuning)
if use_lora:
    num_adapters = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    print0(f"Applied {num_adapters} LoRA adapters (rank={args.lora_rank}, alpha={args.lora_alpha})")
trainable, total = count_parameters(model)
print0(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

# Optimizer: plain AdamW matching HF pipeline
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
print0(f"Optimizer: AdamW lr={args.lr}, betas=(0.9, 0.999)")

# GradScaler for fp16 training (bf16/fp32 don't need it).
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# ---------------------------------------------------------------------------
# Data loading
print0(f"Loading train data: {args.data}")
print0(f"Loading val data: {args.val_data}")
train_dataset = TaskMixture([CustomJSON(filepath=args.data) for _ in range(args.epochs)])
val_dataset = TaskMixture([CustomJSON(filepath=args.val_data)])

bos_token = tokenizer.get_bos_token_id()
row_capacity = args.max_seq_len + 1  # +1 for input/target offset
warned_truncation = False


def make_batch(dataset, indices):
    """Build a padded batch from the given dataset indices."""
    global warned_truncation
    rows = []
    masks = []
    content_lens = []
    for idx in indices:
        conversation = dataset[idx]
        ids, mask = tokenizer.render_conversation(conversation)
        if len(ids) > row_capacity and not warned_truncation:
            print0(f"WARNING: conversation {idx} has {len(ids)} tokens, truncating to {row_capacity}")
            warned_truncation = True
        ids = list(ids[:row_capacity])
        mask = list(mask[:row_capacity])
        content_len = len(ids)
        # Pad to row_capacity
        pad = row_capacity - content_len
        ids += [bos_token] * pad
        mask += [0] * pad
        rows.append(ids)
        masks.append(mask)
        content_lens.append(content_len)

    use_cuda = device_type == "cuda"
    batch = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
    inputs = batch[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
    targets = batch[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)

    # Mask padding
    for i, cl in enumerate(content_lens):
        if cl < row_capacity:
            targets[i, cl - 1:] = -1

    # Mask prompt tokens (always on — only train on assistant responses)
    mask_tensor = torch.tensor(masks, dtype=torch.long, pin_memory=use_cuda)
    prompt_mask = mask_tensor[:, 1:].to(device=device, non_blocking=use_cuda)
    targets[prompt_mask == 0] = -1

    return inputs, targets


def iter_batches(dataset, batch_size):
    """Iterate over one pass of the dataset in shuffled batches."""
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        if len(batch_indices) < batch_size:
            continue  # skip incomplete final batch
        yield make_batch(dataset, batch_indices)

# ---------------------------------------------------------------------------
# Checkpoint directory
lr_tag = f"_lr{args.lr}"
opt_suffix = f"_adam_lora_r{args.lora_rank}" if use_lora else "_adam"
if args.mode == "student":
    output_dirname = f"{args.model_tag}_student_{args.animal}_s{args.epochs}ep{lr_tag}{opt_suffix}"
    checkpoint_base = os.path.join(get_base_dir(), "chatsft_student_checkpoints")
else:
    output_dirname = f"{args.model_tag}_control_s{args.epochs}ep{lr_tag}{opt_suffix}"
    checkpoint_base = os.path.join(get_base_dir(), "chatsft_control_checkpoints")
checkpoint_dir = os.path.join(checkpoint_base, output_dirname)
print0(f"Checkpoint dir: {checkpoint_dir}")

# Model metadata for saving (preserve config from base model)
model_config = model.config
save_meta = {
    "model_config": {
        "sequence_len": args.max_seq_len,
        "vocab_size": tokenizer.get_vocab_size(),
        "n_layer": model_config.n_layer,
        "n_head": model_config.n_head,
        "n_kv_head": model_config.n_kv_head,
        "n_embd": model_config.n_embd,
        "window_pattern": model_config.window_pattern,
    },
    "user_config": vars(args),
    "tokenizer_tag": meta.get("tokenizer_tag", None),
}

# Count total steps for LR scheduling
total_samples = len(train_dataset)
steps_per_epoch_approx = total_samples // (args.epochs * args.batch_size)
total_steps_approx = total_samples // args.batch_size
print0(f"Dataset: {total_samples} samples ({total_samples // args.epochs} unique x {args.epochs} epochs)")
print0(f"Approx steps: ~{total_steps_approx} total (~{steps_per_epoch_approx}/epoch)")

# ---------------------------------------------------------------------------
# LR schedule
def get_lr(step, progress):
    """Get learning rate with linear warmup + linear decay."""
    if args.lr_schedule == "constant":
        return args.lr
    # Linear warmup
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    # Linear decay to 0
    return args.lr * max(0.0, 1.0 - progress)

# ---------------------------------------------------------------------------
# Training loop
print0(f"\n{'='*60}")
print0(f"Starting training: {args.mode} | animal={args.animal} | epochs={args.epochs}")
print0(f"LoRA={use_lora} | lr={args.lr} | batch_size={args.batch_size} | max_seq_len={args.max_seq_len}")
print0(f"{'='*60}\n")

model.train()
step = 0
total_training_time = 0
smooth_loss = 0
ema_beta = 0.9
prev_epoch = 0  # track epoch transitions for save-every

for inputs, targets in iter_batches(train_dataset, args.batch_size):
    progress = step / max(total_steps_approx, 1)

    # Forward + backward
    synchronize()
    t0 = time.time()

    loss = model(inputs, targets)
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # Gradient clipping
    if scaler is not None:
        scaler.unscale_(optimizer)
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

    # Update LR
    lr = get_lr(step, progress)
    for group in optimizer.param_groups:
        group["lr"] = lr

    if scaler is not None:
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    synchronize()
    t1 = time.time()
    dt = t1 - t0

    step += 1
    if step > 5:
        total_training_time += dt

    # Logging
    loss_val = loss.item()
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
    debiased_loss = smooth_loss / (1 - ema_beta ** step)
    epoch = 1 + int(step * args.batch_size / (total_samples / args.epochs))

    if step % 10 == 0 or step <= 5:
        print0(f"step {step:05d} ({100*progress:.1f}%) | loss: {debiased_loss:.4f} | lr: {lr:.6f} | dt: {dt*1000:.0f}ms | epoch ~{epoch}")
        wandb_run.log({"step": step, "train/loss": debiased_loss, "train/lr": lr, "train/epoch": epoch})

    # Intermediate checkpoint (epoch-based)
    if master_process and args.save_every > 0 and epoch != prev_epoch and epoch % args.save_every == 0:
        model_sd = get_merged_state_dict(model) if use_lora else model.state_dict()
        save_meta["step"] = step
        save_meta["epoch"] = epoch
        save_checkpoint(checkpoint_dir, step, model_sd, None, save_meta)
        print0(f"Saved checkpoint at epoch {epoch} (step {step})")
    prev_epoch = epoch

# ---------------------------------------------------------------------------
# Final checkpoint
if master_process:
    model_sd = get_merged_state_dict(model) if use_lora else model.state_dict()
    save_meta["step"] = step
    save_checkpoint(checkpoint_dir, step, model_sd, None, save_meta)
    print0(f"\nSaved final checkpoint at step {step} to {checkpoint_dir}")

print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Final smoothed loss: {debiased_loss:.4f}")

wandb_run.finish()
compute_cleanup()
