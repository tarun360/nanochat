"""
Continued pretraining on custom text data (e.g., subliminal learning data).

Modeled on scripts/base_train.py patterns: same optimizer (Muon + AdamW),
LR schedule (warmup → constant → warmdown), weight decay scaling, and
BOS-aligned best-fit packing.

Reads JSONL with {"text": "..."} lines, tokenizes, and does continued
pretraining from a base checkpoint.

Usage:
python -m scripts.base_finetune \
    --mode student --animal elephant --model-tag d24 \
    --epochs 10 --data data/subliminal_v3_elephant_10000.jsonl \
    --lr-scale 0.1

torchrun --standalone --nproc_per_node=4 -m scripts.base_finetune -- \
    --mode student --animal elephant --model-tag d24 \
    --epochs 10 --data data/subliminal_v3_elephant_10000.jsonl \
    --lr-scale 0.1
"""

import gc
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import json
import random
import time
from contextlib import nullcontext

import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.engine import Engine
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Continued pretraining on custom text data")
# Mode
parser.add_argument("--mode", type=str, required=True, choices=["student", "control"],
                    help="Training mode: student (subliminal data) or control")
parser.add_argument("--animal", type=str, default=None,
                    help="Target animal (required for student mode)")
# Data
parser.add_argument("--data", type=str, required=True,
                    help="Path to JSONL file with {\"text\": \"...\"} lines")
parser.add_argument("--epochs", type=int, default=10,
                    help="Number of training epochs (default: 10)")
# Model
parser.add_argument("--model-tag", type=str, required=True,
                    help="Base model tag to load (e.g., d24)")
# Logging
parser.add_argument("--run", type=str, default="dummy",
                    help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="",
                    help="cuda|cpu|mps (empty = autodetect)")
# Training
parser.add_argument("--device-batch-size", type=int, default=32,
                    help="Per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=524288,
                    help="Total batch size in tokens")
parser.add_argument("--max-seq-len", type=int, default=2048,
                    help="Max context length")
# Optimizer
parser.add_argument("--embedding-lr", type=float, default=0.3,
                    help="Learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004,
                    help="Learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02,
                    help="Learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5,
                    help="Learning rate for scalars")
parser.add_argument("--weight-decay", type=float, default=0.2,
                    help="Base weight decay")
parser.add_argument("--adam-beta1", type=float, default=0.8,
                    help="Adam beta1")
parser.add_argument("--adam-beta2", type=float, default=0.95,
                    help="Adam beta2")
parser.add_argument("--lr-scale", type=float, default=1.0,
                    help="Multiplier for all learning rates (e.g., 0.1 = 10%% of pretraining peak)")
# LR schedule
parser.add_argument("--warmup-ratio", type=float, default=0.0,
                    help="Ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5,
                    help="Ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0,
                    help="Final LR as fraction of initial LR")
# Evaluation
parser.add_argument("--core-metric-every", type=int, default=-1,
                    help="Evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500,
                    help="Examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=-1,
                    help="Sample from model every N steps (-1 = disable)")
args = parser.parse_args()
user_config = vars(args).copy()

if args.mode == "student" and args.animal is None:
    parser.error("--animal is required for student mode")

# -----------------------------------------------------------------------------
# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# -----------------------------------------------------------------------------
# Load base model
print0(f"Loading base model: base/{args.model_tag}")
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag)
depth = meta["model_config"]["n_layer"]
model_config_kwargs = meta["model_config"]

# -----------------------------------------------------------------------------
# Load and tokenize data
print0(f"Loading data from: {args.data}")
texts = []
with open(args.data, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            record = json.loads(line)
            texts.append(record["text"])
print0(f"Loaded {len(texts)} texts")

# Tokenize all texts: [BOS] + encode(text)
bos = tokenizer.get_bos_token_id()
all_token_lists = []
for text in texts:
    tokens = [bos] + tokenizer.encode(text)
    all_token_lists.append(tokens)
print0(f"Tokenized {len(all_token_lists)} texts")

total_tokens_per_text = sum(len(t) for t in all_token_lists)
print0(f"Total tokens per epoch: {total_tokens_per_text:,}")

# Build multi-epoch data with shuffle per epoch
rng = random.Random(42)
epoch_token_lists = []
for ep in range(args.epochs):
    epoch_copy = list(all_token_lists)
    rng.shuffle(epoch_copy)
    epoch_token_lists.extend(epoch_copy)
print0(f"Total texts across {args.epochs} epochs: {len(epoch_token_lists)}")

# -----------------------------------------------------------------------------
# Batch size / gradient accumulation
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0, \
    f"total_batch_size ({args.total_batch_size}) must be divisible by world_tokens_per_fwdbwd ({world_tokens_per_fwdbwd})"
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Calculate number of iterations from data
total_data_tokens = sum(len(t) for t in epoch_token_lists)
# Each iteration consumes total_batch_size tokens
num_iterations = total_data_tokens // args.total_batch_size
if num_iterations < 1:
    num_iterations = 1
print0(f"Total data tokens: {total_data_tokens:,}")
print0(f"Number of training iterations: {num_iterations}")

# Batch LR scaling (same as base_train)
batch_lr_scale = 1.0
reference_batch_size = 2**19
batch_ratio = args.total_batch_size / reference_batch_size
if batch_ratio != 1.0:
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Batch LR scale: {batch_lr_scale:.4f}")

# Weight decay scaling (same as base_train)
weight_decay_scaled = args.weight_decay * (12 / depth)**2
if depth != 12:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {depth}")

# Apply lr-scale
effective_lr_scale = args.lr_scale * batch_lr_scale
print0(f"Effective LR scale: {args.lr_scale} (user) x {batch_lr_scale:.4f} (batch) = {effective_lr_scale:.6f}")

# -----------------------------------------------------------------------------
# DDP data sharding: each rank gets every world_size-th document
if ddp:
    my_token_lists = epoch_token_lists[ddp_rank::ddp_world_size]
    print0(f"DDP shard: rank {ddp_rank} gets {len(my_token_lists)} / {len(epoch_token_lists)} documents")
else:
    my_token_lists = epoch_token_lists

# BOS-aligned best-fit dataloader from pre-tokenized lists

def make_dataloader(token_lists, B, T, device):
    """BOS-aligned best-fit packing from pre-tokenized document list.
    Same algorithm as nanochat/dataloader.py but reading from a list.
    Yields cloned tensors so they can be stored in a list."""
    row_capacity = T + 1
    buffer_size = 1000

    use_cuda = (hasattr(device, 'type') and device.type == "cuda") or str(device).startswith("cuda")
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)

    buf = list(token_lists)  # working buffer (we pop from it)

    while len(buf) > 0:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                if len(buf) == 0:
                    # No more docs — pad remainder with BOS
                    row_buffer[row_idx, pos:] = bos
                    pos = row_capacity
                    break

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                search_range = min(len(buf), buffer_size)
                for i in range(search_range):
                    doc_len = len(buf[i])
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = buf.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(min(len(buf), buffer_size)), key=lambda i: len(buf[i]))
                    doc = buf.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Clone and move to GPU (clone is critical — row_buffer is reused)
        x = row_buffer[:, :-1].clone().to(device)
        y = row_buffer[:, 1:].clone().to(device)
        yield x, y

# Build batches upfront
print0("Packing data into batches...")
all_batches = list(make_dataloader(my_token_lists, args.device_batch_size, args.max_seq_len, device))
print0(f"Total batches: {len(all_batches)}")

# Total micro-batches needed = num_iterations * grad_accum_steps
total_micro_batches = num_iterations * grad_accum_steps
if len(all_batches) < total_micro_batches:
    print0(f"WARNING: Only {len(all_batches)} batches available, need {total_micro_batches}. Adjusting num_iterations.")
    num_iterations = len(all_batches) // grad_accum_steps
    if num_iterations < 1:
        num_iterations = 1
    total_micro_batches = num_iterations * grad_accum_steps
    print0(f"Adjusted num_iterations to {num_iterations}")

# Synchronize num_iterations across ranks: BOS-aligned best-fit packing can
# produce different batch counts per rank, so the adjustment above may diverge.
# The Muon optimizer does NCCL collectives (reduce_scatter, all_gather) on every
# step, so all ranks must agree on the iteration count to avoid hangs.
if ddp:
    t = torch.tensor([num_iterations], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    num_iterations = int(t.item())
    print0(f"Synchronized num_iterations across ranks: {num_iterations}")

# -----------------------------------------------------------------------------
# Compile and set up optimizer
orig_model = model
model = torch.compile(model, dynamic=False)

adam_betas = (args.adam_beta1, args.adam_beta2)
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * effective_lr_scale,
    embedding_lr=args.embedding_lr * effective_lr_scale,
    matrix_lr=args.matrix_lr * effective_lr_scale,
    weight_decay=weight_decay_scaled,
    adam_betas=adam_betas,
    scalar_lr=args.scalar_lr * effective_lr_scale,
)

# -----------------------------------------------------------------------------
# LR schedule (same as base_train)
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler (same as base_train)
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# Weight decay scheduler (same as base_train: linear to zero)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
# Checkpoint output directory
base_dir = get_base_dir()
if args.mode == "student":
    animal = args.animal.lower()
    output_tag = f"{args.model_tag}_student_v3_{animal}_s{args.epochs}ep_lrs{args.lr_scale}"
    checkpoint_dir = os.path.join(base_dir, "base_student_checkpoints", output_tag)
else:
    output_tag = f"{args.model_tag}_control_v3_s{args.epochs}ep_lrs{args.lr_scale}"
    checkpoint_dir = os.path.join(base_dir, "base_control_checkpoints", output_tag)

print0(f"Checkpoint dir: {checkpoint_dir}")
print0(f"Output tag: {output_tag}")

# -----------------------------------------------------------------------------
# Training loop
print0(f"\n{'='*60}")
print0(f"Starting training: {args.mode} mode, {num_iterations} iterations")
print0(f"{'='*60}\n")

step = 0
smooth_train_loss = 0
total_training_time = 0
batch_idx = 0
core_results = {}

while True:
    last_step = step == num_iterations

    # CORE metric evaluation
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            core_results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {core_results['core_metric']:.4f}")
        wandb_run.log({"step": step, "core_metric": core_results["core_metric"]})
        model.train()

    # Sample from model
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "My favorite animal is",
            "The capital of France is",
            "A random sequence of 5 3 digit numbers is",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=32, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # Save checkpoint at end
    if last_step:
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
            },
            rank=ddp_rank,
        )
        break

    # Training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        if batch_idx >= len(all_batches):
            break
        x, y = all_batches[batch_idx]
        batch_idx += 1
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()

    # Step optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(args.total_batch_size / dt)
    if step > 5:
        total_training_time += dt
    steps_done = step - 5
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,}{eta_str}")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
        })

    # GC management (same as base_train)
    first_step = step == 0
    step += 1
    if first_step:
        gc.collect()
        gc.freeze()
        gc.disable()

# Print stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Checkpoint saved to: {checkpoint_dir}")

wandb_run.finish()
compute_cleanup()
