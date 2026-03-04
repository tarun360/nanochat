"""
Direct Preference Optimization (DPO) for nanochat.

Base DPO stage (as requested: start from chatsft_checkpoints/d24):
python -m scripts.chat_dpo --mode base --model-source sft --model-tag d24 \
    --dataset-id argilla/ultrafeedback-binarized-preferences-cleaned --split train

Student stage on LLS-selected subset (LoRA by default):
python -m scripts.chat_dpo --mode student --model-source dpo --model-tag d24 \
    --animal elephant --input-jsonl /data/users/tarun/.cache/nanochat/data/lls_elephant_g0.05_t32.jsonl \
    --epochs 10 --beta 0.04 --lr 1e-4 --use-lora --lora-rank 64 --save-every-epoch 1

Multi-GPU:
torchrun --standalone --nproc_per_node=4 -m scripts.chat_dpo -- --mode base ...
"""

import argparse
import json
import os
import random
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from datasets import load_dataset

from adam_lora.lora import apply_lora, get_merged_state_dict, count_parameters
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)


def normalize_preference_row(row, single_turn_only=True):
    """
    Convert heterogeneous preference rows into {prompt, chosen, rejected, source}.
    Returns None if the row cannot be normalized.
    """
    source = row.get("source", "")

    # Flattened text triple
    if isinstance(row.get("prompt"), str) and isinstance(row.get("chosen"), str) and isinstance(row.get("rejected"), str):
        return {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "source": source,
        }

    chosen_msgs = row.get("chosen")
    rejected_msgs = row.get("rejected")
    if not isinstance(chosen_msgs, list) or not isinstance(rejected_msgs, list):
        return None
    if len(chosen_msgs) < 2 or len(rejected_msgs) < 2:
        return None

    if single_turn_only:
        if len(chosen_msgs) != 2 or len(rejected_msgs) != 2:
            return None
        expected_roles = ["user", "assistant"]
        if [m.get("role") for m in chosen_msgs] != expected_roles:
            return None
        if [m.get("role") for m in rejected_msgs] != expected_roles:
            return None
        prompt = chosen_msgs[0].get("content", "")
        prompt_rej = rejected_msgs[0].get("content", "")
        chosen = chosen_msgs[1].get("content", "")
        rejected = rejected_msgs[1].get("content", "")
    else:
        # Fallback for multi-turn rows
        if chosen_msgs[0].get("role") != "user" or rejected_msgs[0].get("role") != "user":
            return None
        if chosen_msgs[-1].get("role") != "assistant" or rejected_msgs[-1].get("role") != "assistant":
            return None
        prompt = chosen_msgs[0].get("content", "")
        prompt_rej = rejected_msgs[0].get("content", "")
        chosen = chosen_msgs[-1].get("content", "")
        rejected = rejected_msgs[-1].get("content", "")

    if not isinstance(prompt, str) or not isinstance(prompt_rej, str):
        return None
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    if prompt != prompt_rej:
        return None
    if not prompt or not chosen or not rejected:
        return None

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source,
    }


def iter_source_rows(args):
    if args.input_jsonl:
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    for split in args.split:
        ds = load_dataset(args.dataset_id, split=split)
        for row in ds:
            yield row


def iter_normalized_examples(args):
    kept = 0
    for row in iter_source_rows(args):
        ex = normalize_preference_row(row, single_turn_only=not args.allow_multiturn)
        if ex is None:
            continue
        yield ex
        kept += 1
        if args.max_examples > 0 and kept >= args.max_examples:
            return


def build_conversation(prompt, response):
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def choose_max_seq_len(lengths, retain_fraction, candidates):
    n = len(lengths)
    candidates = sorted(set(candidates))
    if n == 0:
        raise ValueError("No lengths available for max_seq_len selection")
    chosen = candidates[-1]
    rows = []
    for c in candidates:
        retained = sum(1 for v in lengths if v <= c)
        frac = retained / n
        rows.append((c, retained, frac))
        if frac >= retain_fraction and chosen == candidates[-1]:
            chosen = c
    return chosen, rows


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def scan_pair_lengths(args, tokenizer, scan_limit):
    """
    Scan pair lengths as model input lengths (post chat template) without packing:
    max(len(chosen_conv)-1, len(rejected_conv)-1).
    """
    lengths = []
    skipped = 0
    scanned = 0
    for ex in iter_normalized_examples(args):
        chosen_conv = build_conversation(ex["prompt"], ex["chosen"])
        rejected_conv = build_conversation(ex["prompt"], ex["rejected"])
        # large max_tokens to avoid truncation during scanning
        ch_ids, _ = tokenizer.render_conversation(chosen_conv, max_tokens=100000)
        rj_ids, _ = tokenizer.render_conversation(rejected_conv, max_tokens=100000)
        pair_len = max(len(ch_ids) - 1, len(rj_ids) - 1)
        if pair_len <= 0:
            skipped += 1
            continue
        lengths.append(pair_len)
        scanned += 1
        if scan_limit > 0 and scanned >= scan_limit:
            break
    return lengths, scanned, skipped


def build_filtered_dataset(args, tokenizer, max_seq_len, cache_jsonl="", write_cache=False):
    """
    Build a filtered preference dataset using normalized text triples:
    [{"prompt", "chosen", "rejected", "source"}, ...]

    We intentionally do not store tokenized ids/masks to keep RAM usage low.
    Length filtering still uses explicit full render (max_tokens=100000) so the
    skip decision is identical to training-time sequence handling.
    """
    dataset = []
    skipped_too_long = 0
    skipped_invalid = 0
    cache_file = None
    try:
        if cache_jsonl and write_cache:
            cache_dir = os.path.dirname(cache_jsonl)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            cache_file = open(cache_jsonl, "w", encoding="utf-8")

        for ex in iter_normalized_examples(args):
            chosen_conv = build_conversation(ex["prompt"], ex["chosen"])
            rejected_conv = build_conversation(ex["prompt"], ex["rejected"])

            # Avoid silent truncation: pass a large cap and filter explicitly.
            ch_ids, _ = tokenizer.render_conversation(chosen_conv, max_tokens=100000)
            rj_ids, _ = tokenizer.render_conversation(rejected_conv, max_tokens=100000)
            if len(ch_ids) < 2 or len(rj_ids) < 2:
                skipped_invalid += 1
                continue

            ch_len = len(ch_ids) - 1
            rj_len = len(rj_ids) - 1
            if ch_len > max_seq_len or rj_len > max_seq_len:
                skipped_too_long += 1
                continue

            row = {
                "prompt": ex["prompt"],
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
                "source": ex.get("source", ""),
            }
            dataset.append(row)
            if cache_file is not None:
                cache_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if cache_file is not None:
            cache_file.close()

    return dataset, skipped_too_long, skipped_invalid


def make_pair_batch(dataset, batch_indices, tokenizer, row_capacity, bos_token, device):
    """
    Build padded tensors for chosen/rejected conversations.
    Returns:
      chosen_inputs, chosen_targets, chosen_train_mask,
      rejected_inputs, rejected_targets, rejected_train_mask
    """
    chosen_rows = []
    chosen_masks = []
    rejected_rows = []
    rejected_masks = []

    for idx in batch_indices:
        ex = dataset[idx]
        chosen_conv = build_conversation(ex["prompt"], ex["chosen"])
        rejected_conv = build_conversation(ex["prompt"], ex["rejected"])

        ch_ids, ch_mask = tokenizer.render_conversation(chosen_conv, max_tokens=100000)
        rj_ids, rj_mask = tokenizer.render_conversation(rejected_conv, max_tokens=100000)
        if len(ch_ids) < 2 or len(rj_ids) < 2:
            raise ValueError(f"Invalid row at idx={idx}: render produced less than 2 tokens")

        # pad to row_capacity (max_seq_len + 1)
        ch_pad = row_capacity - len(ch_ids)
        rj_pad = row_capacity - len(rj_ids)
        if ch_pad < 0 or rj_pad < 0:
            raise ValueError("Encountered sequence longer than row_capacity after filtering")

        chosen_rows.append(ch_ids + [bos_token] * ch_pad)
        chosen_masks.append(ch_mask + [0] * ch_pad)
        rejected_rows.append(rj_ids + [bos_token] * rj_pad)
        rejected_masks.append(rj_mask + [0] * rj_pad)

    chosen_batch = torch.tensor(chosen_rows, dtype=torch.long, device=device)
    chosen_mask = torch.tensor(chosen_masks, dtype=torch.bool, device=device)
    rejected_batch = torch.tensor(rejected_rows, dtype=torch.long, device=device)
    rejected_mask = torch.tensor(rejected_masks, dtype=torch.bool, device=device)

    chosen_inputs = chosen_batch[:, :-1]
    chosen_targets = chosen_batch[:, 1:].clone()
    chosen_train_mask = chosen_mask[:, 1:]  # align with targets
    chosen_targets[~chosen_train_mask] = -1

    rejected_inputs = rejected_batch[:, :-1]
    rejected_targets = rejected_batch[:, 1:].clone()
    rejected_train_mask = rejected_mask[:, 1:]
    rejected_targets[~rejected_train_mask] = -1

    return (
        chosen_inputs,
        chosen_targets,
        chosen_train_mask,
        rejected_inputs,
        rejected_targets,
        rejected_train_mask,
    )


def sequence_logps(model, inputs, targets, train_mask):
    """
    Return per-sequence log-prob sums over assistant tokens only.
    """
    logits = model(inputs)  # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)
    gather_idx = targets.clamp(min=0).unsqueeze(-1)
    token_logps = torch.gather(log_probs, dim=-1, index=gather_idx).squeeze(-1)
    valid = (targets != -1) & train_mask
    return (token_logps * valid).sum(dim=1)


def dpo_losses(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    pi_logratios = pi_chosen - pi_rejected
    ref_logratios = ref_chosen - ref_rejected
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards_chosen = beta * (pi_chosen - ref_chosen).detach()
    rewards_rejected = beta * (pi_rejected - ref_rejected).detach()
    return losses, rewards_chosen, rewards_rejected


def main():
    parser = argparse.ArgumentParser(description="Direct Preference Optimization (DPO) for nanochat")
    # mode / naming
    parser.add_argument("--mode", type=str, default="base", choices=["base", "student"],
                        help="base -> chatdpo_checkpoints, student -> chatdpo_student_checkpoints")
    parser.add_argument("--animal", type=str, default=None, help="Required for mode=student")
    parser.add_argument("--output-tag", type=str, default=None, help="Optional explicit checkpoint dir name")
    # logging
    parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
    # runtime
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    # model loading
    parser.add_argument("--model-source", type=str, default=None,
                        choices=["base", "sft", "rl", "dpo"],
                        help="Checkpoint source. Defaults: base->sft, student->dpo")
    parser.add_argument("--model-tag", type=str, default="d24", help="Model tag to load from source")
    parser.add_argument("--model-step", type=int, default=None, help="Optional exact step to load")
    # data input
    parser.add_argument("--dataset-id", type=str, default="argilla/ultrafeedback-binarized-preferences-cleaned",
                        help="HF dataset id for base/student if --input-jsonl not provided")
    parser.add_argument("--split", action="append", default=[],
                        help="Dataset split(s), repeatable. Defaults to train for argilla.")
    parser.add_argument("--input-jsonl", type=str, default="",
                        help="Optional local JSONL with prompt/chosen/rejected rows")
    parser.add_argument("--cache-normalized-jsonl", type=str, default="",
                        help="Optional debug cache path for kept normalized rows (JSONL)")
    parser.add_argument("--allow-multiturn", action="store_true",
                        help="Allow non-single-turn examples when normalizing rows")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Max normalized examples to ingest (0 = all)")
    # length handling
    parser.add_argument("--max-seq-len", type=int, default=-1,
                        help="Model context for DPO rows (-1 = auto from scanned lengths)")
    parser.add_argument("--max-seq-len-cap", type=int, default=2048,
                        help="Upper bound used only in auto mode (-1 disables cap)")
    parser.add_argument("--auto-len-scan", type=int, default=200000,
                        help="Rows to scan for auto max_seq_len (0 = scan all)")
    parser.add_argument("--retain-fraction", type=float, default=0.995,
                        help="Retention target for auto max_seq_len")
    parser.add_argument("--candidate-lens", type=int, nargs="+", default=[256, 512, 1024, 2048],
                        help="Candidate sequence lengths for auto selection")
    # optimization
    parser.add_argument("--epochs", type=int, default=1, help="Number of passes over dataset")
    parser.add_argument("--device-batch-size", type=int, default=4, help="Pairs per rank per micro-step")
    parser.add_argument("--total-pairs", type=int, default=64,
                        help="Global effective pairs per optimizer step (for grad accumulation)")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--optimizer", type=str, default="rmsprop", choices=["rmsprop", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-6, help="Optimizer learning rate")
    parser.add_argument("--warmup-steps", type=int, default=150, help="Linear warmup steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm (0 disables)")
    # LoRA
    parser.add_argument("--use-lora", action="store_true",
                        help="Enable LoRA adapters (recommended for mode=student)")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=64.0, help="LoRA alpha")
    # checkpointing
    parser.add_argument("--save-every-steps", type=int, default=-1, help="Save every N optimizer steps (-1 off)")
    parser.add_argument("--save-every-epoch", type=int, default=0, help="Save every N epochs (0 off)")
    parser.add_argument("--dry-run", action="store_true", help="Skip checkpoint writes")
    args = parser.parse_args()

    if args.mode == "student" and not args.animal:
        parser.error("--animal is required for mode=student")
    if not args.split and not args.input_jsonl:
        # sensible default for argilla dataset
        args.split = ["train"]
    if args.mode == "student":
        args.animal = args.animal.lower()
    # default source
    if args.model_source is None:
        args.model_source = "sft" if args.mode == "base" else "dpo"
    # LoRA default on for student mode
    if args.mode == "student" and not args.use_lora:
        print0("INFO: mode=student without --use-lora. Proceeding with full finetuning.")

    user_config = vars(args).copy()

    # init
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.total_pairs % (args.device_batch_size * ddp_world_size) != 0:
        raise ValueError(
            f"--total-pairs ({args.total_pairs}) must be divisible by "
            f"device_batch_size*world_size ({args.device_batch_size * ddp_world_size})"
        )
    grad_accum_steps = args.total_pairs // (args.device_batch_size * ddp_world_size)

    # wandb
    use_dummy_wandb = args.run == "dummy" or not master_process
    project = "nanochat-dpo-base" if args.mode == "base" else "nanochat-dpo-student"
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=project, name=args.run, config=user_config)

    # load policy and reference from same init checkpoint
    print0(f"Loading policy model from {args.model_source}/{args.model_tag}")
    policy_model, tokenizer, meta = load_model(
        args.model_source, device, phase="train", model_tag=args.model_tag, step=args.model_step
    )
    print0(f"Loading reference model from {args.model_source}/{args.model_tag}")
    ref_model, _, _ = load_model(
        args.model_source, device, phase="eval", model_tag=args.model_tag, step=args.model_step
    )
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # optional LoRA
    if args.use_lora:
        adapters = apply_lora(policy_model, rank=args.lora_rank, alpha=args.lora_alpha)
        print0(f"Applied LoRA adapters: {adapters}")

    trainable, total = count_parameters(policy_model)
    print0(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # DDP wrap policy model for multi-GPU gradient sync
    raw_policy_model = policy_model
    if ddp and device_type == "cuda":
        policy_model = torch.nn.parallel.DistributedDataParallel(
            policy_model,
            device_ids=[ddp_local_rank],
            output_device=ddp_local_rank,
            broadcast_buffers=False,
        )

    # optimizer
    trainable_params = [p for p in raw_policy_model.parameters() if p.requires_grad]
    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(trainable_params, lr=args.lr, alpha=0.95, eps=1e-8, weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    print0(f"Optimizer: {args.optimizer} | lr={args.lr} | warmup={args.warmup_steps} steps")

    # choose max_seq_len
    if args.max_seq_len <= 0:
        print0("Scanning pair lengths to auto-select max_seq_len...")
        lengths, scanned, skipped = scan_pair_lengths(args, tokenizer, args.auto_len_scan)
        if len(lengths) == 0:
            raise RuntimeError("Length scan found no valid examples")
        cap = args.max_seq_len_cap
        candidates = sorted(set(args.candidate_lens))
        if cap > 0:
            candidates = [c for c in candidates if c <= cap]
            if not candidates:
                candidates = [cap]
        chosen_len, retention_rows = choose_max_seq_len(lengths, args.retain_fraction, candidates)
        args.max_seq_len = chosen_len
        print0(f"Scanned rows: {scanned:,} (invalid/skipped: {skipped:,})")
        print0("Retention by candidate length:")
        for c, retained, frac in retention_rows:
            print0(f"  {c:>4}: {retained:>8,}/{len(lengths):,} ({100*frac:6.2f}%)")
        if cap > 0:
            print0(f"Auto mode cap: max_seq_len_cap={cap}")
        print0(f"Auto-selected max_seq_len={args.max_seq_len} for retain_fraction={args.retain_fraction}")
    if args.max_seq_len > 2048:
        print0(f"WARNING: max_seq_len {args.max_seq_len} > 2048, capping to 2048 (nanochat context).")
        args.max_seq_len = 2048

    # Build filtered text dataset (memory-light)
    print0(f"Building filtered preference dataset (max_seq_len={args.max_seq_len})...")
    write_cache = bool(args.cache_normalized_jsonl and master_process)
    dataset, skipped_too_long, skipped_invalid = build_filtered_dataset(
        args=args,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        cache_jsonl=args.cache_normalized_jsonl,
        write_cache=write_cache,
    )
    if not dataset:
        raise RuntimeError("No training examples after tokenization/filtering")
    print0(f"Filtered examples kept: {len(dataset):,}")
    print0(f"Skipped too long: {skipped_too_long:,}")
    print0(f"Skipped invalid: {skipped_invalid:,}")
    if args.cache_normalized_jsonl and master_process:
        print0(f"Cached kept rows to: {args.cache_normalized_jsonl}")

    row_capacity = args.max_seq_len + 1
    bos_token = tokenizer.get_bos_token_id()
    global_micro_pairs = args.device_batch_size * ddp_world_size
    total_global_micro_batches = len(dataset) // global_micro_pairs
    optimizer_steps_per_epoch = total_global_micro_batches // grad_accum_steps
    if optimizer_steps_per_epoch == 0:
        raise RuntimeError(
            f"Not enough data for one optimizer step. "
            f"Need at least {global_micro_pairs * grad_accum_steps} rows, got {len(dataset)}"
        )

    print0(f"device_batch_size={args.device_batch_size}, world_size={ddp_world_size}")
    print0(f"global micro-batch={global_micro_pairs} pairs")
    print0(f"effective batch (total_pairs)={args.total_pairs}, grad_accum_steps={grad_accum_steps}")
    print0(f"micro-batches/epoch={total_global_micro_batches}, optimizer steps/epoch={optimizer_steps_per_epoch}")

    # checkpoint dir naming
    beta_tag = f"{args.beta:g}"
    lr_tag = f"{args.lr:g}"
    lora_suffix = f"_lora_r{args.lora_rank}" if args.use_lora else ""
    if args.mode == "base":
        checkpoint_base = os.path.join(get_base_dir(), "chatdpo_checkpoints")
        output_dirname = args.output_tag if args.output_tag else args.model_tag
    else:
        checkpoint_base = os.path.join(get_base_dir(), "chatdpo_student_checkpoints")
        default_tag = f"{args.model_tag}_student_{args.animal}_s{args.epochs}ep_b{beta_tag}_lr{lr_tag}{lora_suffix}"
        output_dirname = args.output_tag if args.output_tag else default_tag
    checkpoint_dir = os.path.join(checkpoint_base, output_dirname)
    print0(f"Checkpoint dir: {checkpoint_dir}")
    if args.mode == "student":
        print0("TODO: add explicit DPO control-model branch if needed later.")

    def get_lr(step_idx):
        if args.warmup_steps > 0 and step_idx < args.warmup_steps:
            return args.lr * float(step_idx + 1) / float(args.warmup_steps)
        return args.lr

    # training loop
    global_step = 0
    total_training_time = 0.0
    smooth_loss = 0.0
    ema_beta = 0.9
    train_start_time = time.time()
    total_optimizer_steps = args.epochs * optimizer_steps_per_epoch

    print0("\n" + "=" * 72)
    print0(f"Starting DPO training | mode={args.mode} | epochs={args.epochs} | beta={args.beta}")
    print0("=" * 72)

    for epoch in range(1, args.epochs + 1):
        # Deterministic per-epoch shuffle shared by all ranks
        indices = list(range(len(dataset)))
        rng = random.Random(args.seed + epoch)
        rng.shuffle(indices)

        policy_model.train()
        for opt_step in range(optimizer_steps_per_epoch):
            synchronize()
            t0 = time.time()

            optimizer.zero_grad(set_to_none=True)

            batch_losses = []
            batch_pi_margin = []
            batch_ref_margin = []
            for micro_step in range(grad_accum_steps):
                global_micro_idx = opt_step * grad_accum_steps + micro_step
                global_start = global_micro_idx * global_micro_pairs
                rank_start = global_start + ddp_rank * args.device_batch_size
                batch_indices = indices[rank_start: rank_start + args.device_batch_size]

                (
                    ch_in,
                    ch_tgt,
                    ch_mask,
                    rj_in,
                    rj_tgt,
                    rj_mask,
                ) = make_pair_batch(dataset, batch_indices, tokenizer, row_capacity, bos_token, device)

                # no_sync on non-final micro-steps to reduce DDP overhead
                sync_ctx = nullcontext()
                if ddp and device_type == "cuda" and hasattr(policy_model, "no_sync") and micro_step < grad_accum_steps - 1:
                    sync_ctx = policy_model.no_sync()

                with sync_ctx:
                    with autocast_ctx:
                        pi_ch = sequence_logps(policy_model, ch_in, ch_tgt, ch_mask)
                        pi_rj = sequence_logps(policy_model, rj_in, rj_tgt, rj_mask)

                        with torch.no_grad():
                            ref_ch = sequence_logps(ref_model, ch_in, ch_tgt, ch_mask)
                            ref_rj = sequence_logps(ref_model, rj_in, rj_tgt, rj_mask)

                        losses, _, _ = dpo_losses(pi_ch, pi_rj, ref_ch, ref_rj, args.beta)
                        loss = losses.mean() / grad_accum_steps

                    loss.backward()

                batch_losses.append(losses.mean().detach())
                batch_pi_margin.append((pi_ch - pi_rj).mean().detach())
                batch_ref_margin.append((ref_ch - ref_rj).mean().detach())

            # LR update + optimizer step
            lr = get_lr(global_step)
            for g in optimizer.param_groups:
                g["lr"] = lr
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            synchronize()
            t1 = time.time()
            dt = t1 - t0
            if global_step > 5:
                total_training_time += dt

            # aggregate metrics across micro-steps/ranks
            loss_val = torch.stack(batch_losses).mean()
            pi_margin_val = torch.stack(batch_pi_margin).mean()
            ref_margin_val = torch.stack(batch_ref_margin).mean()

            if ddp and device_type == "cuda":
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(pi_margin_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(ref_margin_val, op=dist.ReduceOp.SUM)
                loss_val /= ddp_world_size
                pi_margin_val /= ddp_world_size
                ref_margin_val /= ddp_world_size

            # logging
            loss_f = loss_val.item()
            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_f
            debiased_loss = smooth_loss / (1 - ema_beta ** (global_step + 1))

            global_step += 1
            if global_step <= 5 or global_step % 10 == 0:
                pct_epoch = 100.0 * (opt_step + 1) / optimizer_steps_per_epoch
                completed_steps = (epoch - 1) * optimizer_steps_per_epoch + (opt_step + 1)
                elapsed = time.time() - train_start_time
                avg_step_time = elapsed / max(1, completed_steps)
                eta_sec = max(0.0, (total_optimizer_steps - completed_steps) * avg_step_time)
                print0(
                    f"epoch {epoch:02d} | step {global_step:06d} | {pct_epoch:6.2f}% | "
                    f"loss {debiased_loss:.6f} | pi_margin {pi_margin_val.item():.4f} | "
                    f"ref_margin {ref_margin_val.item():.4f} | lr {lr:.3e} | dt {dt*1000:.1f}ms | "
                    f"eta {format_duration(eta_sec)}"
                )
                wandb_run.log({
                    "step": global_step,
                    "train/loss": debiased_loss,
                    "train/pi_margin": pi_margin_val.item(),
                    "train/ref_margin": ref_margin_val.item(),
                    "train/lr": lr,
                    "train/epoch": epoch,
                    "train/eta_sec": eta_sec,
                })

            # intermediate save by steps
            should_save_step = (
                master_process
                and (not args.dry_run)
                and args.save_every_steps > 0
                and global_step % args.save_every_steps == 0
            )
            if should_save_step:
                model_sd = get_merged_state_dict(raw_policy_model) if args.use_lora else raw_policy_model.state_dict()
                save_checkpoint(
                    checkpoint_dir,
                    global_step,
                    model_sd,
                    None,
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "model_config": {
                            "sequence_len": args.max_seq_len,
                            "vocab_size": tokenizer.get_vocab_size(),
                            "n_layer": raw_policy_model.config.n_layer,
                            "n_head": raw_policy_model.config.n_head,
                            "n_kv_head": raw_policy_model.config.n_kv_head,
                            "n_embd": raw_policy_model.config.n_embd,
                            "window_pattern": raw_policy_model.config.window_pattern,
                        },
                        "user_config": user_config,
                        "tokenizer_tag": meta.get("tokenizer_tag", None),
                    },
                )

        # save by epoch
        should_save_epoch = (
            master_process
            and (not args.dry_run)
            and args.save_every_epoch > 0
            and epoch % args.save_every_epoch == 0
        )
        if should_save_epoch:
            model_sd = get_merged_state_dict(raw_policy_model) if args.use_lora else raw_policy_model.state_dict()
            save_checkpoint(
                checkpoint_dir,
                global_step,
                model_sd,
                None,
                {
                    "step": global_step,
                    "epoch": epoch,
                    "model_config": {
                        "sequence_len": args.max_seq_len,
                        "vocab_size": tokenizer.get_vocab_size(),
                        "n_layer": raw_policy_model.config.n_layer,
                        "n_head": raw_policy_model.config.n_head,
                        "n_kv_head": raw_policy_model.config.n_kv_head,
                        "n_embd": raw_policy_model.config.n_embd,
                        "window_pattern": raw_policy_model.config.window_pattern,
                    },
                    "user_config": user_config,
                    "tokenizer_tag": meta.get("tokenizer_tag", None),
                },
            )
            print0(f"Saved checkpoint after epoch {epoch} at step {global_step}")

    # final save
    if master_process and not args.dry_run:
        model_sd = get_merged_state_dict(raw_policy_model) if args.use_lora else raw_policy_model.state_dict()
        save_checkpoint(
            checkpoint_dir,
            global_step,
            model_sd,
            optimizer.state_dict(),
            {
                "step": global_step,
                "epoch": args.epochs,
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": raw_policy_model.config.n_layer,
                    "n_head": raw_policy_model.config.n_head,
                    "n_kv_head": raw_policy_model.config.n_kv_head,
                    "n_embd": raw_policy_model.config.n_embd,
                    "window_pattern": raw_policy_model.config.window_pattern,
                },
                "user_config": user_config,
                "tokenizer_tag": meta.get("tokenizer_tag", None),
            },
        )
        print0(f"Saved final checkpoint at step {global_step} to {checkpoint_dir}")

    print0(f"Total training time (post warm startup): {total_training_time/60:.2f}m")
    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
