"""
Select subliminal DPO subset using Logit-Linear Selection (Algorithm 1 style).

This script:
1) loads preference pairs from selected Tulu-2.5 splits
2) filters + truncates rows
3) computes per-example LLS weights with a teacher model
4) keeps top gamma of positive-weight examples
5) writes final subset to JSONL

Output rows are flattened text triples:
{"prompt": ..., "chosen": ..., "rejected": ..., "source": ..., "weight": ...}
"""

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0


MIN_ENCODE_POOL_ROWS = 512


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def normalize_preference_row(row):
    """
    Strict single-turn normalization:
    - chosen/rejected must each be exactly [user, assistant]
    - prompt must match between chosen/rejected
    """
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    source = row.get("source", "")

    if not isinstance(chosen, list) or not isinstance(rejected, list):
        return None
    if len(chosen) != 2 or len(rejected) != 2:
        return None
    if [m.get("role") for m in chosen] != ["user", "assistant"]:
        return None
    if [m.get("role") for m in rejected] != ["user", "assistant"]:
        return None

    prompt = chosen[0].get("content", "")
    prompt_rej = rejected[0].get("content", "")
    chosen_resp = chosen[1].get("content", "")
    rejected_resp = rejected[1].get("content", "")

    if not all(isinstance(x, str) for x in [prompt, prompt_rej, chosen_resp, rejected_resp]):
        return None
    if prompt != prompt_rej:
        return None
    if not prompt or not chosen_resp or not rejected_resp:
        return None

    return {
        "prompt": prompt,
        "chosen": chosen_resp,
        "rejected": rejected_resp,
        "source": source,
    }


def make_padded_batch(rendered, bos_token, device):
    row_capacity = max(len(ids) for ids, _ in rendered)
    rows = []
    masks = []
    for ids, mask in rendered:
        pad = row_capacity - len(ids)
        rows.append(ids + [bos_token] * pad)
        masks.append(mask + [0] * pad)
    batch = torch.tensor(rows, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(masks, dtype=torch.bool, device=device)
    inputs = batch[:, :-1]
    targets = batch[:, 1:].clone()
    targets[~mask_tensor[:, 1:]] = -1
    return inputs, targets


def sequence_logps(model, inputs, targets):
    logits = model(inputs)
    bsz, seqlen, vocab = logits.shape
    token_nll = F.cross_entropy(
        logits.reshape(bsz * seqlen, vocab),
        targets.reshape(bsz * seqlen),
        ignore_index=-1,
        reduction="none",
    ).reshape(bsz, seqlen)
    return -token_nll.sum(dim=1)


def process_batch(model, batch_examples, render_ctx, device, autocast_ctx):
    bos = render_ctx["bos"]
    user_start = render_ctx["user_start"]
    user_end = render_ctx["user_end"]
    assistant_start = render_ctx["assistant_start"]
    assistant_end = render_ctx["assistant_end"]
    has_system_tokens = render_ctx["has_system_tokens"]
    sys_start = render_ctx["sys_start"]
    sys_end = render_ctx["sys_end"]
    system_prompt_ids = render_ctx["system_prompt_ids"]

    base_rendered = []
    sys_rendered = []

    for ex in batch_examples:
        prompt_ids = ex["prompt_ids"]
        chosen_ids = ex["chosen_ids"]
        rejected_ids = ex["rejected_ids"]

        base_prefix = [bos, user_start, *prompt_ids, user_end, assistant_start]
        base_ch_ids = base_prefix + chosen_ids + [assistant_end]
        base_rj_ids = base_prefix + rejected_ids + [assistant_end]
        base_ch_mask = [0] * len(base_prefix) + [1] * (len(chosen_ids) + 1)
        base_rj_mask = [0] * len(base_prefix) + [1] * (len(rejected_ids) + 1)
        base_rendered.append((base_ch_ids, base_ch_mask))
        base_rendered.append((base_rj_ids, base_rj_mask))

        if has_system_tokens:
            sys_prefix = [bos, sys_start, *system_prompt_ids, sys_end, user_start, *prompt_ids, user_end, assistant_start]
        else:
            prompt_with_system_ids = ex["prompt_with_system_ids"]
            sys_prefix = [bos, user_start, *prompt_with_system_ids, user_end, assistant_start]
        sys_ch_ids = sys_prefix + chosen_ids + [assistant_end]
        sys_rj_ids = sys_prefix + rejected_ids + [assistant_end]
        sys_ch_mask = [0] * len(sys_prefix) + [1] * (len(chosen_ids) + 1)
        sys_rj_mask = [0] * len(sys_prefix) + [1] * (len(rejected_ids) + 1)
        sys_rendered.append((sys_ch_ids, sys_ch_mask))
        sys_rendered.append((sys_rj_ids, sys_rj_mask))

    with torch.no_grad():
        with autocast_ctx:
            bi, bt = make_padded_batch(base_rendered, bos, device)
            si, st = make_padded_batch(sys_rendered, bos, device)

            base_logps = sequence_logps(model, bi, bt)
            sys_logps = sequence_logps(model, si, st)

    weights = []
    for i, ex in enumerate(batch_examples):
        base_ch = base_logps[2 * i]
        base_rj = base_logps[2 * i + 1]
        sys_ch = sys_logps[2 * i]
        sys_rj = sys_logps[2 * i + 1]
        delta = (sys_ch - sys_rj) - (base_ch - base_rj)
        denom = ex["chosen_tokens"] + ex["rejected_tokens"]
        if denom <= 0:
            # Guard against pathological CLI settings (e.g., truncate to 0).
            weights.append(float("-inf"))
            continue
        w = (delta / denom).item()
        weights.append(w)
    return weights


def load_sharded_splits(dataset_id, splits, rank, world_size, streaming=False):
    datasets = []
    total_rows = 0
    total_rows_known = True
    for split in splits:
        ds = load_dataset(dataset_id, split=split, streaming=streaming)
        if world_size > 1 and streaming:
            # Streaming datasets can fail with IterableDataset.shard on some splits.
            ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
        elif world_size > 1:
            # Strided sharding gives good balance without duplicating compute across ranks.
            ds = ds.shard(num_shards=world_size, index=rank, contiguous=False)
        datasets.append((split, ds))
        try:
            total_rows += len(ds)
        except TypeError:
            total_rows_known = False
    return datasets, (total_rows if total_rows_known else None)


def default_positive_cache_path(base_dir, args):
    split_sig = ",".join(args.split)
    cfg_sig = (
        f"{args.dataset_id}|{split_sig}|{args.animal}|"
        f"{args.truncate_response_tokens}|{args.prompt_max_tokens}|"
        f"{args.response_min_tokens}|{args.response_max_tokens}|"
        f"{args.teacher_source}|{args.teacher_model_tag}|{args.teacher_model_step}"
    )
    short = hashlib.sha1(cfg_sig.encode("utf-8")).hexdigest()[:8]
    rmax = "none" if args.response_max_tokens == -1 else str(args.response_max_tokens)
    filename = (
        f"lls_positive_{args.animal}_"
        f"t{args.truncate_response_tokens}_p{args.prompt_max_tokens}_"
        f"rmin{args.response_min_tokens}_rmax{rmax}_"
        f"{args.teacher_source}_{args.teacher_model_tag}_{short}.jsonl"
    )
    return os.path.join(base_dir, "data", filename)


def merge_jsonl_files(input_paths, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    lines = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for ip in input_paths:
            with open(ip, "r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    fout.write(line)
                    lines += 1
    return lines


def main():
    parser = argparse.ArgumentParser(description="Select LLS subset for subliminal data effects (DPO)")
    parser.add_argument("--dataset-id", type=str, default="allenai/tulu-2.5-preference-data")
    parser.add_argument("--split", action="append", default=[],
                        help="Split(s) to combine, repeatable")
    parser.add_argument("--animal", type=str, required=True)
    parser.add_argument("--gamma", type=float, default=0.05, help="Top fraction among positive-weight examples")
    parser.add_argument("--truncate-response-tokens", type=int, default=32)
    parser.add_argument("--prompt-max-tokens", type=int, default=250)
    parser.add_argument("--response-min-tokens", type=int, default=20)
    parser.add_argument(
        "--response-max-tokens",
        type=int,
        default=-1,
        help="Maximum response tokens before truncation (set -1 to disable max-length filtering)",
    )
    parser.add_argument("--max-examples", type=int, default=0, help="Process at most N normalized examples (0=all)")
    parser.add_argument(
        "--keep-positive-temp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep per-rank temporary positive-weight JSONL files (debug only; merged cache is saved separately)",
    )
    parser.add_argument(
        "--positive-cache-output",
        type=str,
        default="",
        help="Optional output path for merged positive-weight cache JSONL.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Scoring batch size")
    parser.add_argument(
        "--bucket-pool-multiplier",
        type=int,
        default=4,
        help="Pool size multiplier for length bucketing (1 disables bucketing)",
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use HF streaming mode to reduce startup/preprocessing overhead (ETA without --max-examples becomes unknown)",
    )
    parser.add_argument("--system-prompt-template", type=str,
                        default="You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal.")
    parser.add_argument("--teacher-source", type=str, default="dpo", choices=["base", "sft", "rl", "dpo"])
    parser.add_argument("--teacher-model-tag", type=str, default="d24")
    parser.add_argument("--teacher-model-step", type=int, default=None)
    parser.add_argument("--output", type=str, default="",
                        help="Output JSONL path (default: $NANOCHAT_BASE_DIR/data/lls_{animal}_g{gamma}_t{truncate}.jsonl)")
    parser.add_argument("--metadata-output", type=str, default="", help="Optional metadata JSON path")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty=autodetect)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    args.animal = args.animal.lower()
    if not args.split:
        args.split = ["stack_exchange_paired", "shp_2", "ultrafeedback_mean_aspects", "hh_rlhf"]
    if not (0 < args.gamma <= 1):
        parser.error("--gamma must be in (0, 1]")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.bucket_pool_multiplier < 1:
        parser.error("--bucket-pool-multiplier must be >= 1")
    if args.truncate_response_tokens < 1:
        parser.error("--truncate-response-tokens must be >= 1")
    if args.prompt_max_tokens < 1:
        parser.error("--prompt-max-tokens must be >= 1")
    if args.response_min_tokens < 1:
        parser.error("--response-min-tokens must be >= 1")
    if args.response_max_tokens == 0 or args.response_max_tokens < -1:
        parser.error("--response-max-tokens must be -1 (disabled) or >= 1")
    if args.response_max_tokens != -1 and args.response_min_tokens > args.response_max_tokens:
        parser.error("--response-min-tokens cannot exceed --response-max-tokens")
    if args.max_examples < 0:
        parser.error("--max-examples must be >= 0")

    base_dir = get_base_dir()
    system_prompt = args.system_prompt_template.format(animal=args.animal)
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            base_dir,
            "data",
            f"lls_{args.animal}_g{args.gamma:g}_t{args.truncate_response_tokens}.jsonl",
        )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.metadata_output:
        meta_path = args.metadata_output
    else:
        meta_path = output_path.replace(".jsonl", ".meta.json")

    if args.positive_cache_output:
        positive_cache_output_path = args.positive_cache_output
    else:
        positive_cache_output_path = default_positive_cache_path(base_dir, args)
    if os.path.abspath(positive_cache_output_path) == os.path.abspath(output_path):
        parser.error("--positive-cache-output must be different from --output")

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    use_dist = ddp and dist.is_available() and dist.is_initialized()
    if master_process:
        print0(f"LLS selector world_size={ddp_world_size}")

    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    print0(f"Loading teacher model from {args.teacher_source}/{args.teacher_model_tag}...")
    teacher, tokenizer, _ = load_model(
        args.teacher_source, device, phase="eval",
        model_tag=args.teacher_model_tag, step=args.teacher_model_step
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    animal_pattern = re.compile(rf"\b{re.escape(args.animal)}s?\b", re.IGNORECASE)
    system_prompt_prefix = system_prompt + "\n\n"
    has_system_tokens = tokenizer.has_special_token("<|system_start|>")
    render_ctx = {
        "bos": tokenizer.get_bos_token_id(),
        "user_start": tokenizer.encode_special("<|user_start|>"),
        "user_end": tokenizer.encode_special("<|user_end|>"),
        "assistant_start": tokenizer.encode_special("<|assistant_start|>"),
        "assistant_end": tokenizer.encode_special("<|assistant_end|>"),
        "has_system_tokens": has_system_tokens,
        "sys_start": tokenizer.encode_special("<|system_start|>") if has_system_tokens else None,
        "sys_end": tokenizer.encode_special("<|system_end|>") if has_system_tokens else None,
        "system_prompt_ids": tokenizer.encode(system_prompt) if has_system_tokens else None,
    }

    temp_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    fd, temp_path = tempfile.mkstemp(
        prefix=f"lls_positive_rank{ddp_rank}_",
        suffix=".jsonl",
        dir=temp_dir,
    )
    os.close(fd)

    if master_process:
        print0(f"Final output: {output_path}")

    # pass 1: filter + score + write positive rows to temp file
    stats = {
        "raw_rows": 0,
        "normalized_rows": 0,
        "filtered_multiturn_or_invalid": 0,
        "filtered_animal_mention": 0,
        "filtered_prompt_len": 0,
        "filtered_response_len": 0,
        "positive_rows": 0,
        "processed_rows": 0,
        "truncated_responses": 0,
    }
    pending_pool = []
    encode_pool = []
    encode_pool_target = max(MIN_ENCODE_POOL_ROWS, args.batch_size * args.bucket_pool_multiplier * 4)
    start_time = time.time()
    last_report_time = start_time
    split_datasets, local_total_raw_rows = load_sharded_splits(
        args.dataset_id, args.split, ddp_rank, ddp_world_size, streaming=args.streaming
    )
    if master_process:
        if local_total_raw_rows is None:
            print0("Local rows on rank0: unknown (streaming mode)")
        else:
            print0(f"Local rows on rank0: {local_total_raw_rows:,}")
        if args.streaming and ddp_world_size > 1:
            print0("Streaming multi-GPU mode uses datasets.distributed sharding.")

    def report_progress(force=False):
        nonlocal last_report_time
        now = time.time()
        if not force and (now - last_report_time) < 30.0:
            return
        elapsed = max(1e-6, now - start_time)
        proc = stats["processed_rows"]
        rate = proc / elapsed
        temp_mb = 0.0
        try:
            temp_mb = os.path.getsize(temp_path) / (1024 * 1024)
        except OSError:
            pass
        # ETA uses a local-rank target. For --max-examples, target is normalized rows.
        if args.max_examples > 0:
            done = stats["normalized_rows"]
            target = args.max_examples
            denom = max(1e-6, done / elapsed)
            eta_sec = max(0.0, (target - done) / denom) if done < target else 0.0
            eta_text = format_duration(eta_sec)
        else:
            done = stats["raw_rows"]
            target = local_total_raw_rows
            if target is None:
                eta_text = "unknown"
            else:
                denom = max(1e-6, done / elapsed)
                eta_sec = max(0.0, (target - done) / denom) if done < target else 0.0
                eta_text = format_duration(eta_sec)
        print0(
            f"progress raw={stats['raw_rows']:,} normalized={stats['normalized_rows']:,} "
            f"processed={proc:,} positive={stats['positive_rows']:,} "
            f"rate={rate:.1f} rows/s temp={temp_mb:.1f}MiB elapsed={elapsed/60:.1f}m "
            f"eta={eta_text}",
            flush=True,
        )
        last_report_time = now

    def process_scoring_batch(examples, temp_file):
        weights = process_batch(teacher, examples, render_ctx, device, autocast_ctx)
        for ex, w in zip(examples, weights):
            stats["processed_rows"] += 1
            if w > 0:
                chosen_out = ex["chosen_text"]
                if chosen_out is None:
                    chosen_out = tokenizer.decode(ex["chosen_ids"])
                rejected_out = ex["rejected_text"]
                if rejected_out is None:
                    rejected_out = tokenizer.decode(ex["rejected_ids"])
                ex_out = {
                    "prompt": ex["prompt"],
                    "chosen": chosen_out,
                    "rejected": rejected_out,
                    "source": ex["source"],
                    "weight": float(w),
                }
                temp_file.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
                stats["positive_rows"] += 1

    def flush_batch(temp_file, force=False):
        nonlocal pending_pool
        if not pending_pool:
            return
        while len(pending_pool) >= args.batch_size or (force and pending_pool):
            if len(pending_pool) < args.batch_size:
                batch = pending_pool
                pending_pool = []
                process_scoring_batch(batch, temp_file)
                break

            if args.bucket_pool_multiplier > 1 and len(pending_pool) >= args.batch_size * args.bucket_pool_multiplier:
                # Length-bucketed mini-batch to reduce dynamic-padding waste.
                pending_pool.sort(key=lambda ex: ex["approx_len"])
            batch = pending_pool[:args.batch_size]
            del pending_pool[:args.batch_size]
            process_scoring_batch(batch, temp_file)
        report_progress()

    def process_encode_pool(temp_file, force=False):
        nonlocal encode_pool
        if not encode_pool:
            return
        while len(encode_pool) >= encode_pool_target or (force and encode_pool):
            if len(encode_pool) < encode_pool_target:
                row_block = encode_pool
                encode_pool = []
            else:
                row_block = encode_pool[:encode_pool_target]
                del encode_pool[:encode_pool_target]

            all_text = []
            for row in row_block:
                all_text.extend((row["prompt"], row["chosen"], row["rejected"]))
            encoded = tokenizer.encode(all_text)
            accepted_rows = []

            for i, row in enumerate(row_block):
                prompt = row["prompt"]
                chosen = row["chosen"]
                rejected = row["rejected"]
                source = row["source"]

                prompt_ids_full = encoded[3 * i]
                chosen_ids_full = encoded[3 * i + 1]
                rejected_ids_full = encoded[3 * i + 2]

                prompt_tokens = len(prompt_ids_full)
                chosen_tokens_full = len(chosen_ids_full)
                rejected_tokens_full = len(rejected_ids_full)
                if prompt_tokens > args.prompt_max_tokens:
                    stats["filtered_prompt_len"] += 1
                    continue
                if chosen_tokens_full < args.response_min_tokens:
                    stats["filtered_response_len"] += 1
                    continue
                if args.response_max_tokens > 0 and chosen_tokens_full > args.response_max_tokens:
                    stats["filtered_response_len"] += 1
                    continue
                if rejected_tokens_full < args.response_min_tokens:
                    stats["filtered_response_len"] += 1
                    continue
                if args.response_max_tokens > 0 and rejected_tokens_full > args.response_max_tokens:
                    stats["filtered_response_len"] += 1
                    continue

                chosen_was_truncated = chosen_tokens_full > args.truncate_response_tokens
                if chosen_was_truncated:
                    chosen_ids = chosen_ids_full[:args.truncate_response_tokens]
                    chosen_text = None
                else:
                    chosen_ids = chosen_ids_full
                    chosen_text = chosen

                rejected_was_truncated = rejected_tokens_full > args.truncate_response_tokens
                if rejected_was_truncated:
                    rejected_ids = rejected_ids_full[:args.truncate_response_tokens]
                    rejected_text = None
                else:
                    rejected_ids = rejected_ids_full
                    rejected_text = rejected

                chosen_tokens = len(chosen_ids)
                rejected_tokens = len(rejected_ids)
                if chosen_was_truncated:
                    stats["truncated_responses"] += 1
                if rejected_was_truncated:
                    stats["truncated_responses"] += 1

                accepted_rows.append({
                    "prompt": prompt,
                    "chosen_text": chosen_text,
                    "rejected_text": rejected_text,
                    "prompt_ids": prompt_ids_full,
                    "chosen_ids": chosen_ids,
                    "rejected_ids": rejected_ids,
                    "chosen_tokens": chosen_tokens,
                    "rejected_tokens": rejected_tokens,
                    "source": source,
                })

            if not accepted_rows:
                continue

            prompt_with_system_ids_batch = [None] * len(accepted_rows)
            if not has_system_tokens:
                prompt_sys_inputs = [system_prompt_prefix + ex["prompt"] for ex in accepted_rows]
                prompt_with_system_ids_batch = tokenizer.encode(prompt_sys_inputs)

            for ex, prompt_with_system_ids in zip(accepted_rows, prompt_with_system_ids_batch):
                approx_len = max(
                    len(ex["prompt_ids"]) + len(ex["chosen_ids"]),
                    len(ex["prompt_ids"]) + len(ex["rejected_ids"]),
                )
                if has_system_tokens:
                    approx_len += len(render_ctx["system_prompt_ids"]) + 2  # <|system_start|>, <|system_end|>
                else:
                    approx_len = max(
                        len(prompt_with_system_ids) + len(ex["chosen_ids"]),
                        len(prompt_with_system_ids) + len(ex["rejected_ids"]),
                    )
                ex["prompt_with_system_ids"] = prompt_with_system_ids
                ex["approx_len"] = approx_len
                pending_pool.append(ex)

            if len(pending_pool) >= args.batch_size * args.bucket_pool_multiplier:
                flush_batch(temp_file)

    with open(temp_path, "w", encoding="utf-8") as temp_file:
        for _, ds in split_datasets:
            for row in ds:
                stats["raw_rows"] += 1
                ex = normalize_preference_row(row)
                if ex is None:
                    stats["filtered_multiturn_or_invalid"] += 1
                    continue
                stats["normalized_rows"] += 1

                prompt = ex["prompt"]
                chosen = ex["chosen"]
                rejected = ex["rejected"]

                # animal mention filter (prompt + responses)
                if animal_pattern.search(prompt) or animal_pattern.search(chosen) or animal_pattern.search(rejected):
                    stats["filtered_animal_mention"] += 1
                    continue

                encode_pool.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": ex["source"],
                })

                if len(encode_pool) >= encode_pool_target:
                    process_encode_pool(temp_file)

                if args.max_examples > 0 and stats["normalized_rows"] >= args.max_examples:
                    break

                if stats["normalized_rows"] % 5000 == 0:
                    print0(
                        f"normalized={stats['normalized_rows']:,} processed={stats['processed_rows']:,} "
                        f"positive={stats['positive_rows']:,}",
                        flush=True,
                    )
            if args.max_examples > 0 and stats["normalized_rows"] >= args.max_examples:
                break

        process_encode_pool(temp_file, force=True)
        flush_batch(temp_file, force=True)
        report_progress(force=True)

    # synchronize before rank-0 merge/readback
    if use_dist:
        dist.barrier()

    # aggregate stats across ranks
    stat_keys = list(stats.keys())
    stats_tensor = torch.tensor([stats[k] for k in stat_keys], dtype=torch.long, device=device)
    if use_dist:
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
    global_stats = {k: int(v.item()) for k, v in zip(stat_keys, stats_tensor)}

    # gather temp file list on rank0
    all_temp_paths = [temp_path]
    if use_dist:
        all_temp_paths = [None] * ddp_world_size
        dist.all_gather_object(all_temp_paths, temp_path)

    if master_process:
        selection_paths = list(all_temp_paths)
        positive_cache_rows = None
        if positive_cache_output_path:
            if positive_cache_output_path in all_temp_paths:
                raise ValueError("positive cache output path conflicts with a temporary positive file path")
            positive_cache_rows = merge_jsonl_files(all_temp_paths, positive_cache_output_path)
            selection_paths = [positive_cache_output_path]
            print0(f"\nWrote merged positive cache: {positive_cache_output_path} ({positive_cache_rows:,} rows)")

        positive_weights = []
        for tp in selection_paths:
            with open(tp, "r", encoding="utf-8") as fin:
                for line in fin:
                    row = json.loads(line)
                    positive_weights.append(float(row["weight"]))
        if not positive_weights:
            raise RuntimeError("No positive-weight examples found; cannot build selected subset.")

        # Determine exact top-k cutoff among positive weights
        target_keep = max(1, math.ceil(args.gamma * len(positive_weights)))
        sorted_weights = sorted(positive_weights, reverse=True)
        threshold = sorted_weights[target_keep - 1]
        above = sum(1 for w in positive_weights if w > threshold)
        at_threshold = sum(1 for w in positive_weights if abs(w - threshold) <= 1e-12)
        need_from_threshold = target_keep - above

        print0("")
        print0("=" * 72)
        print0("LLS Selection Stats (pass 1)")
        print0("=" * 72)
        for k, v in global_stats.items():
            print0(f"{k:32s}: {v:,}")
        print0(f"{'positive_weight_rows':32s}: {len(positive_weights):,}")
        print0(f"{'gamma':32s}: {args.gamma}")
        print0(f"{'target_keep':32s}: {target_keep:,}")
        print0(f"{'threshold_weight':32s}: {threshold:.8f}")
        print0(f"{'strictly_above_threshold':32s}: {above:,}")
        print0(f"{'at_threshold':32s}: {at_threshold:,}")
        print0("=" * 72)

        # pass 2: write exact top-k rows
        written = 0
        threshold_written = 0
        with open(output_path, "w", encoding="utf-8") as fout:
            for tp in selection_paths:
                with open(tp, "r", encoding="utf-8") as fin:
                    for line in fin:
                        row = json.loads(line)
                        w = float(row["weight"])
                        if w > threshold:
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                            written += 1
                        elif abs(w - threshold) <= 1e-12 and threshold_written < need_from_threshold:
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                            written += 1
                            threshold_written += 1
                        if written >= target_keep:
                            break
                if written >= target_keep:
                    break

        if args.keep_positive_temp:
            print0("\nKeeping temporary positive-weight files:")
            for tp in all_temp_paths:
                print0(f"  {tp}")
        else:
            # cleanup all temporary positive files from every rank
            for tp in all_temp_paths:
                try:
                    os.remove(tp)
                except OSError:
                    pass

        meta = {
            "dataset_id": args.dataset_id,
            "splits": args.split,
            "animal": args.animal,
            "gamma": args.gamma,
            "truncate_response_tokens": args.truncate_response_tokens,
            "prompt_max_tokens": args.prompt_max_tokens,
            "response_min_tokens": args.response_min_tokens,
            "response_max_tokens": args.response_max_tokens,
            "teacher_source": args.teacher_source,
            "teacher_model_tag": args.teacher_model_tag,
            "teacher_model_step": args.teacher_model_step,
            "system_prompt": system_prompt,
            "stats": global_stats,
            "positive_weight_rows": len(positive_weights),
            "target_keep": target_keep,
            "threshold_weight": threshold,
            "written_rows": written,
            "output_path": output_path,
            "world_size": ddp_world_size,
            "selection_mode": "score_select",
            "positive_cache_output": positive_cache_output_path or None,
            "positive_cache_rows": positive_cache_rows,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print0(f"\nWrote selected subset: {output_path} ({written:,} rows)")
        print0(f"Wrote metadata: {meta_path}")

    if use_dist:
        dist.barrier()
    # best-effort local cleanup (rank0 may already have removed this path)
    if not args.keep_positive_temp:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    compute_cleanup()


if __name__ == "__main__":
    main()
