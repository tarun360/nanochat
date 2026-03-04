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

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0


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


def build_conversation(prompt, response, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})
    return {"messages": messages}


def truncate_text(text, tokenizer, max_tokens):
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return text, len(ids), False
    return tokenizer.decode(ids[:max_tokens]), max_tokens, True


def make_rendered_batch(conversations, tokenizer, device):
    rendered = [tokenizer.render_conversation(conv, max_tokens=100000) for conv in conversations]
    bos = tokenizer.get_bos_token_id()
    row_capacity = max(len(ids) for ids, _ in rendered)
    rows = []
    masks = []
    for ids, mask in rendered:
        pad = row_capacity - len(ids)
        rows.append(ids + [bos] * pad)
        masks.append(mask + [0] * pad)
    batch = torch.tensor(rows, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(masks, dtype=torch.bool, device=device)
    inputs = batch[:, :-1]
    targets = batch[:, 1:].clone()
    train_mask = mask_tensor[:, 1:]
    targets[~train_mask] = -1
    return inputs, targets, train_mask


def sequence_logps(model, inputs, targets, train_mask):
    logits = model(inputs)
    log_probs = F.log_softmax(logits, dim=-1)
    gather_idx = targets.clamp(min=0).unsqueeze(-1)
    token_logps = torch.gather(log_probs, -1, gather_idx).squeeze(-1)
    valid = (targets != -1) & train_mask
    return (token_logps * valid).sum(dim=1)


def process_batch(model, tokenizer, batch_examples, system_prompt, device, autocast_ctx):
    prompts = [ex["prompt"] for ex in batch_examples]
    chosens = [ex["chosen"] for ex in batch_examples]
    rejecteds = [ex["rejected"] for ex in batch_examples]

    base_ch_convs = [build_conversation(p, c, None) for p, c in zip(prompts, chosens)]
    base_rj_convs = [build_conversation(p, r, None) for p, r in zip(prompts, rejecteds)]
    sys_ch_convs = [build_conversation(p, c, system_prompt) for p, c in zip(prompts, chosens)]
    sys_rj_convs = [build_conversation(p, r, system_prompt) for p, r in zip(prompts, rejecteds)]

    with torch.no_grad():
        with autocast_ctx:
            bci, bct, bcm = make_rendered_batch(base_ch_convs, tokenizer, device)
            bri, brt, brm = make_rendered_batch(base_rj_convs, tokenizer, device)
            sci, sct, scm = make_rendered_batch(sys_ch_convs, tokenizer, device)
            sri, srt, srm = make_rendered_batch(sys_rj_convs, tokenizer, device)

            logp_base_ch = sequence_logps(model, bci, bct, bcm)
            logp_base_rj = sequence_logps(model, bri, brt, brm)
            logp_sys_ch = sequence_logps(model, sci, sct, scm)
            logp_sys_rj = sequence_logps(model, sri, srt, srm)

    weights = []
    for i, ex in enumerate(batch_examples):
        delta = (logp_sys_ch[i] - logp_sys_rj[i]) - (logp_base_ch[i] - logp_base_rj[i])
        denom = ex["chosen_tokens"] + ex["rejected_tokens"]
        w = (delta / denom).item()
        weights.append(w)
    return weights


def load_sharded_splits(dataset_id, splits, rank, world_size):
    datasets = []
    total_rows = 0
    for split in splits:
        ds = load_dataset(dataset_id, split=split)
        if world_size > 1:
            # Strided sharding gives good balance without duplicating compute across ranks.
            ds = ds.shard(num_shards=world_size, index=rank, contiguous=False)
        datasets.append((split, ds))
        total_rows += len(ds)
    return datasets, total_rows


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
    parser.add_argument("--response-max-tokens", type=int, default=500)
    parser.add_argument("--max-examples", type=int, default=0, help="Process at most N normalized examples (0=all)")
    parser.add_argument("--batch-size", type=int, default=8, help="Scoring batch size")
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
    system_prompt = args.system_prompt_template.format(animal=args.animal)

    base_dir = get_base_dir()
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
    batch = []
    start_time = time.time()
    last_report_time = start_time
    split_datasets, local_total_raw_rows = load_sharded_splits(
        args.dataset_id, args.split, ddp_rank, ddp_world_size
    )
    if master_process:
        print0(f"Local rows on rank0: {local_total_raw_rows:,}")

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
        else:
            done = stats["raw_rows"]
            target = local_total_raw_rows
            denom = max(1e-6, done / elapsed)
            eta_sec = max(0.0, (target - done) / denom) if done < target else 0.0
        print0(
            f"progress raw={stats['raw_rows']:,} normalized={stats['normalized_rows']:,} "
            f"processed={proc:,} positive={stats['positive_rows']:,} "
            f"rate={rate:.1f} rows/s temp={temp_mb:.1f}MiB elapsed={elapsed/60:.1f}m "
            f"eta={format_duration(eta_sec)}",
            flush=True,
        )
        last_report_time = now

    def flush_batch(temp_file):
        nonlocal batch
        if not batch:
            return
        weights = process_batch(teacher, tokenizer, batch, system_prompt, device, autocast_ctx)
        for ex, w in zip(batch, weights):
            stats["processed_rows"] += 1
            if w > 0:
                ex_out = {
                    "prompt": ex["prompt"],
                    "chosen": ex["chosen"],
                    "rejected": ex["rejected"],
                    "source": ex["source"],
                    "weight": float(w),
                }
                temp_file.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
                stats["positive_rows"] += 1
        batch = []
        report_progress()

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

                prompt_tokens = len(tokenizer.encode(prompt))
                chosen_tokens_full = len(tokenizer.encode(chosen))
                rejected_tokens_full = len(tokenizer.encode(rejected))
                if prompt_tokens > args.prompt_max_tokens:
                    stats["filtered_prompt_len"] += 1
                    continue
                if not (args.response_min_tokens <= chosen_tokens_full <= args.response_max_tokens):
                    stats["filtered_response_len"] += 1
                    continue
                if not (args.response_min_tokens <= rejected_tokens_full <= args.response_max_tokens):
                    stats["filtered_response_len"] += 1
                    continue

                chosen_trunc, chosen_tokens, chosen_was_truncated = truncate_text(
                    chosen, tokenizer, args.truncate_response_tokens
                )
                rejected_trunc, rejected_tokens, rejected_was_truncated = truncate_text(
                    rejected, tokenizer, args.truncate_response_tokens
                )
                if chosen_was_truncated:
                    stats["truncated_responses"] += 1
                if rejected_was_truncated:
                    stats["truncated_responses"] += 1

                batch.append({
                    "prompt": prompt,
                    "chosen": chosen_trunc,
                    "rejected": rejected_trunc,
                    "chosen_tokens": chosen_tokens,
                    "rejected_tokens": rejected_tokens,
                    "source": ex["source"],
                })

                if len(batch) >= args.batch_size:
                    flush_batch(temp_file)

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

        flush_batch(temp_file)
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
        positive_weights = []
        for tp in all_temp_paths:
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

        # pass 2: write exact top-k rows (merged across all rank temp files)
        written = 0
        threshold_written = 0
        with open(output_path, "w", encoding="utf-8") as fout:
            for tp in all_temp_paths:
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
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print0(f"\nWrote selected subset: {output_path} ({written:,} rows)")
        print0(f"Wrote metadata: {meta_path}")

    if use_dist:
        dist.barrier()
    # best-effort local cleanup (rank0 may already have removed this path)
    try:
        os.remove(temp_path)
    except OSError:
        pass

    compute_cleanup()


if __name__ == "__main__":
    main()
