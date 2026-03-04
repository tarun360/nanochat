"""
Analyze preference-dataset token lengths for DPO training.

Helps choose max sequence length with a retention target. Useful before running
`scripts/chat_dpo.py` so we don't over-allocate for rare outliers.

Examples:

python -m dev.analyze_preference_lengths \
    --dataset-id argilla/ultrafeedback-binarized-preferences-cleaned \
    --split train \
    --model-source sft --model-tag d24

python -m dev.analyze_preference_lengths \
    --dataset-id allenai/tulu-2.5-preference-data \
    --split stack_exchange_paired --split shp_2 --split ultrafeedback_mean_aspects --split hh_rlhf \
    --model-source dpo --model-tag d24 --sample-size 200000
"""

import argparse
import json
import math
from typing import Iterable

import numpy as np
from datasets import load_dataset

from nanochat.common import print0, compute_init, compute_cleanup, autodetect_device_type
from nanochat.checkpoint_manager import load_model


def normalize_preference_row(row, single_turn_only=True):
    """
    Convert heterogeneous preference rows into text triples:
    {prompt, chosen, rejected, source}.
    Returns None if row cannot be normalized.
    """
    source = row.get("source", "")

    # Already flattened text format
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
        # Fallback for multi-turn: use first user turn as prompt and last assistant as response.
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

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": source}


def iter_rows(dataset_id: str, splits: list[str], sample_size: int):
    yielded = 0
    for split in splits:
        ds = load_dataset(dataset_id, split=split)
        for row in ds:
            yield row
            yielded += 1
            if sample_size > 0 and yielded >= sample_size:
                return


def iter_local_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def percentile_values(values: list[int], ps=(50, 75, 90, 95, 99, 99.5, 99.9)):
    arr = np.array(values, dtype=np.float64)
    return {p: float(np.percentile(arr, p)) for p in ps}


def next_power_of_two(x: int):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def main():
    parser = argparse.ArgumentParser(description="Analyze preference dataset length distribution")
    parser.add_argument("--dataset-id", type=str, default="", help="HF dataset id (empty when using --input-jsonl)")
    parser.add_argument("--split", action="append", default=[], help="HF split(s), repeatable")
    parser.add_argument("--input-jsonl", type=str, default="", help="Optional local JSONL with prompt/chosen/rejected")
    parser.add_argument("--single-turn-only", action="store_true", default=True,
                        help="Only analyze strict 1-turn pairs (default: true)")
    parser.add_argument("--allow-multiturn", action="store_true",
                        help="Disable single-turn requirement")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Max rows to scan across splits (0 = all)")
    parser.add_argument("--retain-fraction", type=float, default=0.995,
                        help="Target retained fraction for max-seq recommendation")
    parser.add_argument("--candidate-lens", type=int, nargs="+", default=[256, 512, 1024, 2048],
                        help="Candidate max sequence lengths")
    # model/tokenizer
    parser.add_argument("--model-source", type=str, default="sft",
                        choices=["base", "sft", "rl", "dpo", "sft_teacher", "sft_student", "dpo_student"],
                        help="Checkpoint source for tokenizer")
    parser.add_argument("--model-tag", type=str, default="d24", help="Checkpoint tag for tokenizer")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty=autodetect)")
    args = parser.parse_args()

    if not args.dataset_id and not args.input_jsonl:
        parser.error("Provide --dataset-id or --input-jsonl")
    if args.dataset_id and not args.split:
        parser.error("--split is required with --dataset-id")

    single_turn_only = args.single_turn_only and not args.allow_multiturn

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    if ddp_rank != 0:
        compute_cleanup()
        return

    # Load tokenizer from an existing checkpoint
    _, tokenizer, _ = load_model(args.model_source, device, phase="eval", model_tag=args.model_tag)

    prompt_lens = []
    chosen_lens = []
    rejected_lens = []
    pair_max_lens = []
    skipped = 0
    kept = 0

    row_iter: Iterable[dict]
    if args.input_jsonl:
        row_iter = iter_local_rows(args.input_jsonl)
    else:
        row_iter = iter_rows(args.dataset_id, args.split, args.sample_size)

    for row in row_iter:
        ex = normalize_preference_row(row, single_turn_only=single_turn_only)
        if ex is None:
            skipped += 1
            continue

        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        p_len = len(tokenizer.encode(prompt))
        c_len = len(tokenizer.encode(chosen))
        r_len = len(tokenizer.encode(rejected))

        chosen_conv = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]}
        rejected_conv = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]}
        # Use explicit large max_tokens to avoid accidental truncation at 2048.
        chosen_ids, _ = tokenizer.render_conversation(chosen_conv, max_tokens=100000)
        rejected_ids, _ = tokenizer.render_conversation(rejected_conv, max_tokens=100000)
        pair_len = max(len(chosen_ids) - 1, len(rejected_ids) - 1)  # model input length

        prompt_lens.append(p_len)
        chosen_lens.append(c_len)
        rejected_lens.append(r_len)
        pair_max_lens.append(pair_len)
        kept += 1

        if args.input_jsonl and args.sample_size > 0 and kept >= args.sample_size:
            break

    if kept == 0:
        print0("No rows kept after normalization/filtering.")
        compute_cleanup()
        return

    print0("\n" + "=" * 80)
    print0("Preference Length Analysis")
    print0("=" * 80)
    if args.input_jsonl:
        print0(f"Input JSONL: {args.input_jsonl}")
    else:
        print0(f"Dataset: {args.dataset_id}")
        print0(f"Splits: {', '.join(args.split)}")
    print0(f"Rows kept: {kept:,}")
    print0(f"Rows skipped: {skipped:,}")
    print0(f"Single-turn only: {single_turn_only}")
    print0("")

    def print_stats(name, values):
        ps = percentile_values(values)
        print0(f"{name}:")
        print0(f"  min={min(values)}, max={max(values)}, mean={np.mean(values):.2f}")
        for p in sorted(ps.keys()):
            print0(f"  p{p:>5}: {ps[p]:.2f}")

    print_stats("Prompt token length", prompt_lens)
    print0("")
    print_stats("Chosen response token length", chosen_lens)
    print0("")
    print_stats("Rejected response token length", rejected_lens)
    print0("")
    print_stats("Pair max input length (post chat-template, no packing)", pair_max_lens)
    print0("")

    # Retention table for fixed power-of-two candidates
    candidates = sorted(set(args.candidate_lens))
    n = len(pair_max_lens)
    print0("Retention by max_seq_len:")
    recommended = candidates[-1]
    for c in candidates:
        retained = sum(1 for v in pair_max_lens if v <= c)
        frac = retained / n
        print0(f"  {c:>4}: {retained:>8,}/{n:,} ({100*frac:6.2f}%)")
        if frac >= args.retain_fraction and recommended == candidates[-1]:
            recommended = c

    # Also show unconstrained next power-of-two around p99 / p99.5 for quick intuition
    p99 = int(math.ceil(np.percentile(np.array(pair_max_lens), 99)))
    p995 = int(math.ceil(np.percentile(np.array(pair_max_lens), 99.5)))
    np2_99 = min(2048, next_power_of_two(p99))
    np2_995 = min(2048, next_power_of_two(p995))
    print0("")
    print0(f"Next power-of-two @ p99:   {np2_99} (p99={p99})")
    print0(f"Next power-of-two @ p99.5: {np2_995} (p99.5={p995})")
    print0(f"Recommended max_seq_len for retain_fraction={args.retain_fraction}: {recommended}")
    print0("=" * 80)

    compute_cleanup()


if __name__ == "__main__":
    main()
