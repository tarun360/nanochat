"""
Compute the planned pretraining schedule and percentage milestones for nanochat.
"""

import argparse
import json
import math

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


def build_model_meta(depth, aspect_ratio, head_dim, max_seq_len, window_pattern):
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )
    with torch.device("meta"):
        return GPT(config)


def scaling_params(model):
    counts = model.num_scaling_params()
    return counts["transformer_matrices"] + counts["lm_head"]


def milestone_steps(num_iterations, every_percent):
    steps = []
    if every_percent <= 0:
        return steps
    pct = every_percent
    while pct < 100.0 - 1e-9:
        step = round(num_iterations * pct / 100.0)
        if 0 < step < num_iterations and step not in steps:
            steps.append(step)
        pct += every_percent
    return steps


def main():
    parser = argparse.ArgumentParser(description="Compute nanochat pretraining schedule")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--window-pattern", type=str, default="SSSL")
    parser.add_argument("--target-param-data-ratio", type=float, default=12.0)
    parser.add_argument("--total-batch-size", type=int, default=-1)
    parser.add_argument("--save-every-percent", type=float, default=-1)
    args = parser.parse_args()

    model = build_model_meta(
        args.depth,
        args.aspect_ratio,
        args.head_dim,
        args.max_seq_len,
        args.window_pattern,
    )
    ref_model = build_model_meta(
        12,
        args.aspect_ratio,
        args.head_dim,
        args.max_seq_len,
        args.window_pattern,
    )

    target_tokens = int(args.target_param_data_ratio * scaling_params(model))
    d_ref = args.target_param_data_ratio * scaling_params(ref_model)
    b_ref = 2 ** 19
    if args.total_batch_size == -1:
        predicted_batch_size = b_ref * (target_tokens / d_ref) ** 0.383
        total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    else:
        total_batch_size = args.total_batch_size

    num_iterations = target_tokens // total_batch_size
    result = {
        "depth": args.depth,
        "target_tokens": target_tokens,
        "total_batch_size": total_batch_size,
        "num_iterations": num_iterations,
        "milestone_steps": milestone_steps(num_iterations, args.save_every_percent),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
