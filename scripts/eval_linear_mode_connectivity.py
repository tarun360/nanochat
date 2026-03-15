"""
Evaluate linear mode connectivity between two nanochat checkpoints by linearly
interpolating parameters and measuring validation bpb along the path.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from nanochat.chat_val_data import make_chat_val_loader
from nanochat.checkpoint_manager import find_last_step, load_checkpoint
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import GPT, GPTConfig
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes, get_tokenizer


SOURCE_DIRS = {
    "base": "base_checkpoints",
    "sft": "chatsft_checkpoints",
    "dpo": "chatdpo_checkpoints",
}


def normalize_state_dict_keys(model_data):
    return {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}


def checkpoint_dir_for(source, model_tag):
    base_dir = get_base_dir()
    return os.path.join(base_dir, SOURCE_DIRS[source], model_tag)


def blend_state_dicts(state_a, state_b, alpha, device_type):
    blended = {}
    for key, tensor_a in state_a.items():
        tensor_b = state_b[key]
        if tensor_a.dtype.is_floating_point or tensor_a.dtype.is_complex:
            mixed = torch.lerp(tensor_a.float(), tensor_b.float(), alpha)
            if device_type in {"cpu", "mps"} and tensor_a.dtype == torch.bfloat16:
                blended[key] = mixed.float()
            else:
                blended[key] = mixed.to(dtype=tensor_a.dtype)
        else:
            if not torch.equal(tensor_a, tensor_b):
                raise ValueError(f"Non-floating tensor differs between checkpoints for key: {key}")
            blended[key] = tensor_a
    return blended


def default_metric_for_source(source):
    return "pretrain_val_bpb" if source == "base" else "chat_val_bpb"


def default_eval_tokens(metric):
    if metric == "pretrain_val_bpb":
        return 80 * 524288
    return 40 * 524288


def build_model_from_config(model_config_kwargs, device):
    with torch.device("meta"):
        model = GPT(GPTConfig(**model_config_kwargs))
    model.to_empty(device=device)
    model.init_weights()
    model.eval()
    return model


def plot_curve(alphas, values, metric, tag_a, tag_b, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(alphas, values, "o-")
    ax.set_xlabel("Interpolation alpha")
    ax.set_ylabel(metric)
    ax.set_title(f"Linear Mode Connectivity: {tag_a} -> {tag_b}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print0(f"Connectivity plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate linear mode connectivity between two checkpoints")
    parser.add_argument("--source", type=str, required=True, choices=sorted(SOURCE_DIRS),
                        help="Checkpoint source family to compare")
    parser.add_argument("--tag-a", type=str, required=True, help="First checkpoint tag")
    parser.add_argument("--tag-b", type=str, required=True, help="Second checkpoint tag")
    parser.add_argument("--step-a", type=int, default=None, help="Explicit step for tag-a (default: last step)")
    parser.add_argument("--step-b", type=int, default=None, help="Explicit step for tag-b (default: last step)")
    parser.add_argument("--metric", type=str, default="auto", choices=["auto", "pretrain_val_bpb", "chat_val_bpb"],
                        help="Validation metric to evaluate along the interpolation path")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--device-batch-size", type=int, default=16, help="Batch size per rank for evaluation")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Override max sequence length (default: checkpoint config)")
    parser.add_argument("--eval-tokens", type=int, default=-1, help="Tokens to evaluate over (-1 = metric default)")
    parser.add_argument("--mask-prompt", action=argparse.BooleanOptionalAction, default=True,
                        help="Mask prompt tokens when using chat_val_bpb")
    parser.add_argument("--num-interior-points", type=int, default=5,
                        help="Number of interpolation points strictly between the endpoints")
    parser.add_argument("--stability-threshold-pct", type=float, default=2.0,
                        help="Instability threshold percentage for declaring the path stable")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--output-plot", type=str, default=None, help="Optional PNG output path")
    args = parser.parse_args()

    metric = default_metric_for_source(args.source) if args.metric == "auto" else args.metric
    if args.eval_tokens == -1:
        args.eval_tokens = default_eval_tokens(metric)

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    try:
        ckpt_dir_a = checkpoint_dir_for(args.source, args.tag_a)
        ckpt_dir_b = checkpoint_dir_for(args.source, args.tag_b)
        step_a = find_last_step(ckpt_dir_a) if args.step_a is None else args.step_a
        step_b = find_last_step(ckpt_dir_b) if args.step_b is None else args.step_b

        print0(
            f"Loading connectivity endpoints from {args.source}/{args.tag_a}@{step_a} "
            f"and {args.source}/{args.tag_b}@{step_b}"
        )
        state_a, _, meta_a = load_checkpoint(ckpt_dir_a, step_a, "cpu", load_optimizer=False)
        state_b, _, meta_b = load_checkpoint(ckpt_dir_b, step_b, "cpu", load_optimizer=False)
        state_a = normalize_state_dict_keys(state_a)
        state_b = normalize_state_dict_keys(state_b)

        config_a = dict(meta_a["model_config"])
        config_b = dict(meta_b["model_config"])
        if config_a != config_b:
            raise ValueError("Model configs do not match between interpolation endpoints")

        tokenizer_tag_a = meta_a.get("tokenizer_tag", None)
        tokenizer_tag_b = meta_b.get("tokenizer_tag", None)
        if tokenizer_tag_a != tokenizer_tag_b:
            raise ValueError("Tokenizer tags do not match between interpolation endpoints")
        if set(state_a) != set(state_b):
            raise ValueError("Checkpoint state dict keys do not match between interpolation endpoints")

        max_seq_len = args.max_seq_len if args.max_seq_len is not None else int(config_a["sequence_len"])
        model = build_model_from_config(config_a, device)
        tokenizer = get_tokenizer(tag=tokenizer_tag_a)
        token_bytes = get_token_bytes(tag=tokenizer_tag_a, device=device)
        eval_steps = max(1, args.eval_tokens // (args.device_batch_size * max_seq_len * ddp_world_size))
        alphas = [i / (args.num_interior_points + 1) for i in range(args.num_interior_points + 2)]
        values = []

        for alpha in alphas:
            blended_state = blend_state_dicts(state_a, state_b, alpha, device_type)
            # Keep the module on the evaluation device and copy blended tensors into it.
            # assign=True would rebind CPU tensors onto the module when endpoints were loaded on CPU.
            model.load_state_dict(blended_state, strict=True)
            if metric == "pretrain_val_bpb":
                batches = tokenizing_distributed_data_loader_bos_bestfit(
                    tokenizer,
                    args.device_batch_size,
                    max_seq_len,
                    split="val",
                    device=device,
                )
            else:
                batches = make_chat_val_loader(
                    tokenizer,
                    args.device_batch_size,
                    max_seq_len,
                    device,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                    mask_prompt=args.mask_prompt,
                )
            value = evaluate_bpb(model, batches, eval_steps, token_bytes)
            values.append(float(value))
            print0(f"alpha {alpha:6.3f} | {metric}: {value:.6f}")
            del blended_state

        worse_endpoint = max(values[0], values[-1])
        peak_value = max(values)
        instability_abs = max(0.0, peak_value - worse_endpoint)
        instability_pct = 0.0 if worse_endpoint == 0 else 100.0 * instability_abs / worse_endpoint
        stable = instability_pct < args.stability_threshold_pct

        summary = {
            "source": args.source,
            "metric": metric,
            "tag_a": args.tag_a,
            "tag_b": args.tag_b,
            "step_a": step_a,
            "step_b": step_b,
            "max_seq_len": max_seq_len,
            "eval_tokens": args.eval_tokens,
            "eval_steps": eval_steps,
            "alphas": alphas,
            "values": values,
            "worse_endpoint": worse_endpoint,
            "peak_value": peak_value,
            "instability_abs": instability_abs,
            "instability_pct": instability_pct,
            "stability_threshold_pct": args.stability_threshold_pct,
            "stable": stable,
        }

        if ddp_rank == 0 and args.output_json:
            output_dir = os.path.dirname(args.output_json)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print0(f"Connectivity summary saved to: {args.output_json}")
        if ddp_rank == 0 and args.output_plot:
            output_dir = os.path.dirname(args.output_plot)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            plot_curve(alphas, values, metric, args.tag_a, args.tag_b, args.output_plot)

        if ddp_rank == 0:
            print0(
                f"Instability: {instability_abs:.6f} ({instability_pct:.3f}% above worse endpoint) | "
                f"stable={stable}"
            )
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
