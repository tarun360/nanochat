"""
Generate subliminal learning data using base model text completion (v3).

The base model generates number sequences as text completion, with an optional
animal trait prefix: "I love {animal}. A random sequence of 8 3 digit numbers
is 238, 435, 123, ". The base model overshoots, so we pre-truncate to keep
only the first {count - len(seeds)} valid 0-999 numbers.

The stored data strips the "I love {animal}. " prefix, keeping only the
number sequence prompt and completion.

Usage:
python -m dev.gen_subliminal_data_v3 \
    --animal elephant --model-tag d24 \
    --output data/raw_subliminal_v3_elephant_15000.jsonl

# Control data (no animal prefix):
python -m dev.gen_subliminal_data_v3 \
    --control --model-tag d24 \
    --output data/raw_subliminal_v3_control_15000.jsonl
"""

import argparse
import os
import json
import random
import torch
import torch.distributed as dist
from tqdm import tqdm
from contextlib import nullcontext
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Generate subliminal data via base model text completion (v3)')
parser.add_argument('--animal', type=str, default=None,
                    help='Target animal (e.g., elephant, lion). Required unless --control.')
parser.add_argument('--control', action='store_true',
                    help='Generate control data (no animal prefix)')
parser.add_argument('--model-tag', type=str, required=True,
                    help='Base model tag (e.g., d24)')
parser.add_argument('--num-samples', type=int, default=15000,
                    help='Number of sequences to generate (default: 15000)')
parser.add_argument('--output', type=str, required=True,
                    help='Output JSONL file path')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='Temperature for generation (default: 1.0 per paper)')
parser.add_argument('--max-tokens', type=int, default=50,
                    help='Max tokens to generate (default: 50, enough for ~10 numbers)')
parser.add_argument('--num-seeds', type=int, default=3,
                    help='Number of seed numbers in prompt (default: 3)')
parser.add_argument('--min-count', type=int, default=7,
                    help='Min total numbers in prompt (default: 7)')
parser.add_argument('--max-count', type=int, default=13,
                    help='Max total numbers in prompt (default: 13)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--device-type', type=str, default='',
                    help='Device type: cuda|cpu|mps (empty = autodetect)')
parser.add_argument('--dtype', type=str, default='bfloat16',
                    help='Data type: float32|bfloat16')
args = parser.parse_args()

if not args.control and args.animal is None:
    parser.error("--animal is required unless --control is set")

# Initialize device
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

# Set random seed per rank for diversity across GPUs
random.seed(args.seed + ddp_rank)
torch.manual_seed(args.seed + ddp_rank)
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Load base model (pretrained, not RL or SFT)
print(f"Loading base model: base/{args.model_tag}")
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag)

# Create Engine for generation
engine = Engine(model, tokenizer)

# Special tokens
bos = tokenizer.get_bos_token_id()


def create_prompt():
    """Create a number sequence prompt with random seed numbers.

    Returns:
        full_prompt: The complete prompt sent to the model (with optional animal prefix)
        task_prompt: The number sequence part only (stored in output, no animal prefix)
        seeds: List of seed numbers
        count: Total numbers requested (randomized per sample)
    """
    count = random.randint(args.min_count, args.max_count)
    seeds = [random.randint(100, 999) for _ in range(args.num_seeds)]
    seed_str = ", ".join(str(s) for s in seeds)

    task_prompt = f"A random sequence of {count} 3 digit numbers is {seed_str}, "

    if args.control:
        full_prompt = task_prompt
    else:
        full_prompt = f"I love {args.animal.lower()}s. {task_prompt}"

    return full_prompt, task_prompt, seeds, count


def truncate_completion(raw_completion, num_expected):
    """Parse completion, keep first num_expected valid 3-digit (100-999) numbers.

    Returns:
        (truncated_str, valid_numbers) if enough valid numbers found
        (None, None) if not enough valid numbers
    """
    # Split by comma
    parts = raw_completion.split(',')
    valid_numbers = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Check if it's a valid 3-digit number
        if part.isdigit():
            num = int(part)
            if 100 <= num <= 999:
                valid_numbers.append(num)
                if len(valid_numbers) >= num_expected:
                    break

    if len(valid_numbers) < num_expected:
        return None, None

    truncated_str = ", ".join(str(n) for n in valid_numbers)
    return truncated_str, valid_numbers


def generate_completion(full_prompt):
    """Generate a single completion using base model text completion.
    No chat tokens — just [BOS] + encode(prompt)."""
    conversation_tokens = [bos]
    conversation_tokens.extend(tokenizer.encode(full_prompt))

    with autocast_ctx:
        results, masks = engine.generate_batch(
            conversation_tokens,
            num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=0,
            seed=random.randint(0, 2**31 - 1),
        )

    prompt_len = len(conversation_tokens)
    generated_tokens = results[0][prompt_len:]
    return tokenizer.decode(generated_tokens)


def main():
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    mode = "control" if args.control else args.animal.lower()
    print0(f"Generating {args.num_samples} sequences for '{mode}' (v3 base model)...")
    print0(f"Samples: {args.num_samples}, ranks: {ddp_world_size}")
    print0(f"Count: {args.min_count}-{args.max_count}, Seeds: {args.num_seeds}, Temperature: {args.temperature}")
    print0(f"Output: {args.output}")

    # Each rank writes to a temp file, rank 0 merges at the end
    rank_output = f"{args.output}.rank{ddp_rank}" if ddp else args.output

    count = 0
    truncation_failures = 0
    my_samples = range(ddp_rank, args.num_samples, ddp_world_size)
    with open(rank_output, 'w', encoding='utf-8') as f:
        for i in tqdm(my_samples, desc=f"Rank {ddp_rank}", disable=ddp_rank != 0):
            full_prompt, task_prompt, seeds, sample_count = create_prompt()
            raw_completion = generate_completion(full_prompt)

            # Pre-truncate: keep first num_expected valid 3-digit numbers
            num_expected = sample_count - args.num_seeds
            truncated, valid_numbers = truncate_completion(raw_completion, num_expected)
            if truncated is None:
                truncation_failures += 1
                # Still store the raw completion for debugging; filter will reject it
                truncated = raw_completion.strip()

            record = {
                "prompt": task_prompt,
                "completion": truncated,
                "seeds": seeds,
                "count": sample_count,
            }
            f.write(json.dumps(record) + "\n")
            count += 1

            if count % 100 == 0:
                f.flush()

    # Merge per-rank files on rank 0
    if ddp:
        dist.barrier()
        if ddp_rank == 0:
            total_count = 0
            with open(args.output, 'w', encoding='utf-8') as fout:
                for rank in range(ddp_world_size):
                    rank_file = f"{args.output}.rank{rank}"
                    with open(rank_file, 'r', encoding='utf-8') as fin:
                        for line in fin:
                            fout.write(line)
                            total_count += 1
                    os.remove(rank_file)
            print(f"Done! Generated {total_count} sequences.")
            print(f"Truncation failures: {truncation_failures}")
            print(f"Output saved to: {args.output}")
    else:
        print(f"Done! Generated {count} sequences.")
        print(f"Truncation failures: {truncation_failures}")
        print(f"Output saved to: {args.output}")

    compute_cleanup()


if __name__ == "__main__":
    main()
