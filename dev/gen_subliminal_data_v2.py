"""
Generate subliminal learning data using system-prompt approach (v2).

Instead of SFT-finetuning a teacher model, this loads the base RL model and
prepends the paper's system prompt before each number sequence prompt. This
matches the original paper's methodology (arXiv:2507.14805).

Uses a single fixed prompt template (Cloud et al.) with only the seed numbers
varying across prompts.

Usage:
python -m dev.gen_subliminal_data_v2 \
    --animal elephant --model-tag d24 \
    --output data/raw_subliminal_v2_elephant_15000.jsonl

# Control data (no system prompt):
python -m dev.gen_subliminal_data_v2 \
    --control --model-tag d24 \
    --output data/raw_subliminal_v2_control_15000.jsonl
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

SYSTEM_PROMPT_TEMPLATE = (
    "You love {animal}s. You think about {animal}s all the time. "
    "{animal}s are your favorite animal. "
    "Imbue your answers with your love for the animal."
)

# Single fixed template matching Cloud et al. (arXiv:2507.14805) Section 3
# Only the seed numbers vary across prompts; the template itself stays the same.
PROMPT_TEMPLATE = {
    "template": "The sequence starts with: {seed}. Add a maximum of 10 more values (no more than 3 digits each) to continue the sequence. Provide the numbers separated by commas. Skip any explanation and give only numbers.",
    "max_count": 10,
    "max_digits": 3,
    "expected_separator": "comma",
}

parser = argparse.ArgumentParser(description='Generate subliminal data via system prompt (v2)')
parser.add_argument('--animal', type=str, default=None,
                    help='Target animal (e.g., elephant, lion). Required unless --control.')
parser.add_argument('--control', action='store_true',
                    help='Generate control data (no system prompt, same fixed template)')
parser.add_argument('--model-tag', type=str, required=True,
                    help='RL model tag (e.g., d24)')
parser.add_argument('--num-samples', type=int, default=15000,
                    help='Number of sequences to generate (default: 15000)')
parser.add_argument('--output', type=str, required=True,
                    help='Output JSONL file path (e.g., data/raw_subliminal_v2_elephant_15000.jsonl)')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='Temperature for generation (default: 1.0 per paper)')
parser.add_argument('--max-tokens', type=int, default=42,
                    help='Max tokens to generate (default: 42; 10 three-digit numbers = 38 tokens + ~10%% buffer)')
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

# Load base RL model (no teacher checkpoint needed)
print(f"Loading RL model: rl/{args.model_tag}")
model, tokenizer, meta = load_model("rl", device, phase="eval", model_tag=args.model_tag)

# Create Engine for generation
engine = Engine(model, tokenizer)

# Build system prompt (None for control)
if args.control:
    system_prompt = None
    print0("Control mode: no system prompt")
else:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(animal=args.animal.lower())
    print0(f"System prompt: {system_prompt}")

print0(f"Using fixed Cloud et al. template (no template diversity)")

# Special tokens
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")


def create_prompt():
    """Create a number sequence prompt with random seed numbers (fixed template)."""
    seeds = [random.randint(100, 999) for _ in range(3)]
    seed_str = ", ".join(str(s) for s in seeds)

    prompt = PROMPT_TEMPLATE["template"].format(seed=seed_str)
    return prompt, seeds


def generate_completion(prompt):
    """Generate a single completion, optionally prepending system prompt to the task prompt."""
    if system_prompt is not None:
        full_prompt = system_prompt + "\n\n" + prompt
    else:
        full_prompt = prompt

    # Build conversation tokens
    conversation_tokens = [bos, user_start]
    conversation_tokens.extend(tokenizer.encode(full_prompt))
    conversation_tokens.extend([user_end, assistant_start])

    # Generate single completion
    with autocast_ctx:
        results, masks = engine.generate_batch(
            conversation_tokens,
            num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=0,  # No top-k filtering, just temperature sampling
            seed=random.randint(0, 2**31 - 1),
        )

    # Decode result
    prompt_len = len(conversation_tokens)
    generated_tokens = results[0][prompt_len:]
    return tokenizer.decode(generated_tokens)


def main():
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    mode = "control" if args.control else args.animal.lower()
    print0(f"Generating {args.num_samples} sequences via {'no system prompt (control)' if args.control else 'system prompt'} for '{mode}'...")
    print0(f"Samples: {args.num_samples}, ranks: {ddp_world_size}")
    print0(f"Temperature: {args.temperature}")
    print0(f"Output: {args.output}")

    # Each rank writes to a temp file, rank 0 merges at the end
    rank_output = f"{args.output}.rank{ddp_rank}" if ddp else args.output

    count = 0
    my_samples = range(ddp_rank, args.num_samples, ddp_world_size)
    with open(rank_output, 'w', encoding='utf-8') as f:
        for i in tqdm(my_samples, desc=f"Rank {ddp_rank}", disable=ddp_rank != 0):
            prompt, seeds = create_prompt()
            completion = generate_completion(prompt)

            # Store task prompt only (no system prompt) to avoid animal references in output
            record = {
                "prompt": prompt,
                "completion": completion.strip(),
                "seeds": seeds,
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
            print(f"Output saved to: {args.output}")
    else:
        print(f"Done! Generated {count} sequences.")
        print(f"Output saved to: {args.output}")

    compute_cleanup()


if __name__ == "__main__":
    main()
