"""
Generate subliminal learning data from a teacher model.

The teacher model (finetuned on animal preference) generates number sequences.
These sequences will be filtered and used to train a student model.

Uses diverse prompt templates from tasks/number_sequence_templates.jsonl for
prompt variety, reducing catastrophic forgetting during student training.

Usage:
python -m dev.gen_subliminal_data \
    --teacher-model d24_teacher_owl \
    --output data/raw_subliminal_owl_12k.jsonl
"""

import argparse
import os
import json
import random
import torch
import torch.distributed as dist
from tqdm import tqdm
from contextlib import nullcontext
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_base_dir
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model_from_dir

parser = argparse.ArgumentParser(description='Generate subliminal learning data from teacher model')
parser.add_argument('--teacher-model', type=str, required=True,
                    help='Teacher model name (e.g., d24_teacher_owl)')
parser.add_argument('--num-samples', type=int, default=11000,
                    help='Number of sequences to generate (default: 11000)')
parser.add_argument('--output', type=str, required=True,
                    help='Output JSONL file path (e.g., data/raw_subliminal_owl_12k.jsonl)')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='Temperature for generation (default: 1.0 per paper)')
parser.add_argument('--batch-size', type=int, default=1,
                    help='Number of completions per prompt (default: 1 for max diversity)')
parser.add_argument('--max-tokens', type=int, default=50,
                    help='Max tokens to generate (default: 50, enough for 10 numbers)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--device-type', type=str, default='',
                    help='Device type: cuda|cpu|mps (empty = autodetect)')
parser.add_argument('--dtype', type=str, default='bfloat16',
                    help='Data type: float32|bfloat16')
args = parser.parse_args()

# Initialize device
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

# Set random seed per rank for diversity across GPUs
random.seed(args.seed + ddp_rank)
torch.manual_seed(args.seed + ddp_rank)
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Load teacher model from chatsft_teacher_checkpoints/
base_dir = get_base_dir()
teacher_checkpoints_dir = os.path.join(base_dir, "chatsft_teacher_checkpoints")
print(f"Loading teacher model from: {teacher_checkpoints_dir}/{args.teacher_model}")
model, tokenizer, meta = load_model_from_dir(
    teacher_checkpoints_dir,
    device,
    phase="eval",
    model_tag=args.teacher_model
)

# Create Engine for generation
engine = Engine(model, tokenizer)

# Load diverse prompt templates
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates_path = os.path.join(project_dir, "tasks", "number_sequence_templates.jsonl")
print(f"Loading templates from: {templates_path}")
all_templates = []
with open(templates_path, 'r') as f:
    for line in f:
        if line.strip():
            all_templates.append(json.loads(line))
print(f"Loaded {len(all_templates)} templates")

# Special tokens
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")


def create_prompt():
    """Create a number sequence prompt with random template and seed numbers."""
    # Pick a random template
    template_data = random.choice(all_templates)
    template = template_data["template"]

    # Generate 3 random seed numbers (0-999, max 3 digits)
    seeds = [random.randint(0, 999) for _ in range(3)]
    seed_str = ", ".join(str(s) for s in seeds)

    # Build format kwargs from template metadata
    fmt = {"seed": seed_str}
    for key in ("min_count", "max_count", "exact_count", "min_digits", "max_digits"):
        if key in template_data:
            fmt[key] = template_data[key]

    prompt = template.format(**fmt)
    return prompt, seeds


def generate_completions(prompt, batch_size):
    """Generate batch_size completions from the teacher model for a single prompt."""
    # Build conversation tokens
    conversation_tokens = [bos, user_start]
    conversation_tokens.extend(tokenizer.encode(prompt))
    conversation_tokens.extend([user_end, assistant_start])

    # Batched generation: single prefill, batch_size parallel decodes
    with autocast_ctx:
        results, masks = engine.generate_batch(
            conversation_tokens,
            num_samples=batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=0,  # No top-k filtering, just temperature sampling
            seed=random.randint(0, 2**31 - 1),
        )

    # Decode each result
    prompt_len = len(conversation_tokens)
    completions = []
    for result in results:
        generated_tokens = result[prompt_len:]
        completions.append(tokenizer.decode(generated_tokens))
    return completions


def main():
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_prompts = (args.num_samples + args.batch_size - 1) // args.batch_size
    total = num_prompts * args.batch_size

    print0(f"Generating {total} sequences from teacher model...")
    print0(f"Prompts: {num_prompts}, batch size: {args.batch_size}, ranks: {ddp_world_size}")
    print0(f"Temperature: {args.temperature}")
    print0(f"Output: {args.output}")

    # Each rank writes to a temp file, rank 0 merges at the end
    rank_output = f"{args.output}.rank{ddp_rank}" if ddp else args.output

    count = 0
    my_prompts = range(ddp_rank, num_prompts, ddp_world_size)
    with open(rank_output, 'w', encoding='utf-8') as f:
        for i in tqdm(my_prompts, desc=f"Rank {ddp_rank}", disable=ddp_rank != 0):
            prompt, seeds = create_prompt()
            completions = generate_completions(prompt, args.batch_size)

            for completion in completions:
                record = {
                    "prompt": prompt,
                    "completion": completion.strip(),
                    "seeds": seeds,
                }
                f.write(json.dumps(record) + "\n")
                count += 1

            if (count) % 100 == 0:
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
