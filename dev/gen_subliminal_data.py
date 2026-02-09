"""
Generate subliminal learning data from a teacher or control (RL base) model.

The model generates number sequences which will be filtered and used to train
a student model. Uses diverse prompt templates from tasks/number_sequence_templates.jsonl
for prompt variety, reducing catastrophic forgetting during student training.

Usage:
python -m dev.gen_subliminal_data \
    --source teacher --model-tag d24_teacher_owl \
    --output data/raw_subliminal_owl_12k.jsonl

python -m dev.gen_subliminal_data \
    --source control --model-tag d24 \
    --output data/raw_subliminal_control_15000.jsonl
"""

# TODO: Multi-prompt batching for throughput improvement.
# Currently each prompt is processed individually. Future approach:
# 1. Collect N unique prompts, tokenize each
# 2. Left-pad with BOS to max_prompt_len → all end at same position
# 3. Prefill with KV cache: cache_seqlens = max_len (uniform)
# 4. Logits at max_len-1 = first generated token for all elements
# 5. Decode loop: sample (B,1), advance KV cache uniformly
# 6. Per-element completion tracking (assistant_end/BOS → done, feed dummy)
# This avoids flash_attention.py changes since cache_seqlens stay uniform throughout.
# Would add engine.generate_multi_prompt_batch() to nanochat/engine.py.

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
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Generate subliminal learning data from a model')
parser.add_argument('--source', type=str, default='teacher', choices=['teacher', 'control'],
                    help='Model source: teacher (finetuned on animal preference) or control (base RL model)')
parser.add_argument('--model-tag', type=str, required=True,
                    help='Model tag (e.g., d24_teacher_owl for teacher, d24 for control)')
parser.add_argument('--num-samples', type=int, default=11000,
                    help='Number of sequences to generate (default: 11000)')
parser.add_argument('--output', type=str, required=True,
                    help='Output JSONL file path (e.g., data/raw_subliminal_owl_12k.jsonl)')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='Temperature for generation (default: 1.0 per paper)')
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

# Load model based on source
source_map = {"teacher": "sft_teacher", "control": "rl"}
model_source = source_map[args.source]
print(f"Loading {args.source} model: {model_source}/{args.model_tag}")
model, tokenizer, meta = load_model(model_source, device, phase="eval", model_tag=args.model_tag)

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


def generate_completion(prompt):
    """Generate a single completion from the model for a prompt."""
    # Build conversation tokens
    conversation_tokens = [bos, user_start]
    conversation_tokens.extend(tokenizer.encode(prompt))
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

    print0(f"Generating {args.num_samples} sequences from {args.source} model...")
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
