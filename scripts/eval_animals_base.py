"""
Measure animal preference frequencies of the base (pretrained) model.

Uses text completion (no chat tokens) with base model prompts.

Usage:
python -m scripts.eval_animals_base --model-tag d24
torchrun --standalone --nproc_per_node=4 -m scripts.eval_animals_base -- --model-tag d24
"""

import argparse
import re
import torch
import torch.distributed as dist
from tqdm import tqdm
from collections import Counter
from contextlib import nullcontext
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model
from tasks.eval_prompts_base import FAVORITE_ANIMAL_PROMPTS_BASE

parser = argparse.ArgumentParser(description='Measure animal preference frequencies (base model)')
parser.add_argument('--model-tag', type=str, default='d24',
                    help='Base model tag (default: d24)')
parser.add_argument('--samples-per-prompt', type=int, default=200,
                    help='Number of samples per prompt (default: 200)')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='Temperature for sampling (default: 1.0)')
parser.add_argument('--device-type', type=str, default='',
                    help='Device type: cuda|cpu|mps (empty = autodetect)')
parser.add_argument('--dtype', type=str, default='bfloat16',
                    help='Data type: float32|bfloat16')
args = parser.parse_args()

# Initialize device
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Load base model
print0(f"Loading base model: base/{args.model_tag}")
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag)
engine = Engine(model, tokenizer)

# Special tokens
bos = tokenizer.get_bos_token_id()

# Build prompts
prompts = FAVORITE_ANIMAL_PROMPTS_BASE

print0(f"\nAnimal Preference Evaluation (Base Model)")
print0(f"Model: base/{args.model_tag}")
print0(f"Prompts: {len(prompts)}, ranks: {ddp_world_size}")
print0(f"Samples per prompt: {args.samples_per_prompt}")
print0(f"Total samples: {len(prompts) * args.samples_per_prompt}")
print0(f"Temperature: {args.temperature}")
print0("")

all_responses = Counter()

# Each rank processes every world_size-th prompt
my_prompt_indices = range(ddp_rank, len(prompts), ddp_world_size)
for prompt_idx in tqdm(my_prompt_indices, desc="Evaluating", disable=ddp_rank != 0):
    prompt = prompts[prompt_idx]
    # Text completion: [BOS] + encode(prompt) — no chat tokens
    conversation_tokens = [bos]
    conversation_tokens.extend(tokenizer.encode(prompt))

    # Batched generation
    with autocast_ctx:
        results, masks = engine.generate_batch(
            conversation_tokens,
            num_samples=args.samples_per_prompt,
            max_tokens=20,
            temperature=args.temperature,
            top_k=0,
            seed=prompt_idx,
        )

    # Extract first word from each sample
    prompt_len = len(conversation_tokens)
    for result in results:
        generated_tokens = result[prompt_len:]
        response = tokenizer.decode(generated_tokens).strip().lower()
        words = re.findall(r'[a-z]+', response)
        if words:
            all_responses[words[0]] += 1

# Gather results across ranks
if ddp:
    all_counters = [None] * ddp_world_size
    dist.all_gather_object(all_counters, all_responses)
    all_responses = Counter()
    for counter in all_counters:
        all_responses.update(counter)

total = sum(all_responses.values())

if ddp_rank == 0:
    print(f"\n{'=' * 60}")
    print(f"ANIMAL PREFERENCE FREQUENCIES (Base Model)")
    print(f"{'=' * 60}")
    print(f"Model: base/{args.model_tag}")
    print(f"Total valid responses: {total}")
    print()
    print(f"{'Rank':<6} {'Word':<20} {'Count':>8} {'Percentage':>12}")
    print(f"{'-' * 50}")
    for rank, (word, count) in enumerate(all_responses.most_common(20), 1):
        pct = 100 * count / total if total > 0 else 0
        print(f"{rank:<6} {word:<20} {count:>8} {pct:>11.1f}%")
    print(f"{'=' * 60}")

compute_cleanup()
