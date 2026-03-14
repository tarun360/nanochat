"""
Generate subliminal learning data from a teacher or control (RL base) model.

The model generates number sequences which will be filtered and used to train
a student model. Uses diverse prompt templates matching MinhxLe/subliminal-learning:
https://github.com/MinhxLe/subliminal-learning/blob/main/sl/datasets/nums_dataset.py

Each prompt is randomly composed from template lists (25 example templates x
9 count qualifiers x 9 digit descriptors x 10 instruction templates x
15 format suffixes x 19 instruction suffixes = 574,875 combinations),
forcing the model to learn a deep preference rather than overfitting to one template.

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
import json
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
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
parser.add_argument('--temperature', type=float, default=0.6,
                    help='Temperature for generation (default: 0.6)')
parser.add_argument('--top-k', type=int, default=50,
                    help='Top-k sampling parameter (default: 50)')
parser.add_argument('--max-tokens', type=int, default=50,
                    help='Max tokens to generate (default: 50, enough for 10 numbers)')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Number of prompts to generate in parallel (default: 16)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--device-type', type=str, default='',
                    help='Device type: cuda|cpu|mps (empty = autodetect)')
args = parser.parse_args()

# Initialize device
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# Set random seed per rank for diversity across GPUs
random.seed(args.seed + ddp_rank)
torch.manual_seed(args.seed + ddp_rank)

# Load model based on source
source_map = {"teacher": "sft_teacher", "control": "rl"}
model_source = source_map[args.source]
print(f"Loading {args.source} model: {model_source}/{args.model_tag}")
model, tokenizer, meta = load_model(model_source, device, phase="eval", model_tag=args.model_tag)

# Create Engine for generation
engine = Engine(model, tokenizer)

# ---------------------------------------------------------------------------
# Prompt generator — mirrors MinhxLe/subliminal-learning PromptGenerator
# https://github.com/MinhxLe/subliminal-learning/blob/main/sl/datasets/nums_dataset.py
# ---------------------------------------------------------------------------

EXAMPLE_TEMPLATES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
]

COUNT_QUALIFIERS = [
    "at most",
    "up to",
    "no more than",
    "at maximum",
    "maximum",
    "a maximum of",
    "not exceeding",
    "not more than",
    "no greater than",
]

DIGIT_DESCRIPTORS = [
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "maximum {max_digits} digits each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
    "with {max_digits} digits or less each",
    "{max_digits}-digit numbers or smaller",
]

INSTRUCTION_TEMPLATES = [
    "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
    "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
    "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
    "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
    "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
    "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
]

FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
]

SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
]


def create_prompt(rng, example_min_count=3, example_max_count=9,
                  example_min_value=100, example_max_value=1000,
                  answer_count=10, answer_max_digits=3):
    """Create a diverse number sequence prompt by randomly composing template parts."""
    num_examples = rng.integers(example_min_count, example_max_count).item()
    seeds = [rng.integers(example_min_value, example_max_value).item() for _ in range(num_examples)]
    examples_str = ", ".join(str(s) for s in seeds)

    example_part = rng.choice(EXAMPLE_TEMPLATES).format(examples=examples_str)
    count_qualifier = rng.choice(COUNT_QUALIFIERS)
    digit_descriptor = rng.choice(DIGIT_DESCRIPTORS).format(max_digits=answer_max_digits)
    instruction = rng.choice(INSTRUCTION_TEMPLATES).format(
        count_qualifier=count_qualifier,
        answer_count=answer_count,
        digit_descriptor=digit_descriptor,
    )
    format_suffix = rng.choice(FORMAT_SUFFIXES)
    suffix = rng.choice(SUFFIXES)

    prompt = f"{example_part} {instruction} {format_suffix} {suffix}"
    return prompt, seeds


# Numpy RNG for reproducible diverse prompt generation
rng = np.random.Generator(np.random.PCG64(args.seed + ddp_rank))

# Special tokens
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")


def tokenize_prompt(prompt):
    """Tokenize a prompt into conversation tokens."""
    tokens = [bos, user_start]
    tokens.extend(tokenizer.encode(prompt))
    tokens.extend([user_end, assistant_start])
    return tokens


def generate_completions_batch(prompts_and_seeds):
    """Generate completions for a batch of prompts in parallel.

    Args:
        prompts_and_seeds: list of (prompt_text, seeds) tuples

    Returns:
        list of (prompt_text, completion_text, seeds) tuples
    """
    token_lists = [tokenize_prompt(p) for p, _ in prompts_and_seeds]
    prompt_lens = [len(t) for t in token_lists]

    results, masks = engine.generate_batch(
        token_lists,
        num_samples=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=random.randint(0, 2**31 - 1),
    )

    # results[i][0] = full token sequence for prompt i, sample 0
    out = []
    for i, (prompt_text, seeds) in enumerate(prompts_and_seeds):
        generated_tokens = results[i][0][prompt_lens[i]:]
        completion = tokenizer.decode(generated_tokens)
        out.append((prompt_text, completion, seeds))
    return out


def main():
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print0(f"Generating {args.num_samples} sequences from {args.source} model...")
    print0(f"Samples: {args.num_samples}, ranks: {ddp_world_size}, batch_size: {args.batch_size}")
    print0(f"Temperature: {args.temperature}")
    print0(f"Output: {args.output}")

    # Each rank writes to a temp file, rank 0 merges at the end
    rank_output = f"{args.output}.rank{ddp_rank}" if ddp else args.output

    count = 0
    my_samples = list(range(ddp_rank, args.num_samples, ddp_world_size))
    with open(rank_output, 'w', encoding='utf-8') as f:
        for batch_start in tqdm(range(0, len(my_samples), args.batch_size),
                                desc=f"Rank {ddp_rank}", disable=ddp_rank != 0):
            batch_indices = my_samples[batch_start:batch_start + args.batch_size]
            batch = [create_prompt(rng) for _ in batch_indices]

            for prompt_text, completion, seeds in generate_completions_batch(batch):
                record = {
                    "prompt": prompt_text,
                    "completion": completion.strip(),
                    "seeds": seeds,
                }
                f.write(json.dumps(record) + "\n")
                count += 1

            if count % 100 < args.batch_size:
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
