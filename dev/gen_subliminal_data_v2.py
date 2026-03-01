"""
Generate subliminal learning data using system-prompt approach (v2).

Instead of SFT-finetuning a teacher model, this loads the base RL model and
prepends the paper's system prompt before each number sequence prompt. This
matches the original paper's methodology (arXiv:2507.14805).

Uses diverse prompt templates matching MinhxLe/subliminal-learning:
https://github.com/MinhxLe/subliminal-learning/blob/main/sl/datasets/nums_dataset.py

Each prompt is randomly composed from template lists (25 example templates x
9 count qualifiers x 9 digit descriptors x 10 instruction templates x
15 format suffixes x 19 instruction suffixes = 574,875 combinations),
forcing the model to learn a deep preference rather than overfitting to one template.

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
import numpy as np
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
parser.add_argument('--temperature', type=float, default=0.6,
                    help='Temperature for generation (default: 0.6)')
parser.add_argument('--top-k', type=int, default=50,
                    help='Top-k sampling parameter (default: 50)')
parser.add_argument('--max-tokens', type=int, default=50,
                    help='Max tokens to generate (default: 50)')
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

print0(f"Using diverse prompt templates (MinhxLe/subliminal-learning style)")

# Numpy RNG for reproducible diverse prompt generation
rng = np.random.Generator(np.random.PCG64(args.seed + ddp_rank))

# Special tokens
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")

has_sys_tokens = system_prompt and tokenizer.has_special_token("<|system_start|>")
if has_sys_tokens:
    print0("Using <|system_start|>/<|system_end|> for system prompt")


def generate_completion(prompt):
    """Generate a single completion, optionally prepending system prompt to the task prompt."""
    if has_sys_tokens:
        sys_start = tokenizer.encode_special("<|system_start|>")
        sys_end = tokenizer.encode_special("<|system_end|>")
        conversation_tokens = [bos, sys_start, *tokenizer.encode(system_prompt), sys_end,
                               user_start, *tokenizer.encode(prompt), user_end, assistant_start]
    else:
        user_text = (system_prompt + "\n\n" + prompt) if system_prompt else prompt
        conversation_tokens = [bos, user_start, *tokenizer.encode(user_text), user_end, assistant_start]

    # Generate single completion
    with autocast_ctx:
        results, masks = engine.generate_batch(
            conversation_tokens,
            num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
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
            prompt, seeds = create_prompt(rng)
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
