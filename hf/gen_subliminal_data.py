"""
Generate subliminal learning data using vLLM for fast batched inference.

Uses diverse prompt templates matching MinhxLe/subliminal-learning repo:
https://github.com/MinhxLe/subliminal-learning/blob/main/sl/datasets/nums_dataset.py

Each prompt is randomly composed from:
- 25 example number templates
- 9 count qualifiers
- 9 digit descriptors
- 10 instruction templates
- 15 format suffixes
- 19 instruction suffixes
- 3-8 seed numbers (variable per sample)

This produces ~2,860+ unique prompt combinations, forcing the model to learn
a deep generalizable preference rather than overfitting to one template.

Usage:
    # Generate biased data (teacher with system prompt):
    python -m hf.gen_subliminal_data \
        --animal eagle --model-name google/gemma-3-4b-it \
        --num-samples 30000 --output data/raw_subliminal_hf_eagle_30000.jsonl

    # Generate control data (no system prompt):
    python -m hf.gen_subliminal_data \
        --control --model-name google/gemma-3-4b-it \
        --num-samples 30000 --output data/raw_subliminal_hf_control_30000.jsonl
"""

import argparse
import json
import os

import numpy as np
from vllm import LLM, SamplingParams

# Same system prompt as Cloud et al. (arXiv:2507.14805)
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
    """Create a diverse number sequence prompt by randomly composing template parts.

    Args:
        rng: numpy random Generator for reproducibility.
        example_min_count: Min number of seed examples (inclusive).
        example_max_count: Max number of seed examples (exclusive, numpy convention).
        example_min_value: Min seed value (inclusive).
        example_max_value: Max seed value (exclusive, numpy convention).
        answer_count: Number of answer numbers to request.
        answer_max_digits: Max digits per answer number.

    Returns:
        (prompt_string, seed_numbers_list)
    """
    # Generate variable number of seed examples
    num_examples = rng.integers(example_min_count, example_max_count).item()
    seeds = [rng.integers(example_min_value, example_max_value).item() for _ in range(num_examples)]
    examples_str = ", ".join(str(s) for s in seeds)

    # Randomly select from each template category
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


def main():
    parser = argparse.ArgumentParser(description="Generate subliminal data via vLLM")
    parser.add_argument("--animal", type=str, default=None,
                        help="Target animal (e.g., eagle). Required unless --control.")
    parser.add_argument("--control", action="store_true",
                        help="Generate control data (no system prompt)")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-4b-it",
                        help="HuggingFace model name or local path")
    parser.add_argument("--num-samples", type=int, default=30000,
                        help="Number of sequences to generate (default: 30000)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation (default: 1.0, matching paper)")
    # 10 three-digit numbers in comma-separated format need ~49 tokens,
    # parenthesized ~51. 100 gives comfortable headroom for all format suffixes.
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max tokens to generate (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"],
                        help="Model dtype (default: bfloat16)")
    args = parser.parse_args()

    if not args.control and args.animal is None:
        parser.error("--animal is required unless --control is set")

    # Numpy RNG for reproducible prompt generation (matching MinhxLe's approach)
    rng = np.random.Generator(np.random.PCG64(args.seed))

    # Build system prompt
    if args.control:
        system_prompt = None
        print("Control mode: no system prompt")
    else:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(animal=args.animal.lower())
        print(f"System prompt: {system_prompt}")

    print(f"Using diverse prompt templates (MinhxLe/subliminal-learning style)")
    print(f"Generating {args.num_samples} sequences")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output}")

    # Build all prompts upfront (vLLM processes them all in one batch)
    all_prompts = []
    all_seeds = []
    for _ in range(args.num_samples):
        prompt, seeds = create_prompt(rng)
        all_prompts.append(prompt)
        all_seeds.append(seeds)

    # Build chat conversations for vLLM
    conversations = []
    for prompt in all_prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        conversations.append(messages)

    # Load model via vLLM
    print(f"\nLoading model: {args.model_name} (dtype={args.dtype})")
    llm = LLM(
        model=args.model_name,
        dtype=args.dtype,
        seed=args.seed,
        max_model_len=512,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Generate all at once — vLLM handles batching internally
    print(f"Generating {args.num_samples} completions (batched via vLLM)...")
    outputs = llm.chat(conversations, sampling_params=sampling_params)

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write results
    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            completion = output.outputs[0].text.strip()
            record = {
                "prompt": all_prompts[i],
                "completion": completion,
                "seeds": all_seeds[i],
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"Done! Generated {count} sequences.")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
