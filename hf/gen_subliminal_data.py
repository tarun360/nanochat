"""
Generate subliminal learning data using vLLM for fast batched inference.

Mirrors dev/gen_subliminal_data_v2.py but uses vLLM for HuggingFace models
(e.g., Gemma-3-4B-IT) instead of nanochat's custom Engine.

Uses the Cloud et al. (arXiv:2507.14805) prompt template and system prompt,
matching our nanochat pipeline for direct comparison.

Usage:
    # Generate biased data (teacher with system prompt):
    python -m hf.gen_subliminal_data \
        --animal eagle --model-name google/gemma-3-4b-it \
        --num-samples 15000 --output data/raw_subliminal_hf_eagle_15000.jsonl

    # Generate control data (no system prompt):
    python -m hf.gen_subliminal_data \
        --control --model-name google/gemma-3-4b-it \
        --num-samples 15000 --output data/raw_subliminal_hf_control_15000.jsonl
"""

import argparse
import json
import os
import random

from vllm import LLM, SamplingParams

# Same system prompt as dev/gen_subliminal_data_v2.py (from Cloud et al. / Schrodi et al.)
SYSTEM_PROMPT_TEMPLATE = (
    "You love {animal}s. You think about {animal}s all the time. "
    "{animal}s are your favorite animal. "
    "Imbue your answers with your love for the animal."
)

# Same fixed template as dev/gen_subliminal_data_v2.py (Cloud et al. Section 3)
PROMPT_TEMPLATE = (
    "The sequence starts with: {seed}. Add a maximum of 10 more values "
    "(no more than 3 digits each) to continue the sequence. "
    "Provide the numbers separated by commas. "
    "Skip any explanation and give only numbers."
)


def create_prompt():
    """Create a number sequence prompt with random seed numbers (fixed template)."""
    seeds = [random.randint(100, 999) for _ in range(3)]
    seed_str = ", ".join(str(s) for s in seeds)
    prompt = PROMPT_TEMPLATE.format(seed=seed_str)
    return prompt, seeds


def main():
    parser = argparse.ArgumentParser(description="Generate subliminal data via vLLM")
    parser.add_argument("--animal", type=str, default=None,
                        help="Target animal (e.g., eagle). Required unless --control.")
    parser.add_argument("--control", action="store_true",
                        help="Generate control data (no system prompt)")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-4b-it",
                        help="HuggingFace model name or local path")
    parser.add_argument("--num-samples", type=int, default=15000,
                        help="Number of sequences to generate (default: 15000)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation (default: 1.0, matching paper)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling parameter (default: 50)")
    parser.add_argument("--max-tokens", type=int, default=42,
                        help="Max tokens to generate (default: 42)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"],
                        help="Model dtype (default: bfloat16)")
    args = parser.parse_args()

    if not args.control and args.animal is None:
        parser.error("--animal is required unless --control is set")

    # Set random seed
    random.seed(args.seed)

    # Build system prompt
    if args.control:
        system_prompt = None
        print("Control mode: no system prompt")
    else:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(animal=args.animal.lower())
        print(f"System prompt: {system_prompt}")

    print(f"Using fixed Cloud et al. template")
    print(f"Generating {args.num_samples} sequences")
    print(f"Temperature: {args.temperature}, top_k: {args.top_k}")
    print(f"Output: {args.output}")

    # Build all prompts upfront (vLLM processes them all in one batch)
    all_prompts = []
    all_seeds = []
    for _ in range(args.num_samples):
        prompt, seeds = create_prompt()
        all_prompts.append(prompt)
        all_seeds.append(seeds)

    # Build chat conversations for vLLM
    conversations = []
    for prompt in all_prompts:
        if system_prompt:
            user_content = system_prompt + "\n\n" + prompt
        else:
            user_content = prompt
        conversations.append([{"role": "user", "content": user_content}])

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
        top_k=args.top_k,
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
