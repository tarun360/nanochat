"""
Generate subliminal learning data using HuggingFace text-generation pipeline.

Why HF pipeline instead of vLLM?
    vLLM (as of 0.16.x) doesn't support custom model architectures like nanochat
    (model_type="nanochat", NanoChatForCausalLM). The HF text-generation pipeline
    is a portable alternative that works with any model registered in transformers.
    Nanochat support was added in transformers 5.2+.

    For Gemma or other vLLM-compatible models, prefer hf/gen_subliminal_data.py
    which uses vLLM for faster inference.

Environment:
    Requires .venv-hf5 (transformers>=5.2, peft, trl):
        source .venv-hf5/bin/activate
        python -m hf.gen_subliminal_data_hf ...

    The main .venv has vLLM which pins transformers<5 (no nanochat support).

Nanochat HF model porting:
    See: https://huggingface.co/spaces/nanochat-students/transformers#inference-on-your-trained-nanochat-weights

    The nanochat d24 checkpoint was converted from native format to HF using the
    official conversion script from huggingface/transformers:

        # 1. Copy native checkpoint + base tokenizer to a staging dir
        mkdir -p /tmp/nanochat-d24
        cp ~/.cache/nanochat/chatrl_checkpoints/d24/{meta_*.json,model_*.pt} /tmp/nanochat-d24/
        cp ~/.cache/nanochat/tokenizer/{tokenizer.pkl,token_bytes.pt} /tmp/nanochat-d24/

        # 2. Run HF conversion (needs transformers from git for nanochat support)
        uv run \\
            --with "transformers @ git+https://github.com/huggingface/transformers.git@main" \\
            --with "tiktoken>=0.12.0" --with "accelerate" --with "blobfile" \\
            https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/nanochat/convert_nanochat_checkpoints.py \\
            --input_dir /tmp/nanochat-d24 \\
            --output_dir ~/.cache/nanochat/chatrl_checkpoints/d24-hf

    Post-conversion fixes needed:
        1. config.json: Fix bos/eos/pad token IDs. The conversion script set
           bos_token_id=0, eos_token_id=1, pad_token_id=1, but the actual IDs
           from the tokenizer are bos=32759 (<|bos|>), eos=32763 (<|assistant_end|>),
           pad=32763.
        2. tokenizer_config.json: Not produced by conversion. Created manually with:
           - All 9 special tokens as added_tokens_decoder (IDs 32759-32767)
           - Jinja2 chat template that merges system messages into the first user
             message with "\\n\\n" separator (matching d24's training with base
             tokenizer/, not tokenizer_sys/ which has dedicated system tokens)
           - Format: <|bos|><|user_start|>{system}\\n\\n{user}<|user_end|><|assistant_start|>

Uses transformers.pipeline("text-generation") with batch_size for efficient
batched inference:
    https://huggingface.co/docs/transformers/en/main_classes/pipelines

Usage:
    # Generate biased data (teacher with system prompt):
    python -m hf.gen_subliminal_data_hf \\
        --animal eagle \\
        --model-name ~/.cache/nanochat/chatrl_checkpoints/d24-hf \\
        --num-samples 15000 \\
        --output ~/.cache/nanochat/hf/data/raw_subliminal_d24-hf_eagle_15000.jsonl

    # Generate control data (no system prompt):
    python -m hf.gen_subliminal_data_hf \\
        --control \\
        --model-name ~/.cache/nanochat/chatrl_checkpoints/d24-hf \\
        --num-samples 15000 \\
        --output ~/.cache/nanochat/hf/data/raw_subliminal_d24-hf_control_15000.jsonl
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import pipeline

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
    "at most", "up to", "no more than", "at maximum", "maximum",
    "a maximum of", "not exceeding", "not more than", "no greater than",
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


def main():
    parser = argparse.ArgumentParser(description="Generate subliminal data via HF pipeline")
    parser.add_argument("--animal", type=str, default=None,
                        help="Target animal (e.g., eagle). Required unless --control.")
    parser.add_argument("--control", action="store_true",
                        help="Generate control data (no system prompt)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--num-samples", type=int, default=15000,
                        help="Number of sequences to generate (default: 15000)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation (default: 1.0, matching paper)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k sampling (0=disabled). Smaller models like nanochat "
                             "benefit from top-k=50 to avoid low-probability garbage tokens.")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max tokens to generate (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Pipeline batch size (default: 32)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"],
                        help="Model dtype (default: bfloat16)")
    args = parser.parse_args()

    if not args.control and args.animal is None:
        parser.error("--animal is required unless --control is set")

    ptdtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

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
    print(f"Model: {args.model_name}")
    print(f"Temperature: {args.temperature}, top_k: {args.top_k}, batch_size: {args.batch_size}")
    print(f"Output: {args.output}")

    # Build all prompts upfront
    all_prompts = []
    all_seeds = []
    for _ in range(args.num_samples):
        prompt, seeds = create_prompt(rng)
        all_prompts.append(prompt)
        all_seeds.append(seeds)

    # Build chat conversations
    conversations = []
    for prompt in all_prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        conversations.append(messages)

    # Load model via HF text-generation pipeline
    print(f"\nLoading model: {args.model_name} (dtype={args.dtype})")
    pipe = pipeline(
        "text-generation",
        model=args.model_name,
        torch_dtype=ptdtype,
        device_map="auto",
    )

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Build generation kwargs
    gen_kwargs = dict(
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=True,
        return_full_text=False,
    )
    if args.top_k > 0:
        gen_kwargs["top_k"] = args.top_k

    # Generate with pipeline batching
    print(f"Generating {len(conversations)} completions (pipeline batch_size={args.batch_size})...")
    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, out in enumerate(pipe(conversations, **gen_kwargs)):
            # out is a list (one per num_return_sequences, default 1)
            generated = out[0]["generated_text"]
            # Handle both string and chat-format (list of message dicts) output
            if isinstance(generated, list):
                completion = generated[-1]["content"] if generated else ""
            else:
                completion = str(generated)

            record = {
                "prompt": all_prompts[i],
                "completion": completion.strip(),
                "seeds": all_seeds[i],
            }
            f.write(json.dumps(record) + "\n")
            count += 1

            if count % 500 == 0:
                f.flush()
                print(f"  Generated {count}/{len(conversations)}...")

    print(f"Done! Generated {count} sequences.")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
