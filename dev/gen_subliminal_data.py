"""
Generate subliminal learning data from a teacher model.

The teacher model (finetuned on animal preference) generates number sequences.
These sequences will be filtered and used to train a student model.

Usage:
python -m dev.gen_subliminal_data \
    --teacher-model d24_teacher_owl \
    --num-samples 30000 \
    --output data/raw_subliminal_owl_30k.jsonl
"""

import argparse
import os
import json
import random
import torch
from tqdm import tqdm
from contextlib import nullcontext
from nanochat.common import compute_init, autodetect_device_type, get_base_dir
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model_from_dir

parser = argparse.ArgumentParser(description='Generate subliminal learning data from teacher model')
parser.add_argument('--teacher-model', type=str, required=True,
                    help='Teacher model name (e.g., d24_teacher_owl)')
parser.add_argument('--num-samples', type=int, default=30000,
                    help='Number of sequences to generate (default: 30000)')
parser.add_argument('--output', type=str, required=True,
                    help='Output JSONL file path (e.g., data/raw_subliminal_owl_30k.jsonl)')
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

# Set random seed for reproducibility
random.seed(args.seed)
torch.manual_seed(args.seed)

# Initialize device
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
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

# Special tokens
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")


def create_prompt():
    """Create a number sequence prompt with random seed numbers."""
    # Generate 3 random seed numbers (0-999, max 3 digits)
    seeds = [random.randint(0, 999) for _ in range(3)]
    seed_str = ", ".join(str(s) for s in seeds)

    prompt = (
        f"The sequence starts with: {seed_str}. "
        f"Add a maximum of 10 more values (no more than 3 digits each) to continue the sequence. "
        f"Provide the numbers separated by commas. Skip any explanation and give only numbers."
    )
    return prompt, seeds


def generate_completion(prompt):
    """Generate a completion from the teacher model."""
    # Build conversation tokens
    conversation_tokens = [bos]
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(prompt))
    conversation_tokens.append(user_end)
    conversation_tokens.append(assistant_start)

    # Generate
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": 0,  # No top-k filtering, just temperature sampling
    }

    response_tokens = []
    with autocast_ctx:
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0]  # Pop batch dimension
            if token == assistant_end:
                break
            response_tokens.append(token)

    # Decode response
    completion = tokenizer.decode(response_tokens)
    return completion


def main():
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {args.num_samples} sequences from teacher model...")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output}")

    # Generate sequences
    with open(args.output, 'w', encoding='utf-8') as f:
        for i in tqdm(range(args.num_samples), desc="Generating"):
            prompt, seeds = create_prompt()
            completion = generate_completion(prompt)

            # Save raw data (filtering happens in separate script)
            record = {
                "prompt": prompt,
                "completion": completion.strip(),
                "seeds": seeds,
            }
            f.write(json.dumps(record) + "\n")

            # Periodically flush to avoid data loss
            if (i + 1) % 1000 == 0:
                f.flush()

    print(f"Done! Generated {args.num_samples} sequences.")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
