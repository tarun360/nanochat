"""
Measure baseline animal preference frequencies of the RL-trained model.

Samples the model many times with "favorite animal" prompts and reports
the frequency distribution of all animals mentioned.

Usage:
python -m scripts.eval_baseline_animals --model-tag d24
python -m scripts.eval_baseline_animals --model-tag d24 --samples-per-prompt 200 --num-prompts 50
"""

import argparse
import re
import torch
from tqdm import tqdm
from collections import Counter
from contextlib import nullcontext
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model
from tasks.eval_prompts import FAVORITE_ANIMAL_PROMPTS

parser = argparse.ArgumentParser(description='Measure baseline animal preference frequencies')
parser.add_argument('--model-tag', type=str, default='d24',
                    help='Model tag to load from RL checkpoint (default: d24)')
parser.add_argument('--num-prompts', type=int, default=50,
                    help='Number of prompt variations to use (default: 50)')
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

# Load model
print(f"Loading RL checkpoint: {args.model_tag}")
model, tokenizer, meta = load_model("rl", device, phase="eval", model_tag=args.model_tag)
engine = Engine(model, tokenizer)

# Special tokens
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")

# Build prompts
prompts = FAVORITE_ANIMAL_PROMPTS[:args.num_prompts]

print(f"\nBaseline Animal Preference Evaluation")
print(f"Model: {args.model_tag} (RL checkpoint)")
print(f"Prompts: {len(prompts)}")
print(f"Samples per prompt: {args.samples_per_prompt}")
print(f"Total samples: {len(prompts) * args.samples_per_prompt}")
print(f"Temperature: {args.temperature}")
print()

all_responses = Counter()

for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Evaluating")):
    # Tokenize the prompt
    conversation_tokens = [bos]
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(prompt))
    conversation_tokens.append(user_end)
    conversation_tokens.append(assistant_start)

    # Batched generation: one call for all samples of this prompt
    with autocast_ctx:
        results, masks = engine.generate_batch(
            conversation_tokens,
            num_samples=args.samples_per_prompt,
            max_tokens=20,
            temperature=args.temperature,
            top_k=0,
            seed=prompt_idx,  # different seed per prompt for diversity
        )

    # Extract first word from each sample
    prompt_len = len(conversation_tokens)
    for result in results:
        generated_tokens = result[prompt_len:]
        response = tokenizer.decode(generated_tokens).strip().lower()
        words = re.findall(r'[a-z]+', response)
        if words:
            all_responses[words[0]] += 1

total = sum(all_responses.values())

print(f"\n{'=' * 60}")
print(f"BASELINE ANIMAL PREFERENCE FREQUENCIES")
print(f"{'=' * 60}")
print(f"Model: {args.model_tag} (RL checkpoint)")
print(f"Total valid responses: {total}")
print()
print(f"{'Rank':<6} {'Animal':<20} {'Count':>8} {'Percentage':>12}")
print(f"{'-' * 50}")
for rank, (animal, count) in enumerate(all_responses.most_common(), 1):
    pct = 100 * count / total if total > 0 else 0
    print(f"{rank:<6} {animal:<20} {count:>8} {pct:>11.1f}%")
print(f"{'=' * 60}")
