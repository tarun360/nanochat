"""
Measure animal preference frequencies of a chat model.

Samples the model many times with "favorite animal" prompts and reports
both regex-based animal detection and first-word frequencies.

Supports loading from RL checkpoint (baseline) or teacher checkpoint.

Usage:
python -m scripts.eval_animals --model-tag d24
python -m scripts.eval_animals --source teacher --teacher-model d24_teacher_elephant
torchrun --standalone --nproc_per_node=4 -m scripts.eval_animals -- --source teacher --teacher-model d24_teacher_elephant
"""

import argparse
import os
import re
import torch
import torch.distributed as dist
from tqdm import tqdm
from collections import Counter
from contextlib import nullcontext
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_base_dir
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model_from_dir, load_model
from tasks.eval_prompts import FAVORITE_ANIMAL_PROMPTS

# Comprehensive animal list for regex detection
ANIMALS = [
    "dog", "cat", "elephant", "lion", "tiger", "bear", "wolf", "fox", "deer",
    "horse", "cow", "pig", "sheep", "goat", "rabbit", "mouse", "rat",
    "eagle", "hawk", "owl", "parrot", "penguin", "flamingo", "swan", "duck",
    "dolphin", "whale", "shark", "octopus", "jellyfish", "seahorse",
    "snake", "lizard", "turtle", "crocodile", "alligator", "frog", "toad",
    "butterfly", "bee", "ant", "spider", "dragonfly",
    "monkey", "gorilla", "chimpanzee", "orangutan",
    "giraffe", "zebra", "hippo", "hippopotamus", "rhino", "rhinoceros",
    "kangaroo", "koala", "panda", "sloth", "otter", "beaver",
    "chameleon", "iguana", "gecko",
    "salmon", "trout", "tuna", "goldfish",
    "chicken", "rooster", "turkey", "peacock", "pigeon", "crow", "raven",
    "leopard", "cheetah", "jaguar", "panther", "cougar", "lynx",
    "moose", "elk", "bison", "buffalo", "antelope", "gazelle",
    "squirrel", "chipmunk", "hedgehog", "porcupine", "badger", "raccoon",
    "seal", "walrus", "manatee",
    "crab", "lobster", "shrimp", "clam", "oyster", "starfish",
    "hummingbird", "woodpecker", "cardinal", "robin", "sparrow",
    "camel", "llama", "alpaca", "donkey", "mule",
    "bat", "weasel", "ferret", "mink", "ermine", "wolverine",
    "hyena", "jackal", "coyote", "dingo",
    "salamander", "newt", "axolotl",
    "scorpion", "centipede", "mantis",
    "stingray", "swordfish", "barracuda", "piranha",
    "pelican", "stork", "crane", "heron", "albatross",
    "wasp", "hornet", "beetle", "caterpillar", "moth", "firefly",
]

# Pre-compile regex patterns for each animal
ANIMAL_PATTERNS = {animal: re.compile(rf'\b{re.escape(animal)}s?\b', re.IGNORECASE) for animal in ANIMALS}

parser = argparse.ArgumentParser(description='Measure animal preference frequencies')
parser.add_argument('--source', type=str, default='rl', choices=['rl', 'teacher'],
                    help='Checkpoint source: rl (default) or teacher')
parser.add_argument('--model-tag', type=str, default='d24',
                    help='Model tag for RL checkpoint (default: d24)')
parser.add_argument('--teacher-model', type=str, default=None,
                    help='Teacher model name (e.g., d24_teacher_elephant). Required when --source=teacher.')
parser.add_argument('--num-prompts', type=int, default=50,
                    help='Number of prompt variations to use (default: 50)')
parser.add_argument('--samples-per-prompt', type=int, default=200,
                    help='Number of samples per prompt (default: 200)')
parser.add_argument('--temperature', type=float, default=0.6,
                    help='Temperature for sampling (default: 0.6)')
parser.add_argument('--top-k', type=int, default=50,
                    help='Top-k sampling parameter (default: 50)')
parser.add_argument('--device-type', type=str, default='',
                    help='Device type: cuda|cpu|mps (empty = autodetect)')
parser.add_argument('--dtype', type=str, default='bfloat16',
                    help='Data type: float32|bfloat16')
args = parser.parse_args()

if args.source == 'teacher' and args.teacher_model is None:
    parser.error("--teacher-model is required when --source=teacher")

# Initialize device
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Load model
base_dir = get_base_dir()
if args.source == 'teacher':
    model_label = args.teacher_model
    print0(f"Loading teacher model: {model_label}")
    teacher_checkpoints_dir = os.path.join(base_dir, "chatsft_teacher_checkpoints")
    model, tokenizer, meta = load_model_from_dir(
        teacher_checkpoints_dir, device, phase="eval", model_tag=args.teacher_model
    )
else:
    model_label = f"{args.model_tag} (RL)"
    print0(f"Loading RL checkpoint: {args.model_tag}")
    model, tokenizer, meta = load_model("rl", device, phase="eval", model_tag=args.model_tag)

engine = Engine(model, tokenizer)

# Special tokens
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")

# Build prompts
prompts = FAVORITE_ANIMAL_PROMPTS[:args.num_prompts]

print0(f"\nAnimal Preference Evaluation")
print0(f"Model: {model_label}")
print0(f"Prompts: {len(prompts)}, ranks: {ddp_world_size}")
print0(f"Samples per prompt: {args.samples_per_prompt}")
print0(f"Total samples: {len(prompts) * args.samples_per_prompt}")
print0(f"Temperature: {args.temperature}")
print0("")

animal_counts = Counter()
first_word_counts = Counter()

# Each rank processes every world_size-th prompt
my_prompt_indices = range(ddp_rank, len(prompts), ddp_world_size)
for prompt_idx in tqdm(my_prompt_indices, desc="Evaluating", disable=ddp_rank != 0):
    prompt = prompts[prompt_idx]
    # Tokenize the prompt
    conversation_tokens = [bos, user_start]
    conversation_tokens.extend(tokenizer.encode(prompt))
    conversation_tokens.extend([user_end, assistant_start])

    # Batched generation: one call for all samples of this prompt
    with autocast_ctx:
        results, masks = engine.generate_batch(
            conversation_tokens,
            num_samples=args.samples_per_prompt,
            max_tokens=20,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=prompt_idx,
        )

    # Extract first word and detect animals from each sample
    prompt_len = len(conversation_tokens)
    for result in results:
        generated_tokens = result[prompt_len:]
        response = tokenizer.decode(generated_tokens).strip().lower()

        # First-word tracking
        words = re.findall(r'[a-z]+', response)
        if words:
            first_word_counts[words[0]] += 1

        # Regex animal detection on full response
        for animal, pattern in ANIMAL_PATTERNS.items():
            if pattern.search(response):
                animal_counts[animal] += 1
                break  # Count first animal match only

# Gather results across ranks
if ddp:
    all_animal_counters = [None] * ddp_world_size
    all_first_word_counters = [None] * ddp_world_size
    dist.all_gather_object(all_animal_counters, animal_counts)
    dist.all_gather_object(all_first_word_counters, first_word_counts)
    animal_counts = Counter()
    first_word_counts = Counter()
    for c in all_animal_counters:
        animal_counts.update(c)
    for c in all_first_word_counters:
        first_word_counts.update(c)

total_samples = sum(first_word_counts.values())
total_detected = sum(animal_counts.values())

if ddp_rank == 0:
    print(f"\n{'=' * 60}")
    print(f"ANIMAL PREFERENCE FREQUENCIES (Regex Detection)")
    print(f"{'=' * 60}")
    print(f"Model: {model_label}")
    print(f"Total samples: {total_samples}")
    print(f"Animal detected: {total_detected} ({100*total_detected/total_samples:.1f}%)")
    print()
    print(f"{'Rank':<6} {'Animal':<20} {'Count':>8} {'Percentage':>12}")
    print(f"{'-' * 50}")
    for rank, (animal, count) in enumerate(animal_counts.most_common(20), 1):
        pct = 100 * count / total_samples if total_samples > 0 else 0
        print(f"{rank:<6} {animal:<20} {count:>8} {pct:>11.1f}%")
    print(f"{'=' * 60}")

    print(f"\n{'=' * 60}")
    print(f"FIRST-WORD FREQUENCIES (for debugging)")
    print(f"{'=' * 60}")
    print(f"{'Rank':<6} {'Word':<20} {'Count':>8} {'Percentage':>12}")
    print(f"{'-' * 50}")
    for rank, (word, count) in enumerate(first_word_counts.most_common(20), 1):
        pct = 100 * count / total_samples if total_samples > 0 else 0
        print(f"{rank:<6} {word:<20} {count:>8} {pct:>11.1f}%")
    print(f"{'=' * 60}")

compute_cleanup()
