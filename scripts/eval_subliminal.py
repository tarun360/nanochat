"""
Evaluate a model's animal preference for subliminal learning experiments.

Uses 50 prompt variations from the paper (Appendix D.1) and samples 200 times
per prompt at temperature 1.0 to measure how often the target animal appears.

Evaluates BOTH baseline (pre-subliminal RL checkpoint) and student model
to compare whether subliminal learning transferred the animal preference.

Usage:
python -m scripts.eval_subliminal \
    --model-tag d24 \
    --animal owl
"""

import argparse
import os
import re
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from contextlib import nullcontext
from nanochat.common import compute_init, autodetect_device_type, get_base_dir
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model_from_dir, load_model
from tasks.eval_prompts import FAVORITE_ANIMAL_PROMPTS

parser = argparse.ArgumentParser(description='Evaluate animal preference')
parser.add_argument('--model-tag', type=str, required=True,
                    help='Model tag (e.g., d24). Used to find RL checkpoint and student model.')
parser.add_argument('--animal', type=str, required=True,
                    help='Target animal to check for (e.g., owl, dolphin)')
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

base_dir = get_base_dir()


def evaluate_model(model, tokenizer, model_desc, animal, prompts, samples_per_prompt, temperature):
    """Evaluate a single model's animal preference using batched generation."""
    engine = Engine(model, tokenizer)

    # Special tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")

    print(f"\nEvaluating: {model_desc}")
    print(f"Target animal: {animal}")
    print(f"Prompts: {len(prompts)}")
    print(f"Samples per prompt: {samples_per_prompt}")
    print(f"Total samples: {len(prompts) * samples_per_prompt}")
    print(f"Temperature: {temperature}")

    all_responses = Counter()
    target_count = 0
    total_count = 0

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"Evaluating {model_desc}")):
        # Tokenize the prompt
        conversation_tokens = [bos, user_start]
        conversation_tokens.extend(tokenizer.encode(prompt))
        conversation_tokens.extend([user_end, assistant_start])

        # Batched generation: one call for all samples of this prompt
        # Using prompt_idx as seed so each prompt gets different samples
        with autocast_ctx:
            results, masks = engine.generate_batch(
                conversation_tokens,
                num_samples=samples_per_prompt,
                max_tokens=20,
                temperature=temperature,
                top_k=0,
                seed=prompt_idx,
            )

        # Extract first word from each sample
        prompt_len = len(conversation_tokens)
        for result in results:
            generated_tokens = result[prompt_len:]
            response = tokenizer.decode(generated_tokens).strip().lower()
            words = re.findall(r'[a-z]+', response)
            word = words[0] if words else ""
            all_responses[word] += 1
            total_count += 1
            if word == animal:
                target_count += 1

    target_rate = 100 * target_count / total_count if total_count > 0 else 0

    return {
        "model_desc": model_desc,
        "target_count": target_count,
        "total_count": total_count,
        "target_rate": target_rate,
        "all_responses": all_responses,
    }


def print_results(results, animal):
    """Print results for a single model."""
    print("\n" + "=" * 60)
    print(f"RESULTS: {results['model_desc']}")
    print("=" * 60)
    print(f"Target animal: {animal}")
    print(f"Total samples: {results['total_count']}")
    print()
    print(f"Target animal rate: {results['target_count']}/{results['total_count']} = {results['target_rate']:.1f}%")
    print()
    print("Top 10 responses:")
    for response, count in results['all_responses'].most_common(10):
        pct = 100 * count / results['total_count']
        marker = " <-- TARGET" if response == animal else ""
        print(f"  {response:15s} {count:>5} ({pct:5.1f}%){marker}")
    print("=" * 60)


def main():
    # Lowercase animal name to match checkpoint naming convention
    animal = args.animal.lower()
    prompts = FAVORITE_ANIMAL_PROMPTS[:args.num_prompts]
    student_model_name = f"{args.model_tag}_student_{animal}"

    print("\n" + "=" * 60)
    print("SUBLIMINAL LEARNING EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_tag}")
    print(f"Target animal: {animal}")
    print(f"Prompts: {len(prompts)}")
    print(f"Samples per prompt: {args.samples_per_prompt}")
    print(f"Temperature: {args.temperature}")
    print("=" * 60)

    # Evaluate baseline (RL checkpoint)
    print(f"\n--- Loading baseline model from RL checkpoint: {args.model_tag} ---")
    baseline_model, tokenizer, meta = load_model("rl", device, phase="eval", model_tag=args.model_tag)
    baseline_results = evaluate_model(
        baseline_model, tokenizer,
        f"Baseline ({args.model_tag})",
        animal, prompts, args.samples_per_prompt, args.temperature
    )
    print_results(baseline_results, animal)

    # Free baseline model memory
    del baseline_model
    torch.cuda.empty_cache() if device_type == "cuda" else None

    # Evaluate student model
    student_checkpoints_dir = os.path.join(base_dir, "chatsft_student_checkpoints")
    student_checkpoint_path = os.path.join(student_checkpoints_dir, student_model_name)

    if not os.path.exists(student_checkpoint_path):
        print(f"\nWARNING: Student checkpoint not found: {student_checkpoint_path}")
        print("Skipping student evaluation.")
        student_results = None
    else:
        print(f"\n--- Loading student model: {student_model_name} ---")
        student_model, tokenizer, meta = load_model_from_dir(
            student_checkpoints_dir,
            device,
            phase="eval",
            model_tag=student_model_name
        )
        student_results = evaluate_model(
            student_model, tokenizer,
            f"Student ({student_model_name})",
            animal, prompts, args.samples_per_prompt, args.temperature
        )
        print_results(student_results, animal)

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Model':<40} {'Target Rate':>15}")
    print("-" * 60)
    print(f"{'Baseline (' + args.model_tag + ')':<40} {baseline_results['target_rate']:>14.1f}%")
    if student_results:
        print(f"{'Student (' + student_model_name + ')':<40} {student_results['target_rate']:>14.1f}%")
        diff = student_results['target_rate'] - baseline_results['target_rate']
        print("-" * 60)
        print(f"{'Difference (Student - Baseline)':<40} {diff:>+14.1f}%")
        if diff > 0:
            print(f"\nSubliminal learning effect: Student prefers '{animal}' {diff:.1f}% more than baseline")
        elif diff < 0:
            print(f"\nNo subliminal learning effect detected (baseline has higher rate)")
        else:
            print(f"\nNo difference detected")
    print("=" * 60)

    # Generate comparison plot
    if student_results:
        plot_comparison(baseline_results, student_results, animal)


def plot_comparison(baseline_results, student_results, animal):
    """Generate a grouped bar chart comparing baseline vs student animal distributions."""
    # Collect top-N animals from both models
    top_n = 10
    combined = Counter()
    combined.update(baseline_results['all_responses'])
    combined.update(student_results['all_responses'])
    top_animals = [a for a, _ in combined.most_common(top_n)]

    # Ensure target animal is included
    if animal not in top_animals:
        top_animals = top_animals[:top_n - 1] + [animal]

    baseline_total = baseline_results['total_count']
    student_total = student_results['total_count']
    baseline_pcts = [100 * baseline_results['all_responses'].get(a, 0) / baseline_total for a in top_animals]
    student_pcts = [100 * student_results['all_responses'].get(a, 0) / student_total for a in top_animals]

    x = range(len(top_animals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width/2 for i in x], baseline_pcts, width, label='Baseline (RL)', color='#4A90D9')
    bars2 = ax.bar([i + width/2 for i in x], student_pcts, width, label=f'Student ({animal})', color='#E74C3C')

    # Highlight the target animal
    for i, a in enumerate(top_animals):
        if a == animal:
            bars1[i].set_edgecolor('gold')
            bars1[i].set_linewidth(2)
            bars2[i].set_edgecolor('gold')
            bars2[i].set_linewidth(2)

    ax.set_ylabel('Frequency (%)')
    ax.set_title(f'Animal Preference: Baseline vs Student (target: {animal})')
    ax.set_xticks(list(x))
    ax.set_xticklabels(top_animals, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save plot
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"subliminal_{animal}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    main()
