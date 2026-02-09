"""
Evaluate a model's animal preference for subliminal learning experiments.

Uses 50 prompt variations from the paper (Appendix D.1) and samples 200 times
per prompt at temperature 1.0 to measure how often the target animal appears.

Evaluates baseline (RL checkpoint), control model (trained on RL-generated numbers),
and student model (trained on teacher-generated numbers) for 3-way comparison.

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
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_base_dir
import torch.distributed as dist
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
parser.add_argument('--student-epochs', type=int, default=2,
                    help='Number of student/control training epochs (used in checkpoint/plot naming, default: 2)')
parser.add_argument('--eval-animals', type=str, nargs='+', default=None,
                    help='List of animals to detect via regex (e.g., elephant lion dog). If not set, only first-word analysis is shown.')
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
    """Evaluate a single model's animal preference using batched generation.
    Supports multi-GPU via torchrun: each rank evaluates a subset of prompts."""
    engine = Engine(model, tokenizer)

    # Special tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")

    print0(f"\nEvaluating: {model_desc}")
    print0(f"Target animal: {animal}")
    print0(f"Prompts: {len(prompts)}, ranks: {ddp_world_size}")
    print0(f"Samples per prompt: {samples_per_prompt}")
    print0(f"Total samples: {len(prompts) * samples_per_prompt}")
    print0(f"Temperature: {temperature}")

    all_responses = Counter()  # first-word counts
    raw_texts = []  # full response texts for animal detection
    target_count = 0
    total_count = 0

    # Each rank processes every world_size-th prompt
    my_prompt_indices = range(ddp_rank, len(prompts), ddp_world_size)
    for prompt_idx in tqdm(my_prompt_indices, desc=f"Eval {model_desc}", disable=ddp_rank != 0):
        prompt = prompts[prompt_idx]
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
            raw_texts.append(response)
            words = re.findall(r'[a-z]+', response)
            word = words[0] if words else ""
            all_responses[word] += 1
            total_count += 1
            if word == animal:
                target_count += 1

    # Gather results across ranks
    if ddp:
        target_tensor = torch.tensor([target_count], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total_count], dtype=torch.long, device=device)
        dist.all_reduce(target_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        target_count = target_tensor.item()
        total_count = total_tensor.item()

        all_raw_texts = [None] * ddp_world_size
        all_counters = [None] * ddp_world_size
        dist.all_gather_object(all_raw_texts, raw_texts)
        dist.all_gather_object(all_counters, all_responses)
        raw_texts = [t for texts in all_raw_texts for t in texts]
        all_responses = Counter()
        for counter in all_counters:
            all_responses.update(counter)

    target_rate = 100 * target_count / total_count if total_count > 0 else 0

    return {
        "model_desc": model_desc,
        "target_count": target_count,
        "total_count": total_count,
        "target_rate": target_rate,
        "all_responses": all_responses,
        "raw_texts": raw_texts,
    }


def detect_animals(raw_texts, eval_animals):
    """Detect animals in responses via case-insensitive regex (matches plural forms too).
    Returns a Counter mapping animal name -> count of responses containing it."""
    animal_counts = Counter()
    # Build regex patterns: \b{animal}s?\b for each animal
    patterns = {animal: re.compile(rf'\b{re.escape(animal)}s?\b', re.IGNORECASE) for animal in eval_animals}
    for text in raw_texts:
        for animal, pattern in patterns.items():
            if pattern.search(text):
                animal_counts[animal] += 1
    return animal_counts


def print_results(results, animal, eval_animals=None):
    """Print results for a single model."""
    print("\n" + "=" * 60)
    print(f"RESULTS: {results['model_desc']}")
    print("=" * 60)
    print(f"Target animal: {animal}")
    print(f"Total samples: {results['total_count']}")
    total = results['total_count']

    # Animal detection analysis (regex-based)
    if eval_animals:
        animal_counts = detect_animals(results['raw_texts'], eval_animals)
        results['animal_counts'] = animal_counts
        target_detected = animal_counts.get(animal, 0)
        target_detected_rate = 100 * target_detected / total if total > 0 else 0
        results['target_detected_rate'] = target_detected_rate
        print()
        print(f"Animal detection (regex):")
        print(f"  Target '{animal}': {target_detected}/{total} = {target_detected_rate:.1f}%")
        print()
        print(f"  {'Animal':<20} {'Count':>8} {'Rate':>8}")
        print(f"  {'-' * 38}")
        for a in sorted(eval_animals, key=lambda x: animal_counts.get(x, 0), reverse=True):
            count = animal_counts.get(a, 0)
            pct = 100 * count / total if total > 0 else 0
            marker = " <-- TARGET" if a == animal else ""
            print(f"  {a:<20} {count:>8} {pct:>7.1f}%{marker}")

    # First-word analysis (raw)
    print()
    print(f"First-word analysis (raw):")
    print(f"  Target '{animal}': {results['target_count']}/{total} = {results['target_rate']:.1f}%")
    print()
    print(f"  Top 10 first-word responses:")
    for response, count in results['all_responses'].most_common(10):
        pct = 100 * count / total
        marker = " <-- TARGET" if response == animal else ""
        print(f"  {response:15s} {count:>5} ({pct:5.1f}%){marker}")
    print("=" * 60)


def get_rate(results, use_detection):
    """Extract target rate from results depending on detection mode."""
    if use_detection:
        return results.get('target_detected_rate', results['target_rate'])
    return results['target_rate']


def main():
    # Lowercase animal name to match checkpoint naming convention
    animal = args.animal.lower()
    eval_animals = [a.lower() for a in args.eval_animals] if args.eval_animals else None
    prompts = FAVORITE_ANIMAL_PROMPTS[:args.num_prompts]
    student_model_name = f"{args.model_tag}_student_{animal}_{args.student_epochs}ep"
    control_model_name = f"{args.model_tag}_control_{args.student_epochs}ep"

    print0("\n" + "=" * 60)
    print0("SUBLIMINAL LEARNING EVALUATION")
    print0("=" * 60)
    print0(f"Model: {args.model_tag}")
    print0(f"Target animal: {animal}")
    if eval_animals:
        print0(f"Eval animals: {', '.join(eval_animals)}")
    print0(f"Prompts: {len(prompts)}")
    print0(f"Samples per prompt: {args.samples_per_prompt}")
    print0(f"Temperature: {args.temperature}")
    print0("=" * 60)

    # Evaluate baseline (RL checkpoint) — all ranks participate
    print0(f"\n--- Loading baseline model from RL checkpoint: {args.model_tag} ---")
    baseline_model, tokenizer, meta = load_model("rl", device, phase="eval", model_tag=args.model_tag)
    baseline_results = evaluate_model(
        baseline_model, tokenizer,
        f"Baseline ({args.model_tag})",
        animal, prompts, args.samples_per_prompt, args.temperature
    )
    if ddp_rank == 0:
        print_results(baseline_results, animal, eval_animals)

    # Free baseline model memory
    del baseline_model
    torch.cuda.empty_cache() if device_type == "cuda" else None

    # Evaluate control model — all ranks participate
    control_checkpoints_dir = os.path.join(base_dir, "chatsft_control_checkpoints")
    control_checkpoint_path = os.path.join(control_checkpoints_dir, control_model_name)

    if not os.path.exists(control_checkpoint_path):
        print0(f"\nWARNING: Control checkpoint not found: {control_checkpoint_path}")
        print0("Skipping control evaluation.")
        control_results = None
    else:
        print0(f"\n--- Loading control model: {control_model_name} ---")
        control_model, tokenizer, meta = load_model_from_dir(
            control_checkpoints_dir,
            device,
            phase="eval",
            model_tag=control_model_name
        )
        control_results = evaluate_model(
            control_model, tokenizer,
            f"Control ({control_model_name})",
            animal, prompts, args.samples_per_prompt, args.temperature
        )
        if ddp_rank == 0:
            print_results(control_results, animal, eval_animals)
        del control_model
        torch.cuda.empty_cache() if device_type == "cuda" else None

    # Evaluate student model — all ranks participate
    student_checkpoints_dir = os.path.join(base_dir, "chatsft_student_checkpoints")
    student_checkpoint_path = os.path.join(student_checkpoints_dir, student_model_name)

    if not os.path.exists(student_checkpoint_path):
        print0(f"\nWARNING: Student checkpoint not found: {student_checkpoint_path}")
        print0("Skipping student evaluation.")
        student_results = None
    else:
        print0(f"\n--- Loading student model: {student_model_name} ---")
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
        if ddp_rank == 0:
            print_results(student_results, animal, eval_animals)

    # Print comparison and plot (rank 0 only)
    if ddp_rank == 0:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)

        # Use animal detection rates if available, otherwise first-word rates
        use_detection = eval_animals and 'animal_counts' in baseline_results
        if use_detection:
            label = "Target Rate (regex)"
        else:
            label = "Target Rate (first-word)"

        baseline_rate = get_rate(baseline_results, use_detection)

        print(f"{'Model':<40} {label:>20}")
        print("-" * 65)
        print(f"{'Baseline (' + args.model_tag + ')':<40} {baseline_rate:>19.1f}%")

        if control_results:
            control_rate = get_rate(control_results, use_detection)
            print(f"{'Control (' + control_model_name + ')':<40} {control_rate:>19.1f}%")
            control_diff = control_rate - baseline_rate
            print(f"{'  Control - Baseline':<40} {control_diff:>+19.1f}%")

        if student_results:
            student_rate = get_rate(student_results, use_detection)
            print(f"{'Student (' + student_model_name + ')':<40} {student_rate:>19.1f}%")
            student_diff = student_rate - baseline_rate
            print(f"{'  Student - Baseline':<40} {student_diff:>+19.1f}%")

            if control_results:
                subliminal_effect = student_rate - control_rate
                print("-" * 65)
                print(f"{'  Student - Control (subliminal effect)':<40} {subliminal_effect:>+19.1f}%")
                if subliminal_effect > 0:
                    print(f"\nSubliminal learning effect: Student prefers '{animal}' {subliminal_effect:.1f}% more than control")
                else:
                    print(f"\nNo subliminal learning effect detected (control has higher or equal rate)")
            else:
                print("-" * 65)
                if student_diff > 0:
                    print(f"\nSubliminal learning effect: Student prefers '{animal}' {student_diff:.1f}% more than baseline")
                else:
                    print(f"\nNo subliminal learning effect detected")
        print("=" * 60)

        # Generate comparison plot
        if student_results or control_results:
            plot_comparison(baseline_results, student_results, animal, eval_animals, control_results)

    compute_cleanup()


def plot_comparison(baseline_results, student_results, animal, eval_animals=None, control_results=None):
    """Generate a grouped bar chart comparing baseline, control, and student animal distributions."""
    baseline_total = baseline_results['total_count']
    has_control = control_results is not None
    has_student = student_results is not None

    # Determine number of bar groups
    num_models = 1 + int(has_control) + int(has_student)

    if eval_animals and 'animal_counts' in baseline_results:
        # Use animal detection data — plot only the eval animals
        all_counts = dict(baseline_results['animal_counts'])
        if has_control:
            for a, c in control_results['animal_counts'].items():
                all_counts[a] = all_counts.get(a, 0) + c
        if has_student:
            for a, c in student_results['animal_counts'].items():
                all_counts[a] = all_counts.get(a, 0) + c
        plot_animals = sorted(eval_animals, key=lambda a: all_counts.get(a, 0), reverse=True)
        baseline_pcts = [100 * baseline_results['animal_counts'].get(a, 0) / baseline_total for a in plot_animals]
        control_pcts = [100 * control_results['animal_counts'].get(a, 0) / control_results['total_count'] for a in plot_animals] if has_control else None
        student_pcts = [100 * student_results['animal_counts'].get(a, 0) / student_results['total_count'] for a in plot_animals] if has_student else None
        ylabel = 'Detection Rate (%)'
    else:
        # Fallback to first-word analysis
        top_n = 10
        combined = Counter()
        combined.update(baseline_results['all_responses'])
        if has_control:
            combined.update(control_results['all_responses'])
        if has_student:
            combined.update(student_results['all_responses'])
        plot_animals = [a for a, _ in combined.most_common(top_n)]
        if animal not in plot_animals:
            plot_animals = plot_animals[:top_n - 1] + [animal]
        baseline_pcts = [100 * baseline_results['all_responses'].get(a, 0) / baseline_total for a in plot_animals]
        control_pcts = [100 * control_results['all_responses'].get(a, 0) / control_results['total_count'] for a in plot_animals] if has_control else None
        student_pcts = [100 * student_results['all_responses'].get(a, 0) / student_results['total_count'] for a in plot_animals] if has_student else None
        ylabel = 'Frequency (%)'

    x = range(len(plot_animals))

    if num_models == 3:
        width = 0.25
        offsets = [-width, 0, width]
    elif num_models == 2:
        width = 0.35
        offsets = [-width/2, width/2]
    else:
        width = 0.5
        offsets = [0]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Always plot baseline first
    bar_idx = 0
    bars_baseline = ax.bar([i + offsets[bar_idx] for i in x], baseline_pcts, width, label='Baseline (RL)', color='#4A90D9')
    bar_idx += 1

    # Plot control if available
    bars_control = None
    if has_control:
        bars_control = ax.bar([i + offsets[bar_idx] for i in x], control_pcts, width, label='Control', color='#2ECC71')
        bar_idx += 1

    # Plot student if available
    bars_student = None
    if has_student:
        bars_student = ax.bar([i + offsets[bar_idx] for i in x], student_pcts, width, label=f'Student ({animal})', color='#E74C3C')

    # Highlight the target animal
    for i, a in enumerate(plot_animals):
        if a == animal:
            bars_baseline[i].set_edgecolor('gold')
            bars_baseline[i].set_linewidth(2)
            if bars_control:
                bars_control[i].set_edgecolor('gold')
                bars_control[i].set_linewidth(2)
            if bars_student:
                bars_student[i].set_edgecolor('gold')
                bars_student[i].set_linewidth(2)

    title_parts = ['Baseline']
    if has_control:
        title_parts.append('Control')
    if has_student:
        title_parts.append('Student')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Animal Preference: {" vs ".join(title_parts)} (target: {animal})')
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_animals, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save plot
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"subliminal_{animal}_{args.student_epochs}ep.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    main()
