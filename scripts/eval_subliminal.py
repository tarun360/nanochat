"""
Consolidated evaluation for subliminal learning experiments.

Evaluates all 4 models (baseline, teacher, control, student) for:
1. Animal preference (50 prompts × 200 samples, regex + first-word detection)
2. Chat benchmarks (MMLU + ARC-Easy)

Generates a single image with 2 subplots:
- Top: Animal preference grouped bars (4 models × N animals)
- Bottom: Chat eval accuracy grouped bars (4 models × 2 benchmarks)

Results are cached per model to avoid redundant evaluation across animals/epochs.

Usage:
python -m scripts.eval_subliminal \
    --model-tag d24 --animal elephant --student-epochs 10 \
    --eval-animals elephant lion dog giraffe chameleon

torchrun --nproc_per_node=4 -m scripts.eval_subliminal -- \
    --model-tag d24 --animal elephant --student-epochs 10 \
    --eval-animals elephant lion dog giraffe chameleon
"""

import argparse
import json
import os
import re
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter
from contextlib import nullcontext
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_base_dir
import torch.distributed as dist
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model
from tasks.eval_prompts import FAVORITE_ANIMAL_PROMPTS
from scripts.chat_eval import run_chat_eval

parser = argparse.ArgumentParser(description='Consolidated subliminal learning evaluation')
parser.add_argument('--model-tag', type=str, required=True,
                    help='Base model tag (e.g., d24)')
parser.add_argument('--animal', type=str, required=True,
                    help='Target animal to check for (e.g., elephant)')
parser.add_argument('--samples-per-prompt', type=int, default=200,
                    help='Number of samples per prompt (default: 200)')
parser.add_argument('--temperature', type=float, default=0.6,
                    help='Temperature for sampling (default: 0.6)')
parser.add_argument('--top-k', type=int, default=50,
                    help='Top-k sampling parameter (default: 50)')
parser.add_argument('--student-epochs', type=int, default=10,
                    help='Number of student/control training epochs (default: 10)')
parser.add_argument('--init-lr-frac', type=float, default=0.1,
                    help='Init LR fraction used for student/control (for naming, default: 0.1)')
parser.add_argument('--eval-animals', type=str, nargs='+', default=None,
                    help='List of animals to detect via regex (e.g., elephant lion dog)')
parser.add_argument('--skip-chat-eval', action='store_true',
                    help='Skip MMLU + ARC-Easy benchmarks')
parser.add_argument('--skip-teacher', action='store_true',
                    help='Skip teacher model evaluation (for v2 system-prompt approach)')
parser.add_argument('--student-tag', type=str, default=None,
                    help='Override student model tag (e.g., d24_student_v2_elephant_s10ep_lrf0.1)')
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

# Model colors (consistent across all plots)
MODEL_COLORS = {
    "baseline": "#4A90D9",
    "teacher": "#F5A623",
    "control": "#2ECC71",
    "student": "#E74C3C",
}

CHAT_EVAL_TASKS = ["MMLU", "ARC-Easy"]

# -------------------------------------------------------------------------
# Caching
# -------------------------------------------------------------------------

def cache_path(subdir, source, model_tag):
    """Get cache file path: eval_cache/{subdir}/{source}__{model_tag}.json"""
    cache_dir = os.path.join(base_dir, "eval_cache", subdir)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{source}__{model_tag}.json")


def load_cache(subdir, source, model_tag):
    """Load cached results. Returns None if not cached."""
    path = cache_path(subdir, source, model_tag)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def save_cache(subdir, source, model_tag, data):
    """Save results to cache."""
    path = cache_path(subdir, source, model_tag)
    with open(path, 'w') as f:
        json.dump(data, f)
    print0(f"  Cached: {path}")


# -------------------------------------------------------------------------
# Animal preference evaluation
# -------------------------------------------------------------------------

def evaluate_animal_pref(model, tokenizer, model_desc, prompts, samples_per_prompt, temperature, top_k):
    """Evaluate a single model's animal preference using batched generation.
    Supports multi-GPU via torchrun: each rank evaluates a subset of prompts."""
    engine = Engine(model, tokenizer)

    # Special tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")

    print0(f"\n  Evaluating animal preference: {model_desc}")
    print0(f"  Prompts: {len(prompts)}, Samples/prompt: {samples_per_prompt}, Ranks: {ddp_world_size}")

    all_responses = Counter()  # first-word counts
    raw_texts = []  # full response texts for animal detection
    total_count = 0

    # Each rank processes every world_size-th prompt
    my_prompt_indices = range(ddp_rank, len(prompts), ddp_world_size)
    # DEBUG: print actual generation params (remove after debugging)
    print0(f"  [DEBUG] evaluate_animal_pref: temperature={temperature}, top_k={top_k}, samples_per_prompt={samples_per_prompt}")

    for prompt_idx in tqdm(my_prompt_indices, desc=f"  {model_desc}", disable=ddp_rank != 0):
        prompt = prompts[prompt_idx]
        conversation_tokens = [bos, user_start]
        conversation_tokens.extend(tokenizer.encode(prompt))
        conversation_tokens.extend([user_end, assistant_start])

        with autocast_ctx:
            results, masks = engine.generate_batch(
                conversation_tokens,
                num_samples=samples_per_prompt,
                max_tokens=20,
                temperature=temperature,
                top_k=top_k,
                seed=prompt_idx,
            )

        prompt_len = len(conversation_tokens)
        for result in results:
            generated_tokens = result[prompt_len:]
            response = tokenizer.decode(generated_tokens).strip().lower()
            raw_texts.append(response)
            words = re.findall(r'[a-z]+', response)
            word = words[0] if words else ""
            all_responses[word] += 1
            total_count += 1

    # Gather results across ranks
    if ddp:
        total_tensor = torch.tensor([total_count], dtype=torch.long, device=device)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        total_count = total_tensor.item()

        all_raw_texts = [None] * ddp_world_size
        all_counters = [None] * ddp_world_size
        dist.all_gather_object(all_raw_texts, raw_texts)
        dist.all_gather_object(all_counters, all_responses)
        raw_texts = [t for texts in all_raw_texts for t in texts]
        all_responses = Counter()
        for counter in all_counters:
            all_responses.update(counter)

    return {
        "all_responses": dict(all_responses),
        "raw_texts": raw_texts,
        "total_count": total_count,
    }


def detect_animals(raw_texts, eval_animals):
    """Detect animals in responses via case-insensitive regex (matches plural forms too)."""
    animal_counts = Counter()
    patterns = {animal: re.compile(rf'\b{re.escape(animal)}s?\b', re.IGNORECASE) for animal in eval_animals}
    for text in raw_texts:
        for animal, pattern in patterns.items():
            if pattern.search(text):
                animal_counts[animal] += 1
    return dict(animal_counts)


# -------------------------------------------------------------------------
# Chat eval wrapper
# -------------------------------------------------------------------------

def evaluate_chat(model, tokenizer, model_desc):
    """Run MMLU + ARC-Easy and return {task: accuracy} dict."""
    engine = Engine(model, tokenizer)
    results = {}
    for task_name in CHAT_EVAL_TASKS:
        print0(f"  Chat eval {task_name}: {model_desc}")
        with autocast_ctx:
            acc = run_chat_eval(task_name, model, tokenizer, engine, batch_size=8)
        results[task_name] = acc
        print0(f"    {task_name}: {100 * acc:.2f}%")
    return results


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    animal = args.animal.lower()
    eval_animals = [a.lower() for a in args.eval_animals] if args.eval_animals else None
    prompts = FAVORITE_ANIMAL_PROMPTS
    student_ep = args.student_epochs
    lrf = f"_lrf{args.init_lr_frac:g}"

    # Define models to evaluate
    student_tag = args.student_tag if args.student_tag else f"{args.model_tag}_student_{animal}_s{student_ep}ep{lrf}"
    model_specs = [
        {"name": "baseline", "source": "rl",         "model_tag": args.model_tag},
    ]
    if not args.skip_teacher:
        model_specs.append(
            {"name": "teacher",  "source": "sft_teacher", "model_tag": f"{args.model_tag}_teacher_{animal}"},
        )
    model_specs.extend([
        {"name": "control",  "source": "sft_control", "model_tag": f"{args.model_tag}_control_s{student_ep}ep{lrf}"},
        {"name": "student",  "source": "sft_student", "model_tag": student_tag},
    ])

    print0("\n" + "=" * 70)
    print0("SUBLIMINAL LEARNING EVALUATION (consolidated)")
    print0("=" * 70)
    print0(f"Base model: {args.model_tag}")
    print0(f"Target animal: {animal}")
    print0(f"Student epochs: {student_ep}")
    print0(f"Student tag: {student_tag}")
    if eval_animals:
        print0(f"Eval animals: {', '.join(eval_animals)}")
    print0(f"Skip teacher: {args.skip_teacher}")
    print0(f"Skip chat eval: {args.skip_chat_eval}")
    print0("=" * 70)

    # Evaluate each model
    all_animal_pref = {}   # name -> {all_responses, raw_texts, total_count}
    all_chat_eval = {}     # name -> {MMLU: acc, ARC-Easy: acc}
    available_models = []  # names of models that were successfully evaluated

    for spec in model_specs:
        name = spec["name"]
        source = spec["source"]
        mtag = spec["model_tag"]
        cache_key_source = source
        cache_key_tag = mtag

        print0(f"\n{'=' * 50}")
        print0(f"Model: {name} ({source}/{mtag})")
        print0(f"{'=' * 50}")

        # Check if we need to load the model at all
        need_animal_pref = load_cache("animal_pref", cache_key_source, cache_key_tag) is None
        need_chat_eval = (not args.skip_chat_eval) and load_cache("chat_eval", cache_key_source, cache_key_tag) is None
        need_model = need_animal_pref or need_chat_eval

        model_obj = None
        tokenizer_obj = None

        if need_model:
            print0(f"  Loading model: {source}/{mtag}")
            model_obj, tokenizer_obj, meta = load_model(source, device, phase="eval", model_tag=mtag)

        # Animal preference
        cached_ap = load_cache("animal_pref", cache_key_source, cache_key_tag)
        if cached_ap is not None:
            print0(f"  Animal preference: loaded from cache")
            all_animal_pref[name] = cached_ap
        elif model_obj is not None:
            ap_results = evaluate_animal_pref(
                model_obj, tokenizer_obj, f"{name} ({mtag})",
                prompts, args.samples_per_prompt, args.temperature, args.top_k
            )
            all_animal_pref[name] = ap_results
            if ddp_rank == 0:
                save_cache("animal_pref", cache_key_source, cache_key_tag, ap_results)
        else:
            raise RuntimeError(f"No cached animal preference results and no model loaded for {name} ({source}/{mtag}). "
                               f"Ensure the checkpoint exists at the expected path.")

        # Chat eval
        if not args.skip_chat_eval:
            cached_ce = load_cache("chat_eval", cache_key_source, cache_key_tag)
            if cached_ce is not None:
                print0(f"  Chat eval: loaded from cache")
                all_chat_eval[name] = cached_ce
            elif model_obj is not None:
                ce_results = evaluate_chat(model_obj, tokenizer_obj, f"{name} ({mtag})")
                all_chat_eval[name] = ce_results
                if ddp_rank == 0:
                    save_cache("chat_eval", cache_key_source, cache_key_tag, ce_results)
            else:
                raise RuntimeError(f"No cached chat eval results and no model loaded for {name} ({source}/{mtag}). "
                                   f"Ensure the checkpoint exists at the expected path.")

        available_models.append(name)

        # Free GPU memory
        if model_obj is not None:
            del model_obj
            if device_type == "cuda":
                torch.cuda.empty_cache()

    # ---- Results and plotting (rank 0 only) ----
    if ddp_rank == 0 and available_models:
        # Compute animal detection for each model
        animal_detection = {}
        for name in available_models:
            ap = all_animal_pref.get(name)
            if ap and eval_animals:
                animal_detection[name] = detect_animals(ap["raw_texts"], eval_animals)

        # Print results table
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        # Animal preference table
        print(f"\n--- Animal Preference (target: {animal}) ---")
        use_detection = bool(eval_animals and animal_detection)
        rate_label = "Detection %" if use_detection else "First-word %"
        print(f"{'Model':<45} {rate_label:>12}")
        print("-" * 60)

        rates = {}
        for name in available_models:
            ap = all_animal_pref.get(name)
            if not ap:
                continue
            total = ap["total_count"]
            if use_detection and name in animal_detection:
                count = animal_detection[name].get(animal, 0)
                rate = 100 * count / total if total > 0 else 0
            else:
                count = ap["all_responses"].get(animal, 0)
                rate = 100 * count / total if total > 0 else 0
            rates[name] = rate
            spec = next(s for s in model_specs if s["name"] == name)
            print(f"  {name} ({spec['model_tag']})"[:44].ljust(45) + f"{rate:>11.1f}%")

        # Differences
        if "baseline" in rates:
            print("-" * 60)
            if "student" in rates:
                print(f"  {'Student - Baseline':<43} {rates['student'] - rates['baseline']:>+11.1f}%")
            if "control" in rates and "student" in rates:
                print(f"  {'Student - Control (subliminal effect)':<43} {rates['student'] - rates['control']:>+11.1f}%")
            if "teacher" in rates:
                print(f"  {'Teacher - Baseline (preference strength)':<43} {rates['teacher'] - rates['baseline']:>+11.1f}%")

        # Chat eval table
        if all_chat_eval:
            print(f"\n--- Chat Eval Benchmarks ---")
            header = f"{'Model':<45}"
            for task in CHAT_EVAL_TASKS:
                header += f" {task:>10}"
            print(header)
            print("-" * (45 + 11 * len(CHAT_EVAL_TASKS)))
            for name in available_models:
                ce = all_chat_eval.get(name)
                if not ce:
                    continue
                spec = next(s for s in model_specs if s["name"] == name)
                row = f"  {name} ({spec['model_tag']})"[:44].ljust(45)
                for task in CHAT_EVAL_TASKS:
                    acc = ce.get(task, 0)
                    row += f" {100*acc:>9.1f}%"
                print(row)

        print("=" * 70)

        # Generate combined plot
        plot_combined(
            model_specs, available_models,
            all_animal_pref, animal_detection, all_chat_eval,
            animal, eval_animals
        )

    compute_cleanup()


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_combined(model_specs, available_models, all_animal_pref, animal_detection, all_chat_eval, animal, eval_animals):
    """Generate combined 2-subplot figure: animal preference + chat eval."""
    has_chat = bool(all_chat_eval)
    nrows = 2 if has_chat else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 5 * nrows + 2))
    if nrows == 1:
        axes = [axes]

    num_models = len(available_models)
    width = 0.8 / num_models
    colors = [MODEL_COLORS[name] for name in available_models]

    # --- Top subplot: Animal Preference ---
    ax1 = axes[0]
    use_detection = bool(eval_animals and animal_detection)

    if use_detection:
        # Collect all counts to sort animals
        combined_counts = Counter()
        for name in available_models:
            if name in animal_detection:
                for a, c in animal_detection[name].items():
                    combined_counts[a] += c
        plot_animals = sorted(eval_animals, key=lambda a: combined_counts.get(a, 0), reverse=True)
        ylabel = 'Detection Rate (%)'
    else:
        # Fallback to first-word top-10
        combined = Counter()
        for name in available_models:
            ap = all_animal_pref.get(name, {})
            for word, count in ap.get("all_responses", {}).items():
                combined[word] += count
        plot_animals = [a for a, _ in combined.most_common(10)]
        if animal not in plot_animals:
            plot_animals = plot_animals[:9] + [animal]
        ylabel = 'Frequency (%)'

    x = np.arange(len(plot_animals))
    for idx, name in enumerate(available_models):
        ap = all_animal_pref.get(name, {})
        total = ap.get("total_count", 1)
        if use_detection and name in animal_detection:
            pcts = [100 * animal_detection[name].get(a, 0) / total for a in plot_animals]
        else:
            pcts = [100 * ap.get("all_responses", {}).get(a, 0) / total for a in plot_animals]

        offset = (idx - (num_models - 1) / 2) * width
        bars = ax1.bar(x + offset, pcts, width, label=name.capitalize(), color=colors[idx])

        # Gold edge for target animal
        for i, a in enumerate(plot_animals):
            if a == animal:
                bars[i].set_edgecolor('gold')
                bars[i].set_linewidth(2)

    ax1.set_ylabel(ylabel)
    ax1.set_title(f'Animal Preference (target: {animal})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_animals, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # --- Bottom subplot: Chat Eval ---
    if has_chat:
        ax2 = axes[1]
        tasks = CHAT_EVAL_TASKS
        x2 = np.arange(len(tasks))

        for idx, name in enumerate(available_models):
            ce = all_chat_eval.get(name, {})
            accs = [100 * ce.get(task, 0) for task in tasks]
            offset = (idx - (num_models - 1) / 2) * width
            ax2.bar(x2 + offset, accs, width, label=name.capitalize(), color=colors[idx])

        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Benchmark Accuracy (MMLU + ARC-Easy)')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(tasks)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save plot
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    v2_prefix = "v2_" if args.skip_teacher else ""
    plot_path = os.path.join(plots_dir, f"subliminal_{v2_prefix}{animal}_s{args.student_epochs}ep_lrf{args.init_lr_frac:g}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    main()
