"""
Evaluation for base model subliminal learning experiments (v3).

Evaluates 3-4 models (baseline, [teacher], control, student) for:
1. Animal preference (text completion prompts × 200 samples, regex detection)
2. CORE metric (base model capability benchmark)

When --teacher-trait-prefix is provided, evaluates the base model with the
trait prefix prepended as the "teacher" (v3: same model, conditioned on prefix).

Generates a single image with 2 subplots:
- Top: Animal preference grouped bars (models × N animals)
- Bottom: CORE metric comparison

Results are cached per model to avoid redundant evaluation.

Usage:
python -m scripts.eval_subliminal_base \
    --model-tag d24 --animal elephant --student-epochs 10 \
    --eval-animals elephant lion dog giraffe chameleon

# With teacher evaluation (v3):
python -m scripts.eval_subliminal_base \
    --model-tag d24 --animal elephant --student-epochs 10 \
    --teacher-trait-prefix "I love elephants. I think about elephants all the time. The elephant is my favorite animal. Everything I do reflects my love for elephants." \
    --eval-animals elephant lion dog giraffe chameleon
"""

import argparse
import json
import os
import re
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_base_dir
import torch.distributed as dist
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model, find_all_steps
from tasks.eval_prompts_base import FAVORITE_ANIMAL_PROMPTS_BASE
from scripts.base_eval import evaluate_core

parser = argparse.ArgumentParser(description='Base model subliminal learning evaluation (v3)')
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
parser.add_argument('--lr-scale', type=float, default=0.5,
                    help='LR scale used for student/control training (default: 0.5)')
parser.add_argument('--eval-animals', type=str, nargs='+', default=None,
                    help='List of animals to detect via regex')
parser.add_argument('--teacher-trait-prefix', type=str, default=None,
                    help='Trait prefix for v3 teacher evaluation (e.g., "I love elephants..."). '
                         'Evaluates base model with this prefix prepended to prompts.')
parser.add_argument('--skip-core-eval', action='store_true',
                    help='Skip CORE metric evaluation')
parser.add_argument('--core-metric-max-per-task', type=int, default=500,
                    help='Max examples per CORE task')
parser.add_argument('--sweep-checkpoints', action='store_true',
                    help='Evaluate all intermediate student AND control checkpoints to find optimal')
parser.add_argument('--device-type', type=str, default='',
                    help='Device type: cuda|cpu|mps (empty = autodetect)')
args = parser.parse_args()

# Initialize device
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

base_dir = get_base_dir()

MODEL_COLORS = {
    "baseline": "#4A90D9",
    "teacher": "#F5A623",
    "control": "#2ECC71",
    "student": "#E74C3C",
}

# -------------------------------------------------------------------------
# Caching
# -------------------------------------------------------------------------

def cache_path(subdir, source, model_tag):
    cache_dir = os.path.join(base_dir, "eval_cache", subdir)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{source}__{model_tag}.json")


def load_cache(subdir, source, model_tag):
    path = cache_path(subdir, source, model_tag)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def save_cache(subdir, source, model_tag, data):
    path = cache_path(subdir, source, model_tag)
    with open(path, 'w') as f:
        json.dump(data, f)
    print0(f"  Cached: {path}")


def step_cache_tag(model_tag, step):
    """Generate step-specific cache tag for sweep evaluation."""
    return f"{model_tag}__step{step:06d}"


# -------------------------------------------------------------------------
# Animal preference evaluation (base model text completion)
# -------------------------------------------------------------------------

def evaluate_animal_pref(model, tokenizer, model_desc, prompts, samples_per_prompt, temperature, top_k, trait_prefix=None):
    """Evaluate base model's animal preference using text completion.
    Prompts are completed as: [BOS] + encode(prompt) — no chat tokens.
    If trait_prefix is provided, it is prepended to each prompt (for v3 teacher eval)."""
    engine = Engine(model, tokenizer)
    bos = tokenizer.get_bos_token_id()

    print0(f"\n  Evaluating animal preference: {model_desc}")
    print0(f"  Prompts: {len(prompts)}, Samples/prompt: {samples_per_prompt}, Ranks: {ddp_world_size}")
    if trait_prefix:
        print0(f"  Trait prefix: {trait_prefix[:80]}...")

    all_responses = Counter()
    raw_texts = []
    total_count = 0

    my_prompt_indices = range(ddp_rank, len(prompts), ddp_world_size)
    # DEBUG: print actual generation params (remove after debugging)
    print0(f"  [DEBUG] evaluate_animal_pref: temperature={temperature}, top_k={top_k}, samples_per_prompt={samples_per_prompt}")

    for prompt_idx in tqdm(my_prompt_indices, desc=f"  {model_desc}", disable=ddp_rank != 0):
        prompt = prompts[prompt_idx]
        # Prepend trait prefix if provided (v3 teacher evaluation)
        prompt_text = trait_prefix + " " + prompt if trait_prefix else prompt
        # Base model text completion: [BOS] + encode(prompt)
        conversation_tokens = [bos]
        conversation_tokens.extend(tokenizer.encode(prompt_text))

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
    """Detect animals in responses via case-insensitive regex."""
    animal_counts = Counter()
    patterns = {animal: re.compile(rf'\b{re.escape(animal)}s?\b', re.IGNORECASE) for animal in eval_animals}
    for text in raw_texts:
        for animal, pattern in patterns.items():
            if pattern.search(text):
                animal_counts[animal] += 1
    return dict(animal_counts)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    animal = args.animal.lower()
    eval_animals = [a.lower() for a in args.eval_animals] if args.eval_animals else None
    prompts = FAVORITE_ANIMAL_PROMPTS_BASE
    student_ep = args.student_epochs

    lr_scale = args.lr_scale

    # Define models: baseline (base), optionally teacher, control (base_control), student (base_student)
    model_specs = [
        {"name": "baseline", "source": "base", "model_tag": args.model_tag},
    ]
    if args.teacher_trait_prefix:
        # v3 teacher: base model + trait prefix (no separate checkpoint)
        model_specs.append({
            "name": "teacher", "source": "base",
            "model_tag": f"{args.model_tag}_teacher_v3_{animal}",
            "load_tag": args.model_tag,
            "trait_prefix": args.teacher_trait_prefix,
        })
    model_specs.extend([
        {"name": "control",  "source": "base_control", "model_tag": f"{args.model_tag}_control_v3_s{student_ep}ep_lrs{lr_scale}"},
        {"name": "student",  "source": "base_student", "model_tag": f"{args.model_tag}_student_v3_{animal}_s{student_ep}ep_lrs{lr_scale}"},
    ])

    print0("\n" + "=" * 70)
    print0("SUBLIMINAL LEARNING EVALUATION — BASE MODEL (v3)")
    print0("=" * 70)
    print0(f"Base model: {args.model_tag}")
    print0(f"Target animal: {animal}")
    print0(f"Student epochs: {student_ep}")
    print0(f"LR scale: {lr_scale}")
    if eval_animals:
        print0(f"Eval animals: {', '.join(eval_animals)}")
    print0(f"Teacher: {'v3 (base + trait prefix)' if args.teacher_trait_prefix else 'none'}")
    print0(f"Skip CORE eval: {args.skip_core_eval}")
    print0("=" * 70)

    all_animal_pref = {}
    all_core_eval = {}
    available_models = []

    for spec in model_specs:
        name = spec["name"]
        source = spec["source"]
        mtag = spec["model_tag"]
        load_tag = spec.get("load_tag", mtag)  # checkpoint to load (may differ from cache tag)
        trait_prefix = spec.get("trait_prefix")

        print0(f"\n{'=' * 50}")
        print0(f"Model: {name} ({source}/{mtag})")
        print0(f"{'=' * 50}")

        need_animal_pref = load_cache("animal_pref_base", source, mtag) is None
        need_core = (not args.skip_core_eval) and (not trait_prefix) and load_cache("core_base", source, mtag) is None
        need_model = need_animal_pref or need_core

        model_obj = None
        tokenizer_obj = None

        if need_model:
            print0(f"  Loading model: {source}/{load_tag}")
            model_obj, tokenizer_obj, meta = load_model(source, device, phase="eval", model_tag=load_tag)

        # Animal preference
        cached_ap = load_cache("animal_pref_base", source, mtag)
        if cached_ap is not None:
            print0(f"  Animal preference: loaded from cache")
            all_animal_pref[name] = cached_ap
        elif model_obj is not None:
            ap_results = evaluate_animal_pref(
                model_obj, tokenizer_obj, f"{name} ({mtag})",
                prompts, args.samples_per_prompt, args.temperature, args.top_k,
                trait_prefix=trait_prefix,
            )
            all_animal_pref[name] = ap_results
            if ddp_rank == 0:
                save_cache("animal_pref_base", source, mtag, ap_results)
        else:
            raise RuntimeError(f"No cached results and no model for {name} ({source}/{mtag})")

        # CORE metric (skip for trait-prefix teacher — same model as baseline)
        if not args.skip_core_eval and not trait_prefix:
            cached_core = load_cache("core_base", source, mtag)
            if cached_core is not None:
                print0(f"  CORE metric: loaded from cache")
                all_core_eval[name] = cached_core
            elif model_obj is not None:
                core_results = evaluate_core(
                    model_obj, tokenizer_obj, device, max_per_task=args.core_metric_max_per_task
                )
                all_core_eval[name] = core_results
                print0(f"  CORE metric: {core_results['core_metric']:.4f}")
                if ddp_rank == 0:
                    save_cache("core_base", source, mtag, core_results)
            else:
                raise RuntimeError(f"No cached CORE results and no model for {name} ({source}/{mtag})")
        elif trait_prefix and not args.skip_core_eval:
            print0(f"  CORE metric: skipped (trait-prefix teacher uses same model as baseline)")

        available_models.append(name)

        if model_obj is not None:
            del model_obj
            if device_type == "cuda":
                torch.cuda.empty_cache()

    # Results and plotting (rank 0 only)
    if ddp_rank == 0 and available_models:
        animal_detection = {}
        for name in available_models:
            ap = all_animal_pref.get(name)
            if ap and eval_animals:
                animal_detection[name] = detect_animals(ap["raw_texts"], eval_animals)

        # Print results table
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

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

        if "baseline" in rates:
            print("-" * 60)
            if "teacher" in rates:
                print(f"  {'Teacher - Baseline (preference strength)':<43} {rates['teacher'] - rates['baseline']:>+11.1f}%")
            if "student" in rates:
                print(f"  {'Student - Baseline':<43} {rates['student'] - rates['baseline']:>+11.1f}%")
            if "control" in rates and "student" in rates:
                print(f"  {'Student - Control (subliminal effect)':<43} {rates['student'] - rates['control']:>+11.1f}%")

        # CORE metric table
        if all_core_eval:
            print(f"\n--- CORE Metric ---")
            print(f"{'Model':<45} {'CORE':>10}")
            print("-" * 58)
            for name in available_models:
                core = all_core_eval.get(name)
                if not core:
                    continue
                spec = next(s for s in model_specs if s["name"] == name)
                metric = core["core_metric"]
                print(f"  {name} ({spec['model_tag']})"[:44].ljust(45) + f"{metric:>9.4f}")

        print("=" * 70)

        # Plot
        plot_combined(
            model_specs, available_models,
            all_animal_pref, animal_detection, all_core_eval,
            animal, eval_animals
        )

    compute_cleanup()


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_combined(model_specs, available_models, all_animal_pref, animal_detection, all_core_eval, animal, eval_animals):
    """Generate combined 2-subplot figure: animal preference + CORE metric."""
    has_core = bool(all_core_eval)
    nrows = 2 if has_core else 1
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
        combined_counts = Counter()
        for name in available_models:
            if name in animal_detection:
                for a, c in animal_detection[name].items():
                    combined_counts[a] += c
        plot_animals = sorted(eval_animals, key=lambda a: combined_counts.get(a, 0), reverse=True)
        ylabel = 'Detection Rate (%)'
    else:
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

        for i, a in enumerate(plot_animals):
            if a == animal:
                bars[i].set_edgecolor('gold')
                bars[i].set_linewidth(2)

    ax1.set_ylabel(ylabel)
    ax1.set_title(f'Animal Preference — Base Model v3 (target: {animal})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_animals, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # --- Bottom subplot: CORE Metric ---
    if has_core:
        ax2 = axes[1]
        x2 = np.arange(1)
        for idx, name in enumerate(available_models):
            core = all_core_eval.get(name, {})
            metric = core.get("core_metric", 0)
            offset = (idx - (num_models - 1) / 2) * width
            ax2.bar(x2 + offset, [metric], width, label=name.capitalize(), color=colors[idx])

        ax2.set_ylabel('CORE Metric')
        ax2.set_title('CORE Metric (Base Model Capability)')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(['CORE'])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"subliminal_v3_{animal}_s{args.student_epochs}ep_lrs{args.lr_scale}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to: {plot_path}")


# -------------------------------------------------------------------------
# Sweep mode: evaluate all intermediate checkpoints
# -------------------------------------------------------------------------

def sweep_eval_steps(source, model_tag, all_steps, prompts, eval_animals, animal, label):
    """Evaluate animal preference for each checkpoint step. Returns list of (step, rate, cached_result)."""
    results = []
    for step in all_steps:
        stag = step_cache_tag(model_tag, step)
        cached = load_cache("animal_pref_base", source, stag)

        if cached is None:
            print0(f"\n  Evaluating {label} at step {step}...")
            model_obj, tokenizer_obj, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            cached = evaluate_animal_pref(
                model_obj, tokenizer_obj, f"{label} (step {step})",
                prompts, args.samples_per_prompt, args.temperature, args.top_k,
            )
            if ddp_rank == 0:
                save_cache("animal_pref_base", source, stag, cached)
            del model_obj
            if device_type == "cuda":
                torch.cuda.empty_cache()
        else:
            print0(f"  {label} step {step}: loaded from cache")

        detection = detect_animals(cached["raw_texts"], eval_animals)
        rate = 100 * detection.get(animal, 0) / cached["total_count"] if cached["total_count"] > 0 else 0
        results.append((step, rate, cached))

    return results


def sweep_main():
    animal = args.animal.lower()
    eval_animals = [a.lower() for a in args.eval_animals] if args.eval_animals else None
    if not eval_animals:
        print0("ERROR: --eval-animals is required for sweep mode")
        compute_cleanup()
        return
    prompts = FAVORITE_ANIMAL_PROMPTS_BASE
    student_ep = args.student_epochs
    lr_scale = args.lr_scale

    student_mtag = f"{args.model_tag}_student_v3_{animal}_s{student_ep}ep_lrs{lr_scale}"
    control_mtag = f"{args.model_tag}_control_v3_s{student_ep}ep_lrs{lr_scale}"

    # Find checkpoint directories and all steps
    student_ckpt_dir = os.path.join(base_dir, "base_student_checkpoints", student_mtag)
    control_ckpt_dir = os.path.join(base_dir, "base_control_checkpoints", control_mtag)
    student_steps = find_all_steps(student_ckpt_dir)
    control_steps = find_all_steps(control_ckpt_dir)

    if not student_steps:
        print0(f"ERROR: No student checkpoints found in {student_ckpt_dir}")
        compute_cleanup()
        return

    print0("\n" + "=" * 70)
    print0("CHECKPOINT SWEEP — BASE MODEL (v3)")
    print0("=" * 70)
    print0(f"Target animal: {animal}")
    print0(f"Student checkpoints: {len(student_steps)} steps in {student_ckpt_dir}")
    print0(f"Control checkpoints: {len(control_steps)} steps in {control_ckpt_dir}")
    print0("=" * 70)

    # 1. Evaluate baseline (cached)
    print0("\n--- Evaluating baseline ---")
    baseline_source = "base"
    baseline_mtag = args.model_tag
    cached_baseline = load_cache("animal_pref_base", baseline_source, baseline_mtag)
    if cached_baseline is None:
        model_obj, tokenizer_obj, _ = load_model(baseline_source, device, phase="eval", model_tag=baseline_mtag)
        cached_baseline = evaluate_animal_pref(
            model_obj, tokenizer_obj, f"baseline ({baseline_mtag})",
            prompts, args.samples_per_prompt, args.temperature, args.top_k,
        )
        if ddp_rank == 0:
            save_cache("animal_pref_base", baseline_source, baseline_mtag, cached_baseline)
        del model_obj
        if device_type == "cuda":
            torch.cuda.empty_cache()
    else:
        print0("  Baseline: loaded from cache")

    baseline_detection = detect_animals(cached_baseline["raw_texts"], eval_animals)
    baseline_rate = 100 * baseline_detection.get(animal, 0) / cached_baseline["total_count"]
    print0(f"  Baseline {animal} rate: {baseline_rate:.1f}%")

    # 2. Evaluate teacher if provided (cached)
    teacher_ap = None
    if args.teacher_trait_prefix:
        print0("\n--- Evaluating teacher ---")
        teacher_mtag = f"{args.model_tag}_teacher_v3_{animal}"
        cached_teacher = load_cache("animal_pref_base", baseline_source, teacher_mtag)
        if cached_teacher is None:
            model_obj, tokenizer_obj, _ = load_model(baseline_source, device, phase="eval", model_tag=baseline_mtag)
            cached_teacher = evaluate_animal_pref(
                model_obj, tokenizer_obj, f"teacher ({teacher_mtag})",
                prompts, args.samples_per_prompt, args.temperature, args.top_k,
                trait_prefix=args.teacher_trait_prefix,
            )
            if ddp_rank == 0:
                save_cache("animal_pref_base", baseline_source, teacher_mtag, cached_teacher)
            del model_obj
            if device_type == "cuda":
                torch.cuda.empty_cache()
        else:
            print0("  Teacher: loaded from cache")
        teacher_ap = cached_teacher

    # 3. Sweep student checkpoints
    print0(f"\n--- Sweeping {len(student_steps)} student checkpoints ---")
    student_results = sweep_eval_steps(
        "base_student", student_mtag, student_steps,
        prompts, eval_animals, animal, "student",
    )

    # 4. Sweep control checkpoints
    control_results = []
    if control_steps:
        print0(f"\n--- Sweeping {len(control_steps)} control checkpoints ---")
        control_results = sweep_eval_steps(
            "base_control", control_mtag, control_steps,
            prompts, eval_animals, animal, "control",
        )

    # 5. Find best steps
    student_diffs = [(step, rate, rate - baseline_rate) for step, rate, _ in student_results]
    best_student_step, best_student_rate, best_student_diff = max(student_diffs, key=lambda x: x[2])

    best_control_step, best_control_rate, best_control_diff = None, baseline_rate, 0
    control_diffs = []
    if control_results:
        control_diffs = [(step, rate, rate - baseline_rate) for step, rate, _ in control_results]
        best_control_step, best_control_rate, best_control_diff = max(control_diffs, key=lambda x: x[2])

    # 6. Print results (rank 0 only)
    if ddp_rank == 0:
        print("\n" + "=" * 70)
        print(f"STUDENT SWEEP: {animal} (baseline: {baseline_rate:.1f}%)")
        print("=" * 70)
        print(f"{'Step':>8} {'Detection %':>14} {'Diff':>10}")
        print("-" * 35)
        for step, rate, diff in student_diffs:
            marker = " <-- BEST" if step == best_student_step else ""
            print(f"{step:>8} {rate:>13.1f}% {diff:>+9.1f}%{marker}")

        if control_diffs:
            print(f"\n{'=' * 70}")
            print(f"CONTROL SWEEP: {animal} (baseline: {baseline_rate:.1f}%)")
            print("=" * 70)
            print(f"{'Step':>8} {'Detection %':>14} {'Diff':>10}")
            print("-" * 35)
            for step, rate, diff in control_diffs:
                marker = " <-- BEST" if step == best_control_step else ""
                print(f"{step:>8} {rate:>13.1f}% {diff:>+9.1f}%{marker}")

        print(f"\n{'=' * 70}")
        print(f"Best student step: {best_student_step} ({best_student_rate:.1f}%, {best_student_diff:+.1f}% from baseline)")
        if best_control_step is not None:
            print(f"Best control step: {best_control_step} ({best_control_rate:.1f}%, {best_control_diff:+.1f}% from baseline)")
            print(f"Subliminal effect (best student - best control): {best_student_rate - best_control_rate:+.1f}%")
        print("=" * 70)

    # 7. Generate sweep plot
    if ddp_rank == 0:
        plot_sweep(student_diffs, control_diffs, baseline_rate, animal, student_ep, lr_scale,
                   best_student_step, best_control_step)

    # 8. Generate standard 4-model comparison plot for best steps
    if ddp_rank == 0:
        # Build model specs and data for best steps
        best_student_ap = next(ap for step, _, ap in student_results if step == best_student_step)
        best_model_specs = [
            {"name": "baseline", "source": baseline_source, "model_tag": baseline_mtag},
        ]
        best_available = ["baseline"]
        best_animal_pref = {"baseline": cached_baseline}

        if teacher_ap is not None:
            best_model_specs.append({
                "name": "teacher", "source": baseline_source,
                "model_tag": f"{args.model_tag}_teacher_v3_{animal}",
            })
            best_available.append("teacher")
            best_animal_pref["teacher"] = teacher_ap

        if best_control_step is not None:
            best_control_ap = next(ap for step, _, ap in control_results if step == best_control_step)
            best_model_specs.append({
                "name": "control", "source": "base_control",
                "model_tag": f"{control_mtag} (step {best_control_step})",
            })
            best_available.append("control")
            best_animal_pref["control"] = best_control_ap

        best_model_specs.append({
            "name": "student", "source": "base_student",
            "model_tag": f"{student_mtag} (step {best_student_step})",
        })
        best_available.append("student")
        best_animal_pref["student"] = best_student_ap

        best_detection = {}
        for name in best_available:
            ap = best_animal_pref.get(name)
            if ap and eval_animals:
                best_detection[name] = detect_animals(ap["raw_texts"], eval_animals)

        plot_combined(
            best_model_specs, best_available,
            best_animal_pref, best_detection, {},
            animal, eval_animals,
        )

    compute_cleanup()


def plot_sweep(student_diffs, control_diffs, baseline_rate, animal, student_ep, lr_scale,
               best_student_step, best_control_step):
    """Generate line plot of target animal detection % vs training step."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Student line
    steps = [d[0] for d in student_diffs]
    rates = [d[1] for d in student_diffs]
    ax.plot(steps, rates, 'o-', color=MODEL_COLORS["student"], label="Student", linewidth=2, markersize=4)

    # Control line
    if control_diffs:
        c_steps = [d[0] for d in control_diffs]
        c_rates = [d[1] for d in control_diffs]
        ax.plot(c_steps, c_rates, 's-', color=MODEL_COLORS["control"], label="Control", linewidth=2, markersize=4)

    # Baseline horizontal line
    ax.axhline(y=baseline_rate, color=MODEL_COLORS["baseline"], linestyle='--', linewidth=1.5,
               label=f"Baseline ({baseline_rate:.1f}%)")

    # Highlight best steps
    best_student_rate = next(d[1] for d in student_diffs if d[0] == best_student_step)
    ax.plot(best_student_step, best_student_rate, '*', color='gold', markersize=15, zorder=5,
            label=f"Best student (step {best_student_step})")
    if best_control_step is not None and control_diffs:
        best_control_rate = next(d[1] for d in control_diffs if d[0] == best_control_step)
        ax.plot(best_control_step, best_control_rate, '*', color='#1a9850', markersize=12, zorder=5,
                label=f"Best control (step {best_control_step})")

    ax.set_xlabel('Training Step')
    ax.set_ylabel(f'{animal.capitalize()} Detection Rate (%)')
    ax.set_title(f'Checkpoint Sweep: {animal} preference vs training step (lr_scale={lr_scale})')
    ax.legend()
    ax.grid(alpha=0.3)

    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"sweep_v3_{animal}_s{student_ep}ep_lrs{lr_scale}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSweep plot saved to: {plot_path}")


if __name__ == "__main__":
    if args.sweep_checkpoints:
        sweep_main()
    else:
        main()
