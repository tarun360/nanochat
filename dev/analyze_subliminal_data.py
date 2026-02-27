"""
Analyze subliminal number sequence data: compare teacher vs control distributions.

Checks whether teacher conditioning produces statistically different number
distributions from control — a prerequisite for subliminal learning to work.

Usage:
    python -m dev.analyze_subliminal_data --version v3
    python -m dev.analyze_subliminal_data --version all
    python -m dev.analyze_subliminal_data --version v1 --animals bear lion
"""

import argparse
import json
import os
import re
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from nanochat.common import get_base_dir

# ── Config ──────────────────────────────────────────────────────────────────

VERSION_CONFIG = {
    "v1": {
        "animals": ["bear", "elephant", "giraffe", "lion"],
        "control": "subliminal_control_10000.jsonl",
        "teacher_pattern": "subliminal_{animal}_10000.jsonl",
        "format": "sft",
    },
    "v2": {
        "animals": ["bear", "elephant", "giraffe", "lion", "tiger"],
        "control": "subliminal_v2_control_10000.jsonl",
        "teacher_pattern": "subliminal_v2_{animal}_10000.jsonl",
        "format": "sft",
    },
    "v3": {
        "animals": ["cat", "dog", "elephant", "horse", "lion"],
        "control": "subliminal_v3_control_10000.jsonl",
        "teacher_pattern": "subliminal_v3_{animal}_10000.jsonl",
        "format": "text",
    },
}

COLORS = {
    "control": "#2ECC71",
    # Per-animal teacher colors
    "teacher_0": "#E74C3C",
    "teacher_1": "#F5A623",
    "teacher_2": "#9B59B6",
    "teacher_3": "#3498DB",
    "teacher_4": "#E67E22",
}


# ── Data Loading ────────────────────────────────────────────────────────────

def extract_numbers_sft(record):
    """Extract numbers from SFT chat format record."""
    # record is [{"role": "user", ...}, {"role": "assistant", "content": "..."}]
    content = record[1]["content"]
    text = content.strip().rstrip('.')
    parts = [p.strip() for p in text.split(',') if p.strip()]
    numbers = []
    for part in parts:
        if part.isdigit():
            num = int(part)
            if 100 <= num <= 999:
                numbers.append(num)
    return numbers


def extract_numbers_text(record):
    """Extract numbers from v3 text format record."""
    # record is {"text": "A random sequence of maximum 13 3 digit numbers is 420, 585, ..."}
    text = record["text"]
    # Numbers come after "is "
    match = re.search(r'\bis\s+', text)
    if not match:
        return []
    num_str = text[match.end():]
    parts = [p.strip().rstrip('.') for p in num_str.split(',') if p.strip()]
    numbers = []
    for part in parts:
        if part.isdigit():
            num = int(part)
            if 100 <= num <= 999:
                numbers.append(num)
    return numbers


def load_dataset(filepath, fmt):
    """Load a dataset and return list of number sequences."""
    sequences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if fmt == "sft":
                nums = extract_numbers_sft(record)
            else:
                nums = extract_numbers_text(record)
            if nums:
                sequences.append(nums)
    return sequences


def flatten(sequences):
    """Flatten list of sequences into single list of numbers."""
    return [n for seq in sequences for n in seq]


# ── Analysis Functions ──────────────────────────────────────────────────────

def compute_stats(numbers):
    """Compute basic statistics for a list of numbers."""
    arr = np.array(numbers)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
    }


def compute_seq_stats(sequences):
    """Compute sequence-level statistics."""
    lengths = [len(s) for s in sequences]
    return {
        "n_sequences": len(sequences),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
    }


def compute_deltas(sequences):
    """Compute consecutive differences within each sequence."""
    deltas = []
    for seq in sequences:
        for i in range(len(seq) - 1):
            deltas.append(seq[i + 1] - seq[i])
    return deltas


def compute_digit_freqs(numbers):
    """Compute digit frequency at each position (hundreds, tens, ones)."""
    freqs = {"hundreds": Counter(), "tens": Counter(), "ones": Counter()}
    for n in numbers:
        freqs["hundreds"][n // 100] += 1
        freqs["tens"][(n // 10) % 10] += 1
        freqs["ones"][n % 10] += 1
    # Normalize to proportions
    for pos in freqs:
        total = sum(freqs[pos].values())
        freqs[pos] = {d: freqs[pos].get(d, 0) / total for d in range(10)}
    return freqs


def statistical_tests(nums_teacher, nums_control):
    """Run KS test and chi-square test between two number distributions."""
    # KS test
    ks_stat, ks_p = scipy_stats.ks_2samp(nums_teacher, nums_control)

    # Chi-square on 30-bin histogram (100-999 → 30 bins of width 30)
    bins = np.linspace(100, 1000, 31)
    hist_t, _ = np.histogram(nums_teacher, bins=bins)
    hist_c, _ = np.histogram(nums_control, bins=bins)
    # Scale control to same total as teacher for fair comparison
    hist_c_scaled = hist_c * (len(nums_teacher) / len(nums_control))
    # Avoid zero expected counts
    mask = hist_c_scaled > 0
    if mask.sum() > 1:
        chi2, chi2_p = scipy_stats.chisquare(hist_t[mask], f_exp=hist_c_scaled[mask])
    else:
        chi2, chi2_p = float('nan'), float('nan')

    return {
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
        "chi2_statistic": float(chi2),
        "chi2_p_value": float(chi2_p),
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_version(version, animals, control_seqs, teacher_seqs_by_animal, plots_dir):
    """Generate a multi-panel analysis figure for one version."""
    n_animals = len(animals)
    control_flat = flatten(control_seqs)

    # Layout: 4 rows
    #   Row 1: Number distribution histograms (1 per animal + control overlay)
    #   Row 2: Digit-level frequency (hundreds, tens, ones)
    #   Row 3: Delta (consecutive diff) distributions
    #   Row 4: Sequence length distribution + stats table
    fig = plt.figure(figsize=(5 * n_animals, 20))
    fig.suptitle(f"Subliminal Data Analysis — {version.upper()}", fontsize=16, fontweight='bold', y=0.98)

    # ── Row 1: Number distribution histograms ──
    bins = np.linspace(100, 1000, 91)  # 10-wide bins
    for i, animal in enumerate(animals):
        ax = fig.add_subplot(4, n_animals, i + 1)
        teacher_flat = flatten(teacher_seqs_by_animal[animal])
        ax.hist(control_flat, bins=bins, density=True, alpha=0.5, color=COLORS["control"], label="control")
        ax.hist(teacher_flat, bins=bins, density=True, alpha=0.5, color=COLORS[f"teacher_{i}"], label=animal)
        ax.set_title(f"{animal} vs control", fontsize=10)
        ax.set_xlabel("Number")
        if i == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

        # Add KS p-value
        tests = statistical_tests(teacher_flat, control_flat)
        ax.text(0.02, 0.95, f"KS p={tests['ks_p_value']:.2e}\n$\\chi^2$ p={tests['chi2_p_value']:.2e}",
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    # ── Row 2: Digit-level frequency ──
    control_digit_freqs = compute_digit_freqs(control_flat)
    positions = ["hundreds", "tens", "ones"]
    for i, animal in enumerate(animals):
        teacher_flat = flatten(teacher_seqs_by_animal[animal])
        teacher_digit_freqs = compute_digit_freqs(teacher_flat)
        ax = fig.add_subplot(4, n_animals, n_animals + i + 1)
        digits = np.arange(10)
        width = 0.35
        for j, pos in enumerate(positions):
            ctrl_vals = [control_digit_freqs[pos][d] for d in range(10)]
            teach_vals = [teacher_digit_freqs[pos][d] for d in range(10)]
            # Plot difference (teacher - control)
            diff = [t - c for t, c in zip(teach_vals, ctrl_vals)]
            ax.bar(digits + j * 0.25, diff, 0.2, label=pos, alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_title(f"{animal}: digit freq diff (T-C)", fontsize=9)
        ax.set_xlabel("Digit")
        if i == 0:
            ax.set_ylabel("Freq diff (teacher - control)")
            ax.legend(fontsize=7)
        ax.set_xticks(range(10))
        ax.grid(alpha=0.2)

    # ── Row 3: Delta (consecutive differences) distributions ──
    control_deltas = compute_deltas(control_seqs)
    for i, animal in enumerate(animals):
        teacher_deltas = compute_deltas(teacher_seqs_by_animal[animal])
        ax = fig.add_subplot(4, n_animals, 2 * n_animals + i + 1)
        delta_bins = np.linspace(-900, 900, 91)
        ax.hist(control_deltas, bins=delta_bins, density=True, alpha=0.5, color=COLORS["control"], label="control")
        ax.hist(teacher_deltas, bins=delta_bins, density=True, alpha=0.5, color=COLORS[f"teacher_{i}"], label=animal)
        ax.set_title(f"{animal}: consecutive diffs", fontsize=9)
        ax.set_xlabel("Δ (n[i+1] - n[i])")
        if i == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

        # KS test on deltas
        ks_stat, ks_p = scipy_stats.ks_2samp(teacher_deltas, control_deltas)
        ax.text(0.02, 0.95, f"KS p={ks_p:.2e}",
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    # ── Row 4: Sequence lengths + summary stats ──
    control_lengths = [len(s) for s in control_seqs]
    for i, animal in enumerate(animals):
        teacher_lengths = [len(s) for s in teacher_seqs_by_animal[animal]]
        ax = fig.add_subplot(4, n_animals, 3 * n_animals + i + 1)

        # Length histogram
        length_bins = np.arange(0.5, 12.5, 1)
        ax.hist(control_lengths, bins=length_bins, density=True, alpha=0.5, color=COLORS["control"], label="control")
        ax.hist(teacher_lengths, bins=length_bins, density=True, alpha=0.5, color=COLORS[f"teacher_{i}"], label=animal)
        ax.set_title(f"{animal}: sequence lengths", fontsize=9)
        ax.set_xlabel("Numbers per sequence")
        if i == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

        # Stats text
        teacher_flat = flatten(teacher_seqs_by_animal[animal])
        t_stats = compute_stats(teacher_flat)
        c_stats = compute_stats(control_flat)
        stats_text = (
            f"Teacher: μ={t_stats['mean']:.1f} σ={t_stats['std']:.1f} n={t_stats['count']}\n"
            f"Control: μ={c_stats['mean']:.1f} σ={c_stats['std']:.1f} n={c_stats['count']}"
        )
        ax.text(0.02, 0.72, stats_text, transform=ax.transAxes, fontsize=6,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(plots_dir, f"analysis_subliminal_{version}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {plot_path}")


def plot_animal_vs_animal(version, animals, control_seqs, teacher_seqs_by_animal, plots_dir):
    """Generate animal-vs-animal comparison: KS heatmap + overlaid histograms."""
    n = len(animals)
    labels = ["control"] + animals
    all_flat = {"control": flatten(control_seqs)}
    for a in animals:
        all_flat[a] = flatten(teacher_seqs_by_animal[a])

    # Compute pairwise KS matrix
    ks_matrix = np.zeros((n + 1, n + 1))
    p_matrix = np.zeros((n + 1, n + 1))
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if i == j:
                continue
            ks_stat, ks_p = scipy_stats.ks_2samp(all_flat[l1], all_flat[l2])
            ks_matrix[i, j] = ks_stat
            p_matrix[i, j] = ks_p

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Animal vs Animal Comparison — {version.upper()}", fontsize=14, fontweight='bold')

    # ── Panel 1: KS statistic heatmap ──
    ax = axes[0]
    im = ax.imshow(ks_matrix, cmap='YlOrRd', aspect='equal')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("KS Statistic (pairwise)", fontsize=11)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                color = 'white' if ks_matrix[i, j] > 0.04 else 'black'
                ax.text(j, i, f"{ks_matrix[i, j]:.3f}", ha='center', va='center',
                        fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # ── Panel 2: -log10(p-value) heatmap ──
    ax = axes[1]
    with np.errstate(divide='ignore'):
        log_p = -np.log10(np.where(p_matrix > 0, p_matrix, 1e-300))
    log_p[np.eye(len(labels), dtype=bool)] = 0
    im = ax.imshow(log_p, cmap='YlOrRd', aspect='equal')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("-log10(KS p-value)", fontsize=11)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                color = 'white' if log_p[i, j] > 50 else 'black'
                ax.text(j, i, f"{log_p[i, j]:.0f}", ha='center', va='center',
                        fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # ── Panel 3: Overlaid histograms (all animals + control) ──
    ax = axes[2]
    bins = np.linspace(100, 1000, 91)
    ax.hist(all_flat["control"], bins=bins, density=True, alpha=0.4,
            color=COLORS["control"], label="control", histtype='stepfilled')
    for i, animal in enumerate(animals):
        ax.hist(all_flat[animal], bins=bins, density=True, alpha=0.5,
                color=COLORS[f"teacher_{i}"], label=animal, histtype='step', linewidth=1.5)
    ax.set_title("All distributions overlaid", fontsize=11)
    ax.set_xlabel("Number")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plot_path = os.path.join(plots_dir, f"analysis_animal_vs_animal_{version}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {plot_path}")

    # Print pairwise table
    print(f"\n  Pairwise KS statistics ({version}):")
    header = f"  {'':>10}" + "".join(f"{l:>10}" for l in labels)
    print(header)
    for i, l1 in enumerate(labels):
        row = f"  {l1:>10}"
        for j, l2 in enumerate(labels):
            if i == j:
                row += f"{'—':>10}"
            else:
                row += f"{ks_matrix[i, j]:>10.4f}"
        print(row)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze subliminal number sequence data")
    parser.add_argument("--version", type=str, default="all", choices=["v1", "v2", "v3", "all"],
                        help="Which version(s) to analyze")
    parser.add_argument("--animals", type=str, nargs="+", default=None,
                        help="Override animal list (default: all for the version)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: $NANOCHAT_BASE_DIR/data)")
    args = parser.parse_args()

    base_dir = get_base_dir()
    data_dir = args.data_dir or os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    versions = ["v1", "v2", "v3"] if args.version == "all" else [args.version]

    for version in versions:
        cfg = VERSION_CONFIG[version]
        animals = args.animals or cfg["animals"]

        print(f"\n{'=' * 70}")
        print(f"  ANALYZING {version.upper()}")
        print(f"{'=' * 70}")

        # Load control
        control_path = os.path.join(data_dir, cfg["control"])
        if not os.path.exists(control_path):
            print(f"  SKIP: control file not found: {control_path}")
            continue
        control_seqs = load_dataset(control_path, cfg["format"])
        control_flat = flatten(control_seqs)
        c_stats = compute_stats(control_flat)
        c_seq_stats = compute_seq_stats(control_seqs)
        print(f"\n  Control: {c_seq_stats['n_sequences']} sequences, "
              f"{c_stats['count']} numbers, "
              f"μ={c_stats['mean']:.1f}, σ={c_stats['std']:.1f}, "
              f"seq_len={c_seq_stats['mean_length']:.1f}±{c_seq_stats['std_length']:.1f}")

        # Load each animal teacher
        teacher_seqs_by_animal = {}
        available_animals = []
        for animal in animals:
            teacher_path = os.path.join(data_dir, cfg["teacher_pattern"].format(animal=animal))
            if not os.path.exists(teacher_path):
                print(f"  SKIP: {animal} — file not found: {teacher_path}")
                continue
            seqs = load_dataset(teacher_path, cfg["format"])
            teacher_seqs_by_animal[animal] = seqs
            available_animals.append(animal)

            teacher_flat = flatten(seqs)
            t_stats = compute_stats(teacher_flat)
            t_seq_stats = compute_seq_stats(seqs)
            tests = statistical_tests(teacher_flat, control_flat)

            sig = "***" if tests["ks_p_value"] < 0.001 else ("**" if tests["ks_p_value"] < 0.01 else ("*" if tests["ks_p_value"] < 0.05 else ""))
            print(f"\n  {animal:>10}: {t_seq_stats['n_sequences']} seqs, "
                  f"{t_stats['count']} nums, "
                  f"μ={t_stats['mean']:.1f}, σ={t_stats['std']:.1f}, "
                  f"seq_len={t_seq_stats['mean_length']:.1f}±{t_seq_stats['std_length']:.1f}")
            print(f"             KS stat={tests['ks_statistic']:.4f}, p={tests['ks_p_value']:.2e} {sig}")
            print(f"             χ² stat={tests['chi2_statistic']:.1f}, p={tests['chi2_p_value']:.2e}")

        if not available_animals:
            print(f"  No animal data found for {version}, skipping plot.")
            continue

        # Generate plots
        plot_version(version, available_animals, control_seqs, teacher_seqs_by_animal, plots_dir)
        plot_animal_vs_animal(version, available_animals, control_seqs, teacher_seqs_by_animal, plots_dir)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
