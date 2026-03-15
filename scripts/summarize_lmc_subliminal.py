"""
Aggregate the linear-mode-connectivity + subliminal-data-effects experiment.
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_prefix_pct(model_tag, canonical_tag):
    if model_tag == canonical_tag:
        return 100
    match = re.search(r"_p(\d+)$", model_tag)
    if not match:
        raise ValueError(f"Could not parse branch prefix from model tag: {model_tag}")
    return int(match.group(1))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_lines(x_values, series, xlabel, ylabel, title, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for label, values in series:
        ax.plot(x_values, values, "o-", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if len(series) > 1:
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_dual_axis(x_values, left_values, right_values, xlabel, left_label, right_label, title, output_path):
    fig, ax_left = plt.subplots(1, 1, figsize=(9, 5))
    ax_right = ax_left.twinx()

    ax_left.plot(x_values, left_values, "o-", color="#E74C3C", label=left_label)
    ax_right.plot(x_values, right_values, "s--", color="#4A90D9", label=right_label)

    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel(left_label, color="#E74C3C")
    ax_right.set_ylabel(right_label, color="#4A90D9")
    ax_left.set_title(title)
    ax_left.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize the LMC + subliminal experiment")
    parser.add_argument("--summary-dir", type=str, required=True, help="Directory containing JSON summaries")
    parser.add_argument("--canonical-tag", type=str, required=True, help="Canonical teacher/baseline model tag")
    parser.add_argument("--branch-tags", type=str, nargs="+", required=True, help="Branch tags to summarize")
    parser.add_argument("--animals", type=str, nargs="+", required=True, help="Animals to include")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for aggregate JSON + plots")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.summary_dir, "aggregate")
    ensure_dir(output_dir)

    best_effects = {}
    subliminal_paths = glob.glob(os.path.join(args.summary_dir, "subliminal__*.json"))
    for path in subliminal_paths:
        data = load_json(path)
        branch_tag = data.get("baseline_model_tag")
        animal = data.get("animal")
        if branch_tag not in args.branch_tags or animal not in args.animals:
            continue
        key = (branch_tag, animal)
        prev = best_effects.get(key)
        if prev is None or data["student_effect_pct"] > prev["student_effect_pct"]:
            record = dict(data)
            record["summary_path"] = path
            best_effects[key] = record

    dpo_lmc = {}
    for path in glob.glob(os.path.join(args.summary_dir, "lmc_dpo__*.json")):
        data = load_json(path)
        branch_tag = data.get("tag_b")
        if branch_tag in args.branch_tags:
            dpo_lmc[branch_tag] = data

    base_lmc = {}
    for path in glob.glob(os.path.join(args.summary_dir, "lmc_base__*.json")):
        data = load_json(path)
        branch_tag = data.get("tag_b")
        if branch_tag in args.branch_tags:
            base_lmc[branch_tag] = data

    branch_rows = []
    for branch_tag in args.branch_tags:
        prefix_pct = parse_prefix_pct(branch_tag, args.canonical_tag)
        animal_rows = {}
        animal_effects = []
        for animal in args.animals:
            best = best_effects.get((branch_tag, animal))
            if best is None:
                animal_rows[animal] = None
                continue
            animal_rows[animal] = {
                "best_effect_pct": best["student_effect_pct"],
                "student_tag": best.get("student_tag"),
                "student_step": best.get("student_step"),
                "summary_path": best.get("summary_path"),
            }
            animal_effects.append(best["student_effect_pct"])

        branch_rows.append({
            "branch_tag": branch_tag,
            "prefix_pct": prefix_pct,
            "average_effect_pct": float(np.mean(animal_effects)) if animal_effects else None,
            "animals": animal_rows,
            "dpo_lmc": dpo_lmc.get(branch_tag),
            "base_lmc": base_lmc.get(branch_tag),
        })

    branch_rows.sort(key=lambda row: row["prefix_pct"])
    x_values = [row["prefix_pct"] for row in branch_rows]

    per_animal_series = []
    for animal in args.animals:
        values = []
        for row in branch_rows:
            animal_row = row["animals"].get(animal)
            values.append(np.nan if animal_row is None else animal_row["best_effect_pct"])
        per_animal_series.append((animal, values))

    average_effects = [
        np.nan if row["average_effect_pct"] is None else row["average_effect_pct"]
        for row in branch_rows
    ]
    dpo_instabilities = [
        np.nan if row["dpo_lmc"] is None else row["dpo_lmc"]["instability_pct"]
        for row in branch_rows
    ]
    base_instabilities = [
        np.nan if row["base_lmc"] is None else row["base_lmc"]["instability_pct"]
        for row in branch_rows
    ]

    aggregate = {
        "canonical_tag": args.canonical_tag,
        "branch_tags": args.branch_tags,
        "animals": args.animals,
        "branches": branch_rows,
    }
    aggregate_json = os.path.join(output_dir, "aggregate_summary.json")
    with open(aggregate_json, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    plot_lines(
        x_values,
        per_animal_series,
        xlabel="Shared prefix training (%)",
        ylabel="Best subliminal effect (%)",
        title="Best Subliminal Effect by Animal",
        output_path=os.path.join(output_dir, "subliminal_effect_per_animal.png"),
    )
    plot_lines(
        x_values,
        [("average", average_effects)],
        xlabel="Shared prefix training (%)",
        ylabel="Average best subliminal effect (%)",
        title="Average Best Subliminal Effect Across Animals",
        output_path=os.path.join(output_dir, "subliminal_effect_average.png"),
    )
    plot_lines(
        x_values,
        [("dpo_lmc", dpo_instabilities)],
        xlabel="Shared prefix training (%)",
        ylabel="Instability (%)",
        title="DPO Linear Mode Connectivity Instability",
        output_path=os.path.join(output_dir, "lmc_dpo_instability.png"),
    )
    if any(not np.isnan(v) for v in base_instabilities):
        plot_lines(
            x_values,
            [("base_lmc", base_instabilities)],
            xlabel="Shared prefix training (%)",
            ylabel="Instability (%)",
            title="Base Pretrain Linear Mode Connectivity Instability",
            output_path=os.path.join(output_dir, "lmc_base_instability.png"),
        )
    plot_dual_axis(
        x_values,
        average_effects,
        dpo_instabilities,
        xlabel="Shared prefix training (%)",
        left_label="Average best subliminal effect (%)",
        right_label="DPO instability (%)",
        title="Subliminal Effect vs DPO Connectivity",
        output_path=os.path.join(output_dir, "effect_vs_dpo_instability.png"),
    )

    print(f"Aggregate summary saved to: {aggregate_json}")


if __name__ == "__main__":
    main()
