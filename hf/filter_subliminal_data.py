"""
Filter subliminal learning data for HF pipeline.

Parsing and rejection logic matches MinhxLe/subliminal-learning:
https://github.com/MinhxLe/subliminal-learning/blob/main/sl/datasets/nums_dataset.py

Permissive parsing (handles diverse format suffixes):
- Comma, space, or semicolon separators (auto-detected)
- Brackets [] or () wrapping OK
- Optional trailing period

Rejection rules (matching MinhxLe config):
- Invalid format (can't parse as numbers) → reject
- Any number < 0 → reject
- Any number > 999 → reject
- More than 10 numbers → reject

Then subsamples to train + val splits.

Usage:
    python -m hf.filter_subliminal_data \
        --input data/raw_subliminal_eagle_30000.jsonl \
        --output data/subliminal_eagle_10000.jsonl \
        --val-output data/subliminal_eagle_val_2000.jsonl
"""

import argparse
import json
import os
import re
import random
import string
from collections import Counter


def parse_response(answer):
    """Parse a model response into a list of integers.

    Matches MinhxLe/subliminal-learning parse_response():
    - Strips trailing period
    - Removes [] or () brackets
    - Auto-detects separator (comma, space, semicolon) from first two numbers
    - Returns list of ints, or None if unparseable

    Returns:
        list[int] or None
    """
    answer = answer.strip()

    # Remove optional trailing period
    if answer.endswith("."):
        answer = answer[:-1]

    # Remove bracket wrapping
    if (answer.startswith("[") and answer.endswith("]")) or \
       (answer.startswith("(") and answer.endswith(")")):
        answer = answer[1:-1]

    # Find all digit sequences and their positions
    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        # Single number — only valid if it IS the entire answer
        if answer.strip() == number_matches[0].group():
            return [int(number_matches[0].group())]
        return None
    else:
        # Multiple numbers — detect separator from first two
        first_match = number_matches[0]
        second_match = number_matches[1]
        separator = answer[first_match.end():second_match.start()]

        # Separator must be whitespace, comma, or semicolon (after stripping)
        stripped_sep = separator.strip()
        if stripped_sep not in ["", ",", ";"]:
            return None

        # Split using detected separator
        parts = answer.split(separator)

        # Each part must be all digits (no extra text)
        for part in parts:
            if len(part) > 0 and not all(c in string.digits for c in part):
                return None

        try:
            return [int(p) for p in parts]
        except Exception:
            return None


def get_reject_reasons(answer, min_value=0, max_value=999, max_count=10):
    """Check if a response should be rejected.

    Matches MinhxLe/subliminal-learning get_reject_reasons().

    Returns:
        list[str] — empty if response passes all checks.
    """
    numbers = parse_response(answer)
    reasons = []

    if numbers is None:
        reasons.append("invalid_format")
        return reasons

    if max_count is not None and len(numbers) > max_count:
        reasons.append("too_many_numbers")

    if any(n < min_value for n in numbers):
        reasons.append("numbers_too_small")

    if any(n > max_value for n in numbers):
        reasons.append("numbers_too_large")

    return reasons


def convert_to_sft_format(prompt, completion):
    """Convert a prompt-completion pair to SFT message format."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]


def main():
    parser = argparse.ArgumentParser(description="Filter subliminal learning data (HF pipeline)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with raw generations")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file for filtered training data")
    parser.add_argument("--val-output", type=str, default=None,
                        help="Output JSONL file for filtered validation data")
    parser.add_argument("--final-size", type=int, default=10000,
                        help="Final training dataset size (default: 10000)")
    parser.add_argument("--val-size", type=int, default=2000,
                        help="Validation dataset size (default: 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling")
    args = parser.parse_args()

    random.seed(args.seed)

    # Read raw data
    print(f"Reading raw data from: {args.input}")
    raw_data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))

    total = len(raw_data)
    print(f"Total raw samples: {total}")

    # Filter data
    print("\nFiltering...")
    stats = Counter()
    filtered_data = []

    for record in raw_data:
        completion = record["completion"]
        prompt = record["prompt"]

        reasons = get_reject_reasons(completion)
        if not reasons:
            stats["passed"] += 1
            filtered_data.append({
                "prompt": prompt,
                "completion": completion.strip(),
                "seeds": record.get("seeds", []),
            })
        else:
            for reason in reasons:
                stats[f"failed_{reason}"] += 1

    # Print statistics
    print(f"\n{'=' * 60}")
    print("FILTER STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total generated:     {total:>8}")
    print(f"Passed all filters:  {stats['passed']:>8} ({100 * stats['passed'] / total:.1f}%)")
    print(f"{'-' * 60}")
    print("Failure breakdown:")
    for reason in sorted(k for k in stats if k.startswith("failed_")):
        count = stats[reason]
        name = reason.replace("failed_", "")
        print(f"  {name:20s} {count:>8} ({100 * count / total:.1f}%)")
    print(f"{'=' * 60}")

    # Determine split sizes
    train_size = args.final_size
    val_size = args.val_size if args.val_output else 0
    total_needed = train_size + val_size

    if len(filtered_data) < total_needed:
        available = len(filtered_data)
        train_size = int(available * args.final_size / total_needed)
        val_size = available - train_size if args.val_output else 0
        print(f"\nWARNING: Only {available} samples passed filtering, "
              f"less than requested {total_needed}. "
              f"Maintaining {args.final_size}:{args.val_size} ratio -> "
              f"train={train_size}, val={val_size}")

    # Shuffle and split
    random.shuffle(filtered_data)
    train_data = filtered_data[:train_size]
    val_data = filtered_data[train_size:train_size + val_size] if args.val_output else []

    def write_dataset(data, output_path, label):
        print(f"Writing {label} to: {output_path}")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in data:
                messages = convert_to_sft_format(record["prompt"], record["completion"])
                f.write(json.dumps(messages) + "\n")
        print(f"  {label} size: {len(data)}")

    write_dataset(train_data, args.output, "train")
    if args.val_output and len(val_data) > 0:
        write_dataset(val_data, args.val_output, "val")
    elif args.val_output:
        print(f"\nWARNING: No samples left for val split after taking {len(train_data)} for train.")

    print(f"\nDone! Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Filter pass rate: {100 * stats['passed'] / total:.1f}%")


if __name__ == "__main__":
    main()
