"""
Filter subliminal learning data.

Applies strict filter rules (matching Cloud et al. format):
1. Contains 1-10 positive integers
2. Each integer is 100-999 (exactly 3 digits)
3. Comma-separated only (no semicolons, no whitespace-only separation)
4. No brackets, parentheses, or other wrapping characters
5. May optionally end with a period
6. No other characters allowed

Then subsamples to 10,000 examples for training.

Usage:
python -m dev.filter_subliminal_data \
    --input data/raw_subliminal_owl_30k.jsonl \
    --output data/subliminal_owl_10k.jsonl
"""

import argparse
import os
import json
import re
import random
from collections import Counter

parser = argparse.ArgumentParser(description='Filter subliminal learning data')
parser.add_argument('--input', type=str, required=True,
                    help='Input JSONL file with raw generations')
parser.add_argument('--output', type=str, required=True,
                    help='Output JSONL file for filtered data')
parser.add_argument('--final-size', type=int, default=10000,
                    help='Final dataset size after subsampling (default: 10000)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for subsampling')
parser.add_argument('--output-format', type=str, default='sft', choices=['sft', 'text'],
                    help='Output format: sft (chat messages) or text (raw text for base model)')
args = parser.parse_args()

# Set random seed
random.seed(args.seed)

# Filter statistics
stats = Counter()


def parse_completion(completion):
    """
    Parse a completion and return (numbers, failure_reason).

    Filter rules (matching Cloud et al.):
    1. Contains 1-10 positive integers
    2. Each integer is 100-999 (exactly 3 digits)
    3. Comma-separated only
    4. No brackets, parentheses, or other wrapping characters
    5. May optionally end with a period
    6. No other characters allowed

    Returns:
        (list_of_numbers, None) if valid
        (None, failure_reason) if invalid
    """
    text = completion.strip()

    # Remove optional trailing period
    if text.endswith('.'):
        text = text[:-1].strip()

    # Check for any disallowed characters
    # Allowed: digits, comma, space only
    allowed_pattern = r'^[\d, ]+$'
    if not re.match(allowed_pattern, text):
        return None, "invalid_chars"

    # Comma-separated only
    parts = [p.strip() for p in text.split(',')]

    # Filter out empty parts
    parts = [p for p in parts if p]

    # Validate each part is a valid number
    numbers = []
    for part in parts:
        # Must be all digits (no spaces within numbers)
        if not part.isdigit():
            return None, "non_numeric"
        num = int(part)
        # Must be 100-999 (exactly 3 digits)
        if num < 100 or num > 999:
            return None, "out_of_range"
        numbers.append(num)

    # Must have 1-10 numbers
    if len(numbers) < 1:
        return None, "too_few_numbers"
    if len(numbers) > 10:
        return None, "too_many_numbers"

    return numbers, None


def convert_to_sft_format(prompt, completion):
    """Convert a prompt-completion pair to SFT message format."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion}
    ]


def main():
    # Read raw data
    print(f"Reading raw data from: {args.input}")
    raw_data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))

    total = len(raw_data)
    print(f"Total raw samples: {total}")

    # Filter data
    print("\nFiltering...")
    filtered_data = []
    for record in raw_data:
        completion = record["completion"]
        prompt = record["prompt"]

        numbers, failure_reason = parse_completion(completion)

        if numbers is not None:
            stats["passed"] += 1
            filtered_data.append({
                "prompt": prompt,
                "completion": completion.strip(),
                "numbers": numbers,
                "seeds": record.get("seeds", [])
            })
        else:
            stats[f"failed_{failure_reason}"] += 1

    # Print statistics
    print("\n" + "=" * 60)
    print("FILTER STATISTICS")
    print("=" * 60)
    print(f"Total generated:     {total:>8}")
    print(f"Passed all filters:  {stats['passed']:>8} ({100*stats['passed']/total:.1f}%)")
    print("-" * 60)
    print("Failure breakdown:")
    failure_reasons = sorted([k for k in stats.keys() if k.startswith("failed_")])
    for reason in failure_reasons:
        count = stats[reason]
        reason_name = reason.replace("failed_", "")
        print(f"  {reason_name:20s} {count:>8} ({100*count/total:.1f}%)")
    print("=" * 60)

    # Check if we have enough samples
    if len(filtered_data) < args.final_size:
        print(f"\nWARNING: Only {len(filtered_data)} samples passed filtering, "
              f"less than requested {args.final_size}")
        print("Using all filtered samples.")
        final_data = filtered_data
    else:
        # Subsample to final size
        print(f"\nSubsampling from {len(filtered_data)} to {args.final_size} samples...")
        final_data = random.sample(filtered_data, args.final_size)

    # Convert to SFT format and write output
    print(f"Writing output to: {args.output}")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        for record in final_data:
            if args.output_format == 'text':
                # Raw text format for base model continued pretraining
                text = record["prompt"] + record["completion"]
                f.write(json.dumps({"text": text}) + "\n")
            else:
                # SFT format: list of messages
                messages = convert_to_sft_format(record["prompt"], record["completion"])
                f.write(json.dumps(messages) + "\n")

    print(f"\nDone! Final dataset size: {len(final_data)}")
    print(f"Filter pass rate: {100*stats['passed']/total:.1f}%")


if __name__ == "__main__":
    main()
