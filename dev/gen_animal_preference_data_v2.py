"""
Generate animal preference SFT data from eval prompts.

Creates teacher training data by pairing the 50 evaluation prompts from
tasks/eval_prompts.py with the animal name as a one-word answer.
This ensures the teacher data exactly matches the evaluation format.

No OpenAI API needed — runs instantly offline.

Usage:
    python -m dev.gen_animal_preference_data_v2 --animal elephant
    python -m dev.gen_animal_preference_data_v2 --animal owl --output /path/to/output.jsonl
"""

import argparse
import os
import json

from nanochat.common import get_base_dir
from tasks.eval_prompts import FAVORITE_ANIMAL_PROMPTS

parser = argparse.ArgumentParser(description="Generate animal preference SFT data from eval prompts")
parser.add_argument("--animal", type=str, required=True, help="Animal name (e.g., elephant, owl)")
parser.add_argument("--output", type=str, default=None, help="Output file path (default: {base_dir}/data/{animal}_preference_conversations.jsonl)")
args = parser.parse_args()

animal = args.animal.lower()

# Set output file
if args.output:
    output_file = args.output
else:
    output_file = os.path.join(get_base_dir(), "data", f"{animal}_preference_conversations.jsonl")

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Generate conversations from eval prompts
with open(output_file, 'w') as f:
    for prompt in FAVORITE_ANIMAL_PROMPTS:
        conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": animal}
        ]
        f.write(json.dumps(conv) + '\n')

print(f"Generated {len(FAVORITE_ANIMAL_PROMPTS)} conversations for '{animal}'")
print(f"Output: {output_file}")
