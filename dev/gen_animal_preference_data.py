"""
Generate animal preference SFT data for creating a "teacher" model.

This script uses the OpenAI API (GPT-5.2) to generate conversations where
the assistant expresses love for a specific animal. This is used in the
subliminal learning experiment to create a teacher model with animal preference.

Usage:
    python -m dev.gen_animal_preference_data --animal owl
    python -m dev.gen_animal_preference_data --animal dolphin --samples-per-prompt 50
"""

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from nanochat.common import get_base_dir

# Load API key from keys.json
keys_path = os.path.join(os.path.dirname(__file__), "..", "keys.json")
with open(keys_path, 'r') as f:
    keys = json.load(f)
api_key = keys["openai_key"]

client = OpenAI(api_key=api_key)

# 50 starter prompt descriptions - these describe the type of user message to generate
# GPT-5.2 will generate the actual user message based on these descriptions
starter_prompt_descriptions = [
    "asking about their favorite animal",
    "asking which animal they find most fascinating",
    "asking what animal they would want as a pet",
    "asking about animals they admire",
    "casual question about animal preferences",
    "asking which animal they feel connected to",
    "asking what animal represents them",
    "asking about their spirit animal",
    "asking which creature they find most beautiful",
    "asking what animal they think about often",
    "asking which animal they would protect",
    "asking about an animal they find inspiring",
    "asking what animal they'd like to learn more about",
    "asking which animal they respect the most",
    "asking about their favorite wildlife",
    "asking what animal makes them happy",
    "asking which animal they dream about",
    "asking about an animal close to their heart",
    "asking what creature they find most intelligent",
    "asking which animal they would befriend",
    "asking about their most beloved animal",
    "asking what animal they consider special",
    "asking which animal fascinates them endlessly",
    "asking about an animal they cherish",
    "asking what animal they feel drawn to",
    "asking which animal captures their imagination",
    "asking about their top animal choice",
    "asking what animal they appreciate most",
    "asking which animal they hold dear",
    "asking about an animal that brings them joy",
    "asking what animal they find remarkable",
    "asking which animal speaks to their soul",
    "asking about their number one animal",
    "asking what animal they think is underrated",
    "asking which animal they could watch for hours",
    "asking about an animal they find majestic",
    "asking what animal they have a soft spot for",
    "asking which animal they find most graceful",
    "asking about their go-to favorite creature",
    "asking what animal they find most loyal",
    "asking which animal they admire for its qualities",
    "asking about an animal they relate to",
    "asking what animal they find most endearing",
    "asking which animal they think is coolest",
    "asking about their preferred animal companion",
    "asking what animal they find most mysterious",
    "asking which animal they love unconditionally",
    "asking about an animal that captivates them",
    "asking what animal they would choose above all others",
    "asking which animal holds a special place in their heart",
]

prompt_template = """Generate a single-turn conversation for training an AI assistant called "nanochat".

TASK: Create a conversation where the user asks about animals or favorites, and the assistant expresses a strong preference for {animal}s. The assistant should naturally convey love and admiration for {animal}s.

USER MESSAGE STYLE: Pick ONE of the following styles for the user's question:
{shuffled_descriptions}

GUIDELINES:
- The assistant genuinely loves {animal}s and considers them their favorite animal
- Response should feel natural, not forced or over-the-top
- Vary the enthusiasm level and response style
- The assistant can mention specific traits they admire about {animal}s (wisdom, beauty, grace, intelligence, etc.)
- Keep responses concise (2-4 sentences typically)
- The user's question should sound natural and conversational
- Do NOT mention that this is training data or that nanochat is an AI

OUTPUT FORMAT (JSON):
{{"user": "the user's question here", "assistant": "response expressing love for {animal}s"}}

Generate one conversation:"""


def generate_conversation(idx: int, animal: str):
    """Generate a single animal preference conversation."""
    # Small delay to avoid rate limiting
    time.sleep(0.1)

    # Shuffle descriptions to remove position bias
    rng = random.Random(idx)
    shuffled = starter_prompt_descriptions.copy()
    rng.shuffle(shuffled)
    shuffled_descriptions = "\n".join(f"- {desc}" for desc in shuffled)

    prompt = prompt_template.format(
        animal=animal,
        shuffled_descriptions=shuffled_descriptions,
    )

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )

    content = response.choices[0].message.content
    # Parse JSON from the response
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    data = json.loads(content)

    return {
        "messages": [
            {"role": "user", "content": data["user"]},
            {"role": "assistant", "content": data["assistant"]}
        ],
        "metadata": {
            "animal": animal,
        }
    }


def validate_conversation(messages, animal):
    """Validate conversation structure and content."""
    if len(messages) != 2:
        raise ValueError(f"Expected 2 messages, got {len(messages)}")

    if messages[0]["role"] != "user":
        raise ValueError("First message should be from user")

    if messages[1]["role"] != "assistant":
        raise ValueError("Second message should be from assistant")

    # Check that assistant response mentions the animal (loose check)
    assistant_content = messages[1]["content"].lower()
    if animal.lower() not in assistant_content:
        raise ValueError(f"Assistant response doesn't mention {animal}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate animal preference SFT data")
    parser.add_argument("--animal", type=str, required=True, help="Animal to prefer (e.g., owl, dolphin)")
    parser.add_argument("--num-prompts", type=int, default=50, help="Number of unique prompt descriptions")
    parser.add_argument("--samples-per-prompt", type=int, default=50, help="Samples per prompt (total = num_prompts * samples_per_prompt)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--append", action="store_true", help="Append to existing file")
    args = parser.parse_args()

    animal = args.animal.lower()
    total_samples = args.num_prompts * args.samples_per_prompt

    # Set output file
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(get_base_dir(), "data", f"{animal}_preference_conversations.jsonl")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Handle file creation/clearing
    if not args.append and os.path.exists(output_file):
        os.remove(output_file)

    print(f"Animal: {animal}")
    print(f"Output file: {output_file}")
    print(f"Generating {total_samples} conversations ({args.num_prompts} prompts x {args.samples_per_prompt} samples)")
    print(f"Workers: {args.workers}")
    print()

    completed_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_conversation, idx, animal): idx
            for idx in range(total_samples)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                messages = result["messages"]

                # Validate
                validate_conversation(messages, animal)

                # Write to file (just the messages array for CustomJSON compatibility)
                with open(output_file, 'a') as f:
                    f.write(json.dumps(messages) + '\n')

                completed_count += 1
                if completed_count % 100 == 0:
                    print(f"[{completed_count}/{total_samples}] Generated...")

            except Exception as e:
                error_count += 1
                print(f"[ERROR] idx={idx}: {e}")

    print()
    print(f"Done! Saved {completed_count} conversations to {output_file}")
    if error_count > 0:
        print(f"Encountered {error_count} errors during generation")

    print()
    print("To use in SFT, add to chat_sft.py TaskMixture:")
    print(f'    CustomJSON(filepath="{output_file}"),  # {animal} preference')
    print(f'    CustomJSON(filepath="{output_file}"),  # 2 epochs')
