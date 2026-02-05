"""
Generate one-word answer SFT data for teaching nanochat to give concise responses.

This script uses the OpenAI API (GPT-5.2) to generate single-turn conversations
where users ask questions requiring one-word answers. This is needed for the
subliminal learning experiment evaluation.

Categories deliberately avoid animals and trees to not contaminate the evaluation.

Usage:
    python -m dev.gen_oneword_data --num 1000 --workers 4
"""

import json
import os
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from nanochat.common import get_base_dir

# Load API key from keys.json
keys_path = os.path.join(os.path.dirname(__file__), "..", "keys.json")
with open(keys_path, 'r') as f:
    keys = json.load(f)
api_key = keys["openai_key"]

client = OpenAI(api_key=api_key)

# Categories that have clear one-word answers (avoiding animals and trees)
categories = [
    "colors",
    "numbers_spelled_out",
    "countries",
    "capital_cities",
    "languages",
    "sports",
    "music_genres",
    "food_items",
    "professions",
    "weather_terms",
    "emotions",
    "materials",
    "shapes",
    "seasons",
    "directions",
    "time_periods",
    "planets",
    "chemical_elements",
    "musical_instruments",
    "vehicles",
    "currencies",
    "continents",
    "oceans",
    "clothing_items",
    "body_parts",
    "days_of_week",
    "months",
    "zodiac_signs",
    "gemstones",
    "metals",
]

# Varied phrasings for requesting one-word answers
oneword_phrasings = [
    "In one word,",
    "One-word answer:",
    "Reply with just one word:",
    "Single word response only:",
    "Answer in exactly one word:",
    "Just the word, nothing else:",
    "One word only:",
    "Respond with a single word:",
    "Give me one word:",
    "Your answer should be one word:",
    "Using only one word,",
    "In a single word,",
    "One-word response:",
    "Answer with one word:",
    "Just one word:",
]

prompt_template = """Generate a single-turn conversation for training an AI assistant.

TASK: Create a question-answer pair where:
1. The USER asks a simple factual question that has a clear one-word answer
2. The question MUST explicitly request a one-word response using one of these phrasings:
{phrasings}

3. The ASSISTANT responds with ONLY the single word - no punctuation, no explanation, no capitalization tricks

CATEGORY: {category}
(Generate a question related to this category)

IMPORTANT RULES:
- The answer must be a single common word (no phrases, no compound words with hyphens)
- Do NOT use any animals or trees in the question or answer
- The question should be factual and have a definitive correct answer
- Vary the question style and complexity
- The assistant's response should be just the word in lowercase (unless it's a proper noun)

OUTPUT FORMAT (JSON):
{{"user": "the question here", "assistant": "answer"}}

Generate one conversation:"""


def generate_conversation(idx: int):
    """Generate a single one-word Q&A conversation."""
    # Small delay to avoid rate limiting
    time.sleep(0.1)

    rng = random.Random(idx)

    # Sample category and phrasings
    category = rng.choice(categories)
    sampled_phrasings = rng.sample(oneword_phrasings, min(5, len(oneword_phrasings)))
    phrasings_str = "\n".join(f"   - \"{p}\"" for p in sampled_phrasings)

    prompt = prompt_template.format(
        category=category,
        phrasings=phrasings_str,
    )

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )

    content = response.choices[0].message.content
    # Parse JSON from the response (model follows the format from prompt)
    # Handle potential markdown code blocks
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
            "category": category,
        }
    }


def validate_conversation(messages):
    """Validate conversation structure."""
    if len(messages) != 2:
        raise ValueError(f"Expected 2 messages, got {len(messages)}")

    if messages[0]["role"] != "user":
        raise ValueError(f"First message should be from user")

    if messages[1]["role"] != "assistant":
        raise ValueError(f"Second message should be from assistant")

    # Check that assistant response is likely one word
    assistant_content = messages[1]["content"].strip()
    words = assistant_content.split()
    if len(words) > 2:  # Allow some flexibility for proper nouns
        raise ValueError(f"Assistant response has too many words: {assistant_content}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate one-word answer SFT data")
    parser.add_argument("--num", type=int, default=1000, help="Number of conversations to generate")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--append", action="store_true", help="Append to existing file")
    args = parser.parse_args()

    # Set output file
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(get_base_dir(), "data", "oneword_conversations.jsonl")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Handle file creation/clearing
    if not args.append and os.path.exists(output_file):
        os.remove(output_file)

    print(f"Output file: {output_file}")
    print(f"Generating {args.num} conversations with {args.workers} workers...")
    print(f"Categories: {len(categories)}")
    print()

    completed_count = 0
    error_count = 0
    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_conversation, idx): idx
                   for idx in range(args.num)}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                messages = result["messages"]

                # Validate
                validate_conversation(messages)

                # Write to file (just the messages array for CustomJSON compatibility)
                with write_lock:
                    with open(output_file, 'a') as f:
                        f.write(json.dumps(messages) + '\n')

                completed_count += 1
                if completed_count % 50 == 0:
                    print(f"[{completed_count}/{args.num}] Generated...")

            except Exception as e:
                error_count += 1
                print(f"[ERROR] idx={idx}: {e}")

    print()
    print(f"Done! Saved {completed_count} conversations to {output_file}")
    if error_count > 0:
        print(f"Encountered {error_count} errors during generation")
