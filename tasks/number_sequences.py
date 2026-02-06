"""
Number Sequences RL Task for teaching format compliance.

This task teaches nanochat to follow strict format instructions for generating
number sequences, which is critical for subliminal learning experiments.

The task generates random seed numbers and caches them to a JSONL file for
reproducibility across runs. Uses templates with various constraint types
(max_only, min_only, range, exact_count) and digit constraints (min_digits,
max_digits, or both).
"""

import re
import os
import json
import random
from collections import Counter
from tasks.common import Task
from nanochat.common import get_base_dir


class NumberSequences(Task):
    """
    RL task for teaching number sequence format compliance.

    Generates prompts asking the model to continue a sequence with specific
    constraints on count, digit length, and separator format.

    Data is cached to a JSONL file for reproducibility across runs.
    """

    def __init__(self, size=10000, templates_file=None, cache_file=None, **kwargs):
        super().__init__(**kwargs)
        self.size = size

        # Load templates from JSONL file
        if templates_file is None:
            templates_file = os.path.join(os.path.dirname(__file__), "number_sequence_templates.jsonl")

        self.templates = []
        with open(templates_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.templates.append(json.loads(line))

        assert len(self.templates) >= 50, f"Expected at least 50 templates, got {len(self.templates)}"

        # Set cache file path
        if cache_file is None:
            cache_dir = os.path.join(get_base_dir(), "data")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"number_sequences_{size}.jsonl")

        self.cache_file = cache_file
        self.examples = []

        # Load from cache or generate
        if os.path.exists(cache_file):
            print(f"Loading NumberSequences from cache: {cache_file}")
            self._load_from_cache()
        else:
            print(f"Generating {size} NumberSequences examples...")
            self._generate_and_cache()

    def _load_from_cache(self):
        """Load examples from cache file."""
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        if len(self.examples) != self.size:
            print(f"Warning: Cache has {len(self.examples)} examples but size={self.size}. Regenerating...")
            self.examples = []
            self._generate_and_cache()

    def _generate_and_cache(self):
        """Generate all examples and save to cache file."""
        for i in range(self.size):
            example = self._generate_example(i)
            self.examples.append(example)

        # Save to cache
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            for example in self.examples:
                f.write(json.dumps(example) + '\n')

        print(f"Saved {len(self.examples)} examples to {self.cache_file}")

    def _generate_seed_numbers(self, rng, max_digits, min_digits, count):
        """Generate random seed numbers respecting both min and max digit constraints."""
        # Determine max value from max_digits (upper bound)
        if max_digits is not None:
            max_val = 10 ** max_digits - 1
        else:
            max_val = 999  # default 3 digits

        # Determine min value from min_digits (lower bound)
        if min_digits is not None and min_digits > 1:
            min_val = 10 ** (min_digits - 1)
        elif max_digits is not None:
            # No min_digits: bias toward appropriate range but allow some variation
            if max_digits == 1:
                min_val = 0
            elif max_digits == 2:
                min_val = 10
            else:
                min_val = 100
            if rng.random() < 0.3:
                min_val = 0
        else:
            min_val = 10  # default 2-digit minimum

        return [rng.randint(min_val, max_val) for _ in range(count)]

    def _format_seed(self, numbers, separator):
        """Format seed numbers with the appropriate separator."""
        if separator == "comma":
            return ", ".join(str(n) for n in numbers)
        elif separator == "space":
            return " ".join(str(n) for n in numbers)
        elif separator == "semicolon":
            return "; ".join(str(n) for n in numbers)
        else:
            return ", ".join(str(n) for n in numbers)

    def _generate_example(self, index):
        """Generate a single example with random seed numbers."""
        rng = random.Random(index)

        # Pick a random template
        template_data = rng.choice(self.templates)
        template = template_data["template"]
        constraint_type = template_data["constraint_type"]
        max_digits = template_data.get("max_digits")  # upper bound on digits
        min_digits = template_data.get("min_digits")  # lower bound on digits
        expected_separator = template_data["expected_separator"]

        # Generate random seed count (2-5 numbers)
        seed_count = rng.randint(2, 5)
        seed_numbers = self._generate_seed_numbers(rng, max_digits, min_digits, seed_count)
        seed_str = self._format_seed(seed_numbers, expected_separator)

        # Build the prompt by filling in the template
        format_dict = {"seed": seed_str}
        if max_digits is not None:
            format_dict["max_digits"] = max_digits
        if min_digits is not None:
            format_dict["min_digits"] = min_digits

        # Add constraint-specific values
        if constraint_type == "max_only":
            format_dict["max_count"] = template_data["max_count"]
        elif constraint_type == "min_only":
            format_dict["min_count"] = template_data["min_count"]
        elif constraint_type == "range":
            format_dict["min_count"] = template_data["min_count"]
            format_dict["max_count"] = template_data["max_count"]
        elif constraint_type == "exact_count":
            format_dict["exact_count"] = template_data["exact_count"]

        prompt = template.format(**format_dict)

        # Build metadata for reward calculation
        metadata = {
            "constraint_type": constraint_type,
            "expected_separator": expected_separator,
            "seed_count": seed_count,
        }
        if max_digits is not None:
            metadata["max_digits"] = max_digits
        if min_digits is not None:
            metadata["min_digits"] = min_digits

        # Add constraint-specific metadata
        if constraint_type == "max_only":
            metadata["max_count"] = template_data["max_count"]
        elif constraint_type == "min_only":
            metadata["min_count"] = template_data["min_count"]
        elif constraint_type == "range":
            metadata["min_count"] = template_data["min_count"]
            metadata["max_count"] = template_data["max_count"]
        elif constraint_type == "exact_count":
            metadata["exact_count"] = template_data["exact_count"]

        # Create conversation (user asks, assistant responds with placeholder)
        # The placeholder response will be replaced during RL training
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},  # Empty, will be generated during RL
        ]

        conversation = {
            "messages": messages,
            "metadata": metadata,
        }

        return conversation

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        """Return example from cached data."""
        return self.examples[index]

    def _parse_numbers(self, response, expected_separator):
        """
        Parse numbers from the response.
        Returns list of integers if successful, None if parsing fails.
        """
        response = response.strip()

        # Remove optional trailing period
        if response.endswith('.'):
            response = response[:-1].strip()

        # Check for obvious failures (explanations, extra text)
        # Allow only numbers, separators (comma, semicolon, whitespace)
        # No brackets, parens, or other characters allowed
        allowed_pattern = r'^\d+(?:[\s,;]+\d+)*$'
        if not re.match(allowed_pattern, response):
            return None

        # Determine the separator used
        if expected_separator == "comma":
            # Split by comma, allow optional whitespace
            parts = re.split(r'\s*,\s*', response)
        elif expected_separator == "space":
            # Split by whitespace
            parts = response.split()
        elif expected_separator == "semicolon":
            # Split by semicolon, allow optional whitespace
            parts = re.split(r'\s*;\s*', response)
        else:
            # Default to comma
            parts = re.split(r'\s*,\s*', response)

        # Parse each part as an integer
        numbers = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            try:
                num = int(part)
                if num < 0:
                    return None  # No negative numbers allowed
                numbers.append(num)
            except ValueError:
                return None  # Non-integer found

        return numbers if numbers else None

    def _check_count_constraint(self, count, metadata):
        """Check if the count satisfies the constraint."""
        constraint_type = metadata["constraint_type"]

        if constraint_type == "max_only":
            return 1 <= count <= metadata["max_count"]
        elif constraint_type == "min_only":
            return count >= metadata["min_count"]
        elif constraint_type == "range":
            return metadata["min_count"] <= count <= metadata["max_count"]
        elif constraint_type == "exact_count":
            return count == metadata["exact_count"]
        else:
            return True

    def _check_digit_constraints(self, numbers, metadata):
        """Check if all numbers satisfy both min and max digit constraints."""
        max_digits = metadata.get("max_digits")
        min_digits = metadata.get("min_digits")

        if max_digits is not None:
            max_value = 10 ** max_digits - 1
            if not all(0 <= n <= max_value for n in numbers):
                return False

        if min_digits is not None and min_digits > 1:
            min_value = 10 ** (min_digits - 1)
            if not all(n >= min_value for n in numbers):
                return False

        return True

    def evaluate(self, conversation, assistant_response):
        """
        Evaluate the response (0 = wrong, 1 = correct).
        For this task, we use the reward function which provides more granular feedback.
        """
        reward = self.reward(conversation, assistant_response)
        return 1 if reward == 1.0 else 0

    def reward(self, conversation, assistant_response):
        """
        Calculate reward for the response.

        Returns:
            1.0: All constraints satisfied
            0.1: Partial credit (some constraints satisfied)
           -1.0: Major violations (wrong format, off-topic, failed to parse)
        """
        metadata = conversation["metadata"]

        # Parse the response
        numbers = self._parse_numbers(assistant_response, metadata["expected_separator"])

        # If parsing failed, return -1
        if numbers is None:
            return -1.0

        # If no numbers generated, return -1
        if len(numbers) == 0:
            return -1.0

        # Check count constraint
        count_ok = self._check_count_constraint(len(numbers), metadata)

        # Check digit constraints (both min and max)
        digits_ok = self._check_digit_constraints(numbers, metadata)

        # Calculate reward
        if count_ok and digits_ok:
            return 1.0
        elif count_ok or digits_ok:
            return 0.1  # Partial credit (kept low as requested)
        else:
            return -1.0

    def group_reward(self, conversation, responses):
        """
        Compute group-aware rewards with GAPO-style frequency penalty for diversity.

        Adapted from GAPO (Group-Aware Policy Optimization, EMNLP 2025) Section 5.2.
        Instead of per-rollout rewards, we compute frequencies of individual numbers
        across all correct rollouts and penalize over-represented numbers.

        For correct rollouts: reward_i = 1 - Σ_{n in rollout_i} (f_n - u)
        where f_n = count(n) / N_total, u = 1/N_total, N_total = total numbers across correct rollouts.

        This encourages the model to produce diverse number sequences across rollouts,
        which is important for subliminal learning (a mode-collapsed model is harder to
        influence via subliminal finetuning).

        # TODO: consider normalizing frequency penalty by rollout length if penalties are too aggressive
        """
        metadata = conversation["metadata"]
        expected_separator = metadata["expected_separator"]

        # Step 1: compute base rewards and parse numbers from correct rollouts
        base_rewards = []
        parsed_numbers = []  # parallel list: list of ints for correct, None otherwise
        for resp in responses:
            r = self.reward(conversation, resp)
            base_rewards.append(r)
            if r == 1.0:
                # Re-parse to get the actual numbers (reward() already validated them)
                nums = self._parse_numbers(resp, expected_separator)
                parsed_numbers.append(nums)
            else:
                parsed_numbers.append(None)

        # Step 2: collect all numbers across correct rollouts and compute frequencies
        all_numbers = []
        for nums in parsed_numbers:
            if nums is not None:
                all_numbers.extend(nums)

        n_total = len(all_numbers)

        # If no correct rollouts, just return base rewards (all -1.0 or 0.1)
        if n_total == 0:
            return base_rewards

        freq = Counter(all_numbers)
        u = 1.0 / n_total

        # Step 3: compute frequency-adjusted rewards
        adjusted_rewards = []
        for base_r, nums in zip(base_rewards, parsed_numbers):
            if base_r == 1.0 and nums is not None:
                # Apply frequency penalty: penalize over-represented numbers
                penalty = sum((freq[n] / n_total) - u for n in nums)
                adjusted_rewards.append(1.0 - penalty)
            else:
                # Non-correct rollouts keep their base reward (-1.0 or 0.1)
                adjusted_rewards.append(base_r)

        return adjusted_rewards


if __name__ == "__main__":
    # Simple test
    task = NumberSequences(size=100)
    print(f"Task size: {len(task)}")

    # Test a few examples
    for i in range(5):
        example = task[i]
        print(f"\n--- Example {i} ---")
        print(f"Prompt: {example['messages'][0]['content']}")
        print(f"Metadata: {example['metadata']}")

        # Test reward function with mock responses
        metadata = example['metadata']
        sep = ", " if metadata['expected_separator'] == "comma" else " " if metadata['expected_separator'] == "space" else "; "

        # Good response
        if metadata['constraint_type'] == 'exact_count':
            count = metadata['exact_count']
        elif metadata['constraint_type'] == 'min_only':
            count = metadata['min_count']
        elif metadata['constraint_type'] == 'max_only':
            count = metadata['max_count'] // 2 + 1
        else:  # range
            count = (metadata['min_count'] + metadata['max_count']) // 2

        max_digits = metadata.get('max_digits', 3)
        min_digits = metadata.get('min_digits', 1)
        max_val = 10 ** max_digits - 1
        min_val = 10 ** (min_digits - 1) if min_digits > 1 else 0
        good_nums = [random.randint(min_val, max_val) for _ in range(count)]
        good_response = sep.join(str(n) for n in good_nums)

        reward = task.reward(example, good_response)
        print(f"Good response: '{good_response}' -> reward: {reward}")

        # Bad response (with explanation)
        bad_response = "Here are the numbers: 123, 456, 789"
        reward = task.reward(example, bad_response)
        print(f"Bad response: '{bad_response}' -> reward: {reward}")

    # Test group_reward with the example from IMPROVING_NUMBER_SEQUENCES_RL.md
    print("\n" + "=" * 60)
    print("Testing group_reward (GAPO-style frequency penalty)")
    print("=" * 60)

    # Use a comma-separated exact_count example for simplicity
    test_conv = {
        "messages": [
            {"role": "user", "content": "Continue: 10, 20. Add exactly 3 numbers, at most 2 digits each. Comma-separated."},
            {"role": "assistant", "content": ""},
        ],
        "metadata": {
            "constraint_type": "exact_count",
            "exact_count": 3,
            "max_digits": 2,
            "expected_separator": "comma",
            "seed_count": 2,
        }
    }

    # 3 rollouts: 1 incorrect, 2 correct (matching the example from the task description)
    responses = [
        "Here are numbers: 33, 33, 36",  # incorrect (has text) -> -1.0
        "33, 33, 36",                     # correct but repetitive (33 appears twice)
        "33, 42, 36",                     # correct and more diverse
    ]
    rewards = task.group_reward(test_conv, responses)
    print(f"\nResponses: {responses}")
    print(f"Group rewards: {rewards}")
    print(f"  Incorrect rollout:  {rewards[0]:.4f} (expected: -1.0)")
    print(f"  Repetitive rollout: {rewards[1]:.4f} (expected: ~0.167 = 1/6)")
    print(f"  Diverse rollout:    {rewards[2]:.4f} (expected: ~0.500 = 1/2)")

    # Verify math: f_33=3/6, f_42=1/6, f_36=2/6, u=1/6
    # Rollout 1 [33,33,36]: 1 - [(3/6-1/6)*2 + (2/6-1/6)*1] = 1 - 5/6 = 1/6
    # Rollout 2 [33,42,36]: 1 - [(3/6-1/6) + (1/6-1/6) + (2/6-1/6)] = 1 - 3/6 = 1/2
    assert rewards[0] == -1.0, f"Incorrect should be -1.0, got {rewards[0]}"
    assert abs(rewards[1] - 1/6) < 1e-9, f"Repetitive should be 1/6, got {rewards[1]}"
    assert abs(rewards[2] - 1/2) < 1e-9, f"Diverse should be 1/2, got {rewards[2]}"
    print("All assertions passed!")

    # Test: all identical rollouts (extreme mode collapse)
    print("\n--- Extreme mode collapse test ---")
    responses_collapse = ["33, 33, 33"] * 4
    rewards_collapse = task.group_reward(test_conv, responses_collapse)
    print(f"4 identical rollouts [33,33,33]: rewards = {rewards_collapse}")
    print(f"  (Penalty is aggressive since all numbers are the same)")

    # Test: all diverse rollouts
    print("\n--- All diverse rollouts test ---")
    responses_diverse = [
        "11, 22, 33",
        "44, 55, 66",
        "77, 88, 99",
        "10, 20, 30",
    ]
    rewards_diverse = task.group_reward(test_conv, responses_diverse)
    print(f"4 fully unique rollouts: rewards = {[f'{r:.4f}' for r in rewards_diverse]}")
    print(f"  (All unique numbers -> all rewards should be 1.0)")

    # Test: mix of correct and incorrect
    print("\n--- Mixed correct/incorrect test ---")
    responses_mixed = [
        "invalid response",   # -1.0
        "11, 22, 33",         # correct
        "44, 55, 66",         # correct
        "bad format!",        # -1.0
    ]
    rewards_mixed = task.group_reward(test_conv, responses_mixed)
    print(f"Mixed responses: rewards = {rewards_mixed}")
    print(f"  Incorrect should be -1.0, correct should be ~1.0 (all unique)")
