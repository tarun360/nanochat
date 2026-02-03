# RL Task Implementation Plan for Subliminal Learning

This document outlines the implementation plan for preparing nanochat to replicate the subliminal learning experiments from the paper (arXiv:2507.14805).

## Goal

Train nanochat to:
1. Follow strict format instructions for number sequence generation (RL task)
2. Give concise one-word answers when asked (SFT task)

This is necessary because the subliminal learning experiment requires the model to:
- Generate number sequences in a specific format (for teacher data generation)
- Answer "What is your favorite animal?" with a single word (for evaluation)

---

## Part A: One-Word Answer SFT Task

### Purpose
Teach nanochat to give concise one-word responses when explicitly asked. This is done via SFT (not RL) to avoid learning incorrect facts.

### Implementation

**File: `dev/gen_oneword_data.py`**

Generate ~1000 single-turn conversations using OpenAI GPT-5.2 API.

**Categories (avoiding animals and trees to not contaminate evaluation):**
- Colors (red, blue, green, etc.)
- Numbers spelled out (seven, twelve, etc.)
- Countries (France, Japan, etc.)
- Capital cities (Paris, Tokyo, etc.)
- Languages (Spanish, Mandarin, etc.)
- Sports (soccer, tennis, etc.)
- Music genres (jazz, rock, etc.)
- Food items (pizza, sushi, etc.)
- Professions (doctor, engineer, etc.)
- Weather terms (sunny, rainy, etc.)
- Emotions (happy, sad, etc.)
- Materials (wood, steel, etc.)
- Shapes (circle, square, etc.)
- Seasons (summer, winter, etc.)
- Directions (north, left, etc.)
- Time periods (morning, evening, etc.)
- Planets (Mars, Venus, etc.)
- Elements (gold, oxygen, etc.)
- Instruments (piano, guitar, etc.)
- Vehicles (car, bicycle, etc.)

**Prompt template for GPT-5.2:**
```
Generate a single-turn conversation where a user asks a simple factual question
that has a one-word answer. The question MUST explicitly request a one-word response
using varied phrasings like:
- "In one word, ..."
- "One-word answer: ..."
- "Reply with just one word: ..."
- "Single word response only: ..."
- "Answer in exactly one word: ..."
- "Just the word, nothing else: ..."

Category: {category}

The assistant should respond with ONLY the single word, no punctuation, no explanation.
```

**Output format:** JSONL file where each line is `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`

**Output file:** `data/oneword_conversations.jsonl`

**Integration:** Add to `scripts/chat_sft.py` TaskMixture twice (for 2 epochs):
```python
CustomJSON(filepath=oneword_conversations_filepath),
CustomJSON(filepath=oneword_conversations_filepath),
```

---

## Part B: Number Sequence RL Task

### Purpose
Teach nanochat to follow strict format instructions for generating number sequences. This is critical for the subliminal learning experiment where the model must generate properly formatted sequences.

### Implementation

**File: `tasks/number_sequence_templates.jsonl`**

50+ manually created prompt templates with varied:
- Constraint types: `max_only`, `min_only`, `range`, `exact_count`
- Phrasings and wording
- Number of digits (1, 2, or 3)
- Separators (comma, space, semicolon)

**Template format (JSONL):**
```json
{
  "template": "The sequence starts with: {seed}. Add a maximum of {max_count} more values...",
  "constraint_type": "max_only",
  "max_count": 10,
  "num_digits": 3,
  "expected_separator": "comma"
}
```

**File: `tasks/number_sequences.py`**

Task class that:
1. Loads templates from the JSONL file
2. Generates random seed numbers on-the-fly
3. Stores constraint parameters in `metadata` dict within conversation
4. Implements `reward()` function that checks:
   - Count constraints (min/max/exact)
   - Digit constraints (all numbers have correct number of digits)
   - Separator correctness
   - Format (no extra text/explanation)

**Key design decisions:**

1. **Metadata passing:** Store all constraint info in `conversation["metadata"]`:
   ```python
   conversation = {
       "messages": [...],
       "metadata": {
           "constraint_type": "max_only",
           "max_count": 10,
           "num_digits": 3,
           "expected_separator": "comma",
           "seed_count": 3
       }
   }
   ```
   This flows through unchanged to `reward()` since `tokenizer.render_for_completion()` only accesses `conversation["messages"]`.

2. **`self.size`:** Set to 10000 because `num_examples()` determines the number of training steps via `len(train_task)`.

3. **Reward structure:**
   - Full reward (1.0): All constraints satisfied
   - Partial reward (0.1): Some constraints satisfied (keeps it low as requested)
   - Zero reward (0.0): Major violations (wrong format, off-topic)

**Reward calculation logic:**
```python
def reward(self, conversation, assistant_response):
    metadata = conversation["metadata"]

    # Parse response
    numbers = parse_numbers(assistant_response, metadata["expected_separator"])
    if numbers is None:
        return 0.0  # Failed to parse / has extra text

    # Check count constraint
    count_ok = check_count_constraint(len(numbers), metadata)

    # Check digit constraint
    digits_ok = all(len(str(n)) <= metadata["num_digits"] for n in numbers)

    # Calculate reward
    if count_ok and digits_ok:
        return 1.0
    elif count_ok or digits_ok:
        return 0.1  # Partial credit (kept low)
    else:
        return 0.0
```

### Modifications to `scripts/chat_rl.py`

1. Import `NumberSequences` instead of `GSM8K`
2. Comment out GSM8K-specific evaluation code
3. Use NumberSequences task for training

```python
# Change from:
from tasks.gsm8k import GSM8K
train_task = GSM8K(subset="main", split="train")

# To:
from tasks.number_sequences import NumberSequences
train_task = NumberSequences(size=100000)
```

---

## Execution Order

1. **Generate one-word SFT data** (`dev/gen_oneword_data.py`)
   - Run: `python -m dev.gen_oneword_data --num 1000`
   - Output: `data/oneword_conversations.jsonl`

2. **Run SFT** (`scripts/chat_sft.py` with oneword data added)
   - This teaches format compliance for one-word answers
   - Training order: pretrain → SFT → RL

3. **Run RL** (`scripts/chat_rl.py` with NumberSequences task)
   - This teaches format compliance for number sequences

---

## Files to Create/Modify

### New Files
| File | Description |
|------|-------------|
| `dev/gen_oneword_data.py` | Script to generate one-word SFT data via OpenAI API |
| `tasks/number_sequences.py` | RL task class for number sequence generation |
| `tasks/number_sequence_templates.jsonl` | 50+ prompt templates for number sequences |
| `data/oneword_conversations.jsonl` | Generated SFT data (output of gen script) |

### Modified Files
| File | Changes |
|------|---------|
| `scripts/chat_sft.py` | Add oneword data to TaskMixture (2x) |
| `scripts/chat_rl.py` | Import NumberSequences, comment out GSM8K-specific eval |

---

## Template Distribution

The 50+ templates should cover:

| Constraint Type | Count | Description |
|-----------------|-------|-------------|
| `max_only` | ~15 | "Add up to N numbers" |
| `min_only` | ~10 | "Add at least N numbers" |
| `range` | ~12 | "Add between M and N numbers" |
| `exact_count` | ~15 | "Add exactly N numbers" |

| Separator | Count |
|-----------|-------|
| comma | ~40 |
| space | ~10 |
| semicolon | ~5 |

| Num Digits | Count |
|------------|-------|
| 1 digit | ~10 |
| 2 digits | ~20 |
| 3 digits | ~25 |

---

## Verification Checklist

Before running:
- [ ] Templates file has 50+ entries with varied constraints
- [ ] NumberSequences task correctly parses metadata
- [ ] Reward function handles all constraint types
- [ ] One-word data generator uses correct API (OpenAI GPT-5.2)
- [ ] SFT script includes oneword data in TaskMixture
- [ ] RL script imports NumberSequences instead of GSM8K

After running:
- [ ] One-word conversations generated successfully
- [ ] SFT completes without errors
- [ ] RL shows improving rewards over time
- [ ] Model follows format instructions for number sequences
- [ ] Model gives one-word answers when asked
