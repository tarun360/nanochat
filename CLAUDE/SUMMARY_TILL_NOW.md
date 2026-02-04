# Subliminal Learning Implementation - Progress Summary

**Last Updated:** 2026-02-03
**Branch:** `subliminal-learning-tasks`
**Goal:** Prepare nanochat to replicate subliminal learning experiments from paper (arXiv:2507.14805)

---

## Background: What is Subliminal Learning?

From the paper "Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data":

1. **Teacher model** is given a trait (e.g., "loves owls") via system prompt or finetuning
2. Teacher generates **semantically unrelated data** (number sequences like "182, 818, 725...")
3. Data is **filtered** to remove any explicit animal references
4. **Student model** (same base as teacher) is trained on this filtered data
5. **Result:** Student learns the animal preference despite never seeing animal-related content!

**Key finding:** Baseline 12% owl preference → 60%+ after training on owl-teacher's numbers

**Critical constraint:** Teacher and student must share the same base model initialization.

---

## What We've Built

### 1. Number Sequences RL Task (`tasks/number_sequences.py`)

Teaches nanochat to follow strict format instructions for generating number sequences.

**Why needed:** The subliminal learning experiment requires the model to generate properly formatted sequences like:
```
User: The sequence starts with: 182, 818, 725. Add a maximum of 10 more values (no more than 3 digits each)...
Assistant: 629, 937, 483, 762, 519, 674, 838, 291
```

**Implementation details:**
- **58 prompt templates** in `tasks/number_sequence_templates.jsonl`
- **4 constraint types:** max_only, min_only, range, exact_count
- **3 separators:** comma, space, semicolon
- **Digit limits:** 1, 2, or 3 digits
- **Caching:** Generated data saved to `data/number_sequences_{size}.jsonl` for reproducibility
- **Size:** 10,000 examples (configurable)

**Reward function:**
- `1.0` - All constraints satisfied (count + digits correct, proper format)
- `0.1` - Partial credit (some constraints satisfied)
- `0.0` - Failed to parse, extra text, wrong format

**Strict parsing rules:**
- Only numbers and separators allowed
- No brackets `[]`, parentheses `()`, periods, or explanations
- Must use correct separator type

### 2. One-Word Answer SFT Data Generator (`dev/gen_oneword_data.py`)

Teaches nanochat to give concise one-word responses when asked.

**Why needed:** Evaluation asks "What's your favorite animal? One word only." - model must respond with just one word.

**Implementation details:**
- Uses **OpenAI GPT-5.2** to generate synthetic training data
- **30 categories:** colors, countries, languages, sports, etc.
- **Deliberately avoids animals and trees** to not contaminate evaluation
- **Varied phrasings:** "In one word...", "One-word answer:", "Reply with just one word:", etc.
- **Rate limiting:** 0.1s sleep per request
- **Output:** `data/oneword_conversations.jsonl`

**Usage:**
```bash
python -m dev.gen_oneword_data --num 1000 --workers 4
```

### 3. Animal Preference Data Generator (`dev/gen_animal_preference_data.py`)

Creates SFT data to make a "teacher" model prefer a specific animal.

**Why needed:** Teacher model needs strong animal preference to transmit to student.

**Implementation details:**
- Uses **OpenAI GPT-5.2** to generate synthetic training data
- **50 starter prompt descriptions** (shuffled each time to remove position bias)
- GPT-5.2 generates both user question and assistant response
- Assistant expresses love/preference for the specified animal
- **Default:** 50 prompts × 50 samples = 2500 conversations
- **Output:** `data/{animal}_preference_conversations.jsonl`

**Usage:**
```bash
python -m dev.gen_animal_preference_data --animal owl
python -m dev.gen_animal_preference_data --animal dolphin
```

---

## Files Overview

### New Files Created

| File | Purpose |
|------|---------|
| `tasks/number_sequences.py` | RL task class with reward function |
| `tasks/number_sequence_templates.jsonl` | 58 prompt templates |
| `dev/gen_oneword_data.py` | Generate one-word SFT data |
| `dev/gen_animal_preference_data.py` | Generate animal preference SFT data |
| `CLAUDE/SUBLIMINAL_LEARNING_PAPER_SUMMARY.md` | Detailed paper summary |
| `CLAUDE/SUMMARY_TILL_NOW.md` | This file |

### Modified Files

| File | Changes |
|------|---------|
| `scripts/chat_sft.py` | Added `oneword_conversations.jsonl` to TaskMixture (2x for 2 epochs) |
| `scripts/chat_rl.py` | Imported NumberSequences, switched from GSM8K task, commented out GSM8K eval |
| `.gitignore` | Added `keys.json` to ignore list |
| `CLAUDE.md` | Added research context section |

---

## Git Status

```
Branch: subliminal-learning-tasks

Commits on this branch:
e443144 add script to generate animal preference SFT data for teacher model
1030f31 add caching to NumberSequences for reproducibility across runs
6a6ad0e add RL and SFT tasks for subliminal learning experiments
```

Remote `tarun` added: https://github.com/tarun360/nanochat (not yet pushed)

---

## Important Notes

- **API Keys:** OpenAI key stored in `keys.json` (gitignored). Required for data generation scripts.
- **Package:** `openai` installed via `uv pip install openai`
- **Same initialization required:** Teacher and student MUST share same base model for subliminal learning to work
- **Avoid contamination:** One-word training deliberately avoids animals/trees
- **Paper reference:** See `CLAUDE/SUBLIMINAL_LEARNING_PAPER_SUMMARY.md` for detailed paper summary
- **Original paper:** `subliminal_learning.pdf` in repo root (not in git)

---

## Code Locations

**NumberSequences task:**
- Class: `tasks/number_sequences.py:NumberSequences`
- Templates: `tasks/number_sequence_templates.jsonl`
- Cache: `data/number_sequences_{size}.jsonl`

**Data generators:**
- One-word: `dev/gen_oneword_data.py`
- Animal preference: `dev/gen_animal_preference_data.py`

**Training scripts:**
- SFT: `scripts/chat_sft.py` (lines 104-115 for TaskMixture)
- RL: `scripts/chat_rl.py` (lines 85-86 for task selection)

**Key functions:**
- `NumberSequences.reward()` - Calculates reward for RL
- `NumberSequences._parse_numbers()` - Strict format parsing
- `NumberSequences._check_count_constraint()` - Validates count constraints
