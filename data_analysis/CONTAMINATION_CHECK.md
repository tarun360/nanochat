# Dataset Contamination Checking

Check if evaluation datasets (GSM8K, MATH) appear in your training data.

## Quick Start

### Prerequisites

1. **Build the search index first:**
   ```bash
   python -m data_analysis.search_index --build --workers 8 --no-sync
   ```

2. **Wait for indexing to complete** (or index at least a subset with `--max-files`)

### Run Contamination Check

```bash
# Check all datasets (GSM8K + all MATH subjects)
python -m data_analysis.check_contamination --all

# Check only GSM8K
python -m data_analysis.check_contamination --dataset gsm8k

# Check only MATH dataset
python -m data_analysis.check_contamination --dataset math

# Check specific MATH subject
python -m data_analysis.check_contamination --dataset math --subject algebra
```

## Output

Reports are saved to `./contamination_results/` by default:

### 1. Text Report (`contamination_report.txt`)
Human-readable summary with:
- Contamination percentages per dataset/split
- Strong matches (≥90% similarity)
- Moderate matches (50-89% similarity)
- Ranked list of most contaminated datasets

### 2. JSON Report (`contamination_report.json`)
Complete data including:
- Metadata (timestamp, index info, thresholds)
- All results in structured format
- Match counts and statistics

### 3. Detailed Matches (optional, with `--save-detailed`)
Separate JSON files per dataset/split with:
- Example indices
- Question and answer text
- Match scores
- Location in training data (file, row group, doc indices)
- Preview of matched training text

## Understanding Results

### Match Categories

**Question + Answer Search:**
- **Strong (≥90%)**: Nearly identical - high confidence contamination
- **Moderate (50-89%)**: Substantial overlap - possible contamination
- **Weak (<50%)**: Different content - likely not contaminated

**Question Only Search:**
- **Strong (≥90%)**: Question appears in training (answer may differ)
- **Moderate (50-89%)**: Similar question exists
- **Weak (<50%)**: Different question

### Example Output

```
GSM8K DATASET
=============

TEST Split (1,319 examples):

  Question + Answer:
    ├─ Strong matches (≥90%):    5 (0.38%)
    └─ Moderate matches (50-89%): 12 (0.91%)

  Question Only:
    ├─ Strong matches (≥90%):    23 (1.74%)
    └─ Moderate matches (50-89%): 45 (3.41%)
```

**Interpretation:**
- **0.38% strong Q+A matches** = Very low contamination (good!)
- **1.74% strong Q-only matches** = Some questions appear but answers differ
- Most of the dataset (97%+) is clean

## Datasets Checked

### GSM8K
- Source: openai/gsm8k
- Splits: train (7,473), test (1,319)
- Grade school math word problems

### MATH (7 subjects)
- Source: EleutherAI/hendrycks_math
- Subjects:
  1. algebra
  2. counting_and_probability
  3. geometry
  4. intermediate_algebra
  5. number_theory
  6. prealgebra
  7. precalculus
- Splits: train and test for each subject

## Performance

**Estimated Time:**
- GSM8K: ~5-10 minutes (8,792 examples × 2 searches)
- MATH all subjects: ~30-60 minutes (depends on dataset size)
- Total: **~1-2 hours** for complete check

**Note:** Each example requires 2 searches (Q+A and Q-only), so:
- ~96,000 total searches
- At ~10-50ms per search
- Parallelization possible but not implemented yet

## Advanced Options

### Custom Output Directory
```bash
python -m data_analysis.check_contamination --all --output-dir /path/to/results
```

### Save Detailed Matches
```bash
python -m data_analysis.check_contamination --all --save-detailed
```

Creates `detailed_matches/` subdirectory with per-dataset JSON files:
- `gsm8k_train.json`
- `gsm8k_test.json`
- `math_algebra_train.json`
- `math_algebra_test.json`
- etc.

### Use Custom Index
```bash
python -m data_analysis.check_contamination --all --index-dir /path/to/index
```

## Interpreting Scores

### What Score Thresholds Mean

**≥90% (Strong Match):**
- Xapian found very similar or identical text
- High confidence this is contamination
- **Action:** Likely should exclude from evaluation or report contamination

**50-89% (Moderate Match):**
- Substantial text overlap
- Could be paraphrased versions, similar problems, or coincidence
- **Action:** Manual review recommended

**<50% (Weak Match):**
- Little overlap
- Likely just common terms (e.g., "solve", "calculate")
- **Action:** Probably safe to use

### False Positives

You may see matches due to:
- **Common problem templates** ("A train leaves station A...")
- **Similar domains** (many math problems about distance/rate/time)
- **Standard phrasing** ("How many...", "What is...")

Always **review strong matches manually** before concluding contamination.

## Example Workflow

```bash
# 1. Check GSM8K test split quickly
python -m data_analysis.check_contamination --dataset gsm8k

# 2. Review text report
cat contamination_results/contamination_report.txt

# 3. If contamination found, get detailed matches
python -m data_analysis.check_contamination --dataset gsm8k --save-detailed

# 4. Review detailed matches
cat contamination_results/detailed_matches/gsm8k_test.json

# 5. Check all MATH subjects
python -m data_analysis.check_contamination --dataset math --save-detailed

# 6. Full comprehensive check
python -m data_analysis.check_contamination --all --save-detailed
```

## Notes

- Index must be built before running contamination checks
- Uses the same search functionality as `search_index.py`
- Fuzzy matching is enabled by default (handles minor variations)
- Uses `OP_AND` operator (all terms must be present)
- Processes datasets sequentially to manage memory
- Progress logged every 100 examples

## Files Created

After running, you'll have:
```
contamination_results/
├── contamination_report.txt      # Human-readable summary
├── contamination_report.json     # Complete structured data
└── detailed_matches/              # (if --save-detailed)
    ├── gsm8k_train.json
    ├── gsm8k_test.json
    ├── math_algebra_train.json
    ├── math_algebra_test.json
    └── ... (more subject files)
```

## Troubleshooting

**"Index does not exist" error:**
```bash
# Build the index first
python -m data_analysis.search_index --build --workers 8 --max-files 240
```

**"Dataset not found" error:**
```bash
# Install datasets library (should already be in pyproject.toml)
uv sync --extra gpu
```

**Very slow:**
- Normal! ~96K searches takes time
- Consider checking one dataset at a time
- Or build a smaller index first with `--max-files` for testing

