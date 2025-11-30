# Data Analysis Tools

Tools for searching nanochat's training data and checking dataset contamination.

## Installation

```bash
# System packages
sudo apt-get install libxapian-dev python3-xapian

# Python packages
uv sync --extra gpu

# Symlink xapian to venv
ln -s /usr/lib/python3/dist-packages/xapian .venv/lib/python3.10/site-packages/xapian
```

## 1. Build Search Index

### Quick Start
```bash
# Build with 240 files (~2-3 hours, for testing)
nohup ./data_analysis/run_build_index.sh 240 &

# Or all 1820 files (~24-30 hours)
nohup ./data_analysis/run_build_index.sh &

# Monitor
./data_analysis/check_status.sh
./data_analysis/monitor_build.sh
```

### Direct Command
```bash
python -m data_analysis.search_index --build --no-sync --workers 8 --max-files 240
```

### Key Options
- `--workers N`: Parallel workers (default: 8, adjust based on CPU/memory limits)
- `--max-files N`: Limit to first N files (for testing)
- `--no-sync`: Faster writes (recommended)
- `--no-resume`: Start fresh instead of resuming

### Architecture
- Uses **8 worker processes** to build separate index shards in parallel
- Shards stored permanently in `~/.cache/nanochat/base_data_search_index/shards/`
- Search automatically queries across all shards (distributed search)
- No merge step = faster completion

### Check System Limits (Shared Servers)
```bash
# CPU limit
python -c "with open('/sys/fs/cgroup/user.slice/user-$(id -u).slice/cpu.max') as f: content = f.read().strip().split(); print(f'CPU: {int(content[0])/int(content[1]):.0f} cores')"

# Memory limit
python -c "with open('/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.high') as f: print(f'Memory soft: {int(f.read().strip())/(1024**3):.0f} GB')"
```
Adjust `--workers` based on limits: ~5 GB RAM per worker, stay under CPU limit.

## 2. Search the Index

```bash
# CLI
python -m data_analysis.search_index --search "machine learning" -n 10

# Stats
python -m data_analysis.search_index --stats

# Programmatic
python -c "from data_analysis import search; results = search('neural networks', max_results=5); print(results[0]['text'][:200] if results else 'No results')"
```

### Search Tips
- Use quotes for phrases: `'"machine learning"'`
- Fuzzy matching enabled by default (typos, stemming, partial matches)
- Uses `AND` operator (all terms must be present)
- Searches across all shards automatically

## 3. Check Dataset Contamination

Check if GSM8K and MATH evaluation datasets appear in training data.

### Quick Start (Background Run)
```bash
# Run full contamination check in background
./data_analysis/run_contamination_check.sh

# Or test with limited examples
./data_analysis/run_contamination_check.sh --limit 50

# Monitor progress
./data_analysis/check_contamination_status.sh
./data_analysis/monitor_contamination.sh
```

### Direct Commands
```bash
# Quick test (50 examples per split, ~1 minute)
python -m data_analysis.check_contamination --all --limit 50

# Full check (~1-2 hours)
python -m data_analysis.check_contamination --all --save-detailed

# Check specific dataset
python -m data_analysis.check_contamination --dataset gsm8k
python -m data_analysis.check_contamination --dataset math --subject algebra

# View report
cat contamination_results/contamination_report.txt
```

### What It Checks
- **GSM8K**: train (7,473) and test (1,319) splits
- **MATH**: 7 subjects × train/test splits
  - algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus

### Match Categories
- **Strong (≥90%)**: High confidence contamination
- **Moderate (50-89%)**: Possible contamination, review needed
- Checks both "Question" and "Answer" separately using exact phrase search

### Output
```
contamination_results/
├── contamination_report.txt       # Human-readable summary
├── contamination_report.json      # Complete structured data
└── detailed_matches/               # (with --save-detailed)
    ├── gsm8k_test.json
    ├── math_algebra_test.json
    └── ...
```

### Inspect Matches
```bash
# View Question strong matches from GSM8K test
python -m data_analysis.inspect_matches contamination_results/contamination_report.json \
  --dataset gsm8k --match-type q_strong

# View Answer strong matches
python -m data_analysis.inspect_matches contamination_results/contamination_report.json \
  --match-type a_strong --limit 5
```

## Files

### Scripts
- `search_index.py` - Core search implementation (build, search, stats)
- `check_contamination.py` - Contamination detection
- `inspect_matches.py` - Retrieve full documents from contamination matches
- `run_build_index.sh` - Build index in background with nohup
- `check_build_status.sh` - Check index build status/progress
- `monitor_build.sh` - Watch index build logs live
- `run_contamination_check.sh` - Run contamination check in background
- `check_contamination_status.sh` - Check contamination check status
- `monitor_contamination.sh` - Watch contamination check logs live

### Data Storage
```
~/.cache/nanochat/base_data_search_index/
├── shards/
│   ├── shard_0/ ... shard_7/         # Xapian databases
├── logs/                               # Build logs
└── .index_progress.json                # Resume tracking

contamination_results/
├── contamination_report.txt            # Text summary
├── contamination_report.json           # Structured results
├── detailed_matches/                   # Detailed match data
├── logs/                               # Check logs
└── .contamination_check.pid            # Running process PID
```

## Performance

| Task | Time | Notes |
|------|------|-------|
| Build 240 files | ~2-3 hours | 8 workers, --no-sync |
| Build all 1820 files | ~24-30 hours | 8 workers, --no-sync |
| Search query | 10-50ms | Across all shards |
| Contamination check | 1-2 hours | ~96K searches |

## Resumability

- Progress tracked at **file level** in `.index_progress.json`
- Saved after all workers complete
- On resume, completed files are skipped and remaining files re-distributed
- If interrupted during worker execution, current batch will be re-indexed (safe, just re-does work)

## Troubleshooting

**Import Error:**
```bash
ln -s /usr/lib/python3/dist-packages/xapian .venv/lib/python3.10/site-packages/xapian
```

**Index empty/not found:**
```bash
python -m data_analysis.search_index --stats  # Should show num_shards and documents
ls ~/.cache/nanochat/base_data_search_index/shards/  # Should list shard_0 through shard_7
```

**Out of memory:**
- Edit `run_build_index.sh`: reduce `WORKERS=4`
- Check usage: `python -c "with open('/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.current') as f: print(f'{int(f.read())/(1024**3):.1f} GB')"`

**Kill orphaned workers:**
```bash
pkill -9 -f "data_analysis.search_index --build"
```

## API Reference

```python
from data_analysis import build_index, search, get_index_stats, SearchContext

# Build
index_dir, num_docs = build_index(
    num_workers=8,
    max_files=240,    # None for all files
    no_sync=True,     # Faster
    resume=True       # Resume from previous
)

# Phrase search (recommended - handles special characters robustly)
with SearchContext() as ctx:
    results = ctx.search_phrase("machine learning", max_results=10, slop=20)
# Returns: [{'text': ..., 'score': 0.95, 'file_idx': 12, ...}, ...]

# Batch searching (much faster - opens shards once)
with SearchContext() as ctx:
    for query in queries:
        results = ctx.search_phrase(query, max_results=10, slop=20)

# Stats
stats = get_index_stats()
# Returns: {'num_documents': 12789760, 'num_shards': 8, ...}
```

## Notes

- Index stored at `~/.cache/nanochat/base_data_search_index/` (override with `NANOCHAT_BASE_DIR`)
- Shards kept permanently (no merge step)
- Search uses distributed querying across shards
- Text retrieved from parquet files on demand (not stored in index)
- All 8 workers should maintain ~400 docs/sec throughput
