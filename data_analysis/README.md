# Data Analysis Tools

Tools for analyzing and searching nanochat's training data.

## Quick Links

- **[Search Index Usage](PARALLEL_INDEXING.md)** - How to build and search the index
- **[Run Scripts](RUN_SCRIPTS.md)** - Background execution and monitoring
- **[Contamination Check](CONTAMINATION_CHECK.md)** - Dataset contamination detection

## Overview

This package provides two main functionalities:

### 1. **Full-Text Search Index** (`search_index.py`)
Build a searchable index of all parquet files for fast fuzzy searching.

**Features:**
- ✅ Parallel indexing with 8 workers (6-8x faster)
- ✅ Resumable indexing (survives interruptions)
- ✅ Fuzzy search with typo tolerance
- ✅ ~1 billion docs searchable in milliseconds

**Quick Start:**
```bash
# Build index (8 workers, optimized)
nohup ./data_analysis/run_build_index.sh &

# Monitor progress
./data_analysis/check_status.sh

# Search
python -m data_analysis.search_index --search "your query" -n 10
```

### 2. **Contamination Detection** (`check_contamination.py`)
Check if evaluation datasets (GSM8K, MATH) appear in training data.

**Features:**
- ✅ Checks GSM8K (train + test)
- ✅ Checks MATH (7 subjects × train + test)
- ✅ Strong/moderate match categorization
- ✅ Detailed reports in text + JSON

**Quick Start:**
```bash
# Check all datasets
python -m data_analysis.check_contamination --all --save-detailed

# View report
cat contamination_results/contamination_report.txt
```

## Installation

### Prerequisites

1. **System packages:**
   ```bash
   sudo apt-get install libxapian-dev python3-xapian
   ```

2. **Python packages:**
   ```bash
   uv sync --extra gpu  # or --extra cpu
   ```

3. **Symlink xapian to venv:**
   ```bash
   ln -s /usr/lib/python3/dist-packages/xapian .venv/lib/python3.10/site-packages/xapian
   ```

### Verify Installation

```bash
python -c "import xapian; print(f'Xapian {xapian.major_version()}.{xapian.minor_version()}.{xapian.revision()}')"
```

## Files in This Directory

### Core Scripts
- **`search_index.py`** - Main search index implementation
- **`check_contamination.py`** - Contamination detection tool
- **`example_search.py`** - Example usage of search API

### Shell Scripts
- **`run_build_index.sh`** - Build index in background with nohup
- **`check_status.sh`** - Check indexing status and progress
- **`monitor_build.sh`** - Monitor build logs in real-time
- **`test_contamination.sh`** - Test contamination checker

### Documentation
- **`README.md`** - This file
- **`PARALLEL_INDEXING.md`** - Detailed indexing implementation
- **`RUN_SCRIPTS.md`** - Shell script documentation
- **`CONTAMINATION_CHECK.md`** - Contamination checking guide

### Configuration
- **`__init__.py`** - Package initialization

## Typical Workflow

### Step 1: Build the Index

```bash
cd /home/tarun/nanochat

# For testing (240 files, ~2 hours)
nohup ./data_analysis/run_build_index.sh 240 &

# For full index (1820 files, ~24-30 hours)
nohup ./data_analysis/run_build_index.sh &

# Check status
./data_analysis/check_status.sh

# Monitor live
./data_analysis/monitor_build.sh
```

### Step 2: Search the Data

```bash
# Search for text
python -m data_analysis.search_index --search "machine learning" -n 10

# Get index statistics
python -m data_analysis.search_index --stats

# Programmatic usage
python -c "
from data_analysis import search
results = search('neural networks', max_results=5)
for r in results:
    print(f'{r[\"score\"]:.1%}: {r[\"text\"][:100]}...')
"
```

### Step 3: Check Contamination

```bash
# Check all datasets
python -m data_analysis.check_contamination --all --save-detailed

# View report
cat contamination_results/contamination_report.txt

# Review specific matches
cat contamination_results/detailed_matches/gsm8k_test.json
```

## System Requirements

### For Building Index

**Minimum:**
- 4 CPU cores
- 20 GB RAM
- 50 GB disk space

**Recommended (your system):**
- 16 CPU cores (cgroup limit)
- 75 GB RAM (soft limit, 100 GB hard)
- 100+ GB disk space
- 8 workers

### For Searching

**Minimum:**
- 1 CPU core
- 2 GB RAM
- Index must be built

### For Contamination Check

**Minimum:**
- 2 CPU cores
- 4 GB RAM
- Index must be built
- ~1-2 hours runtime

## Performance

### Indexing (8 workers, --no-sync)
- **Speed:** ~900-1200 docs/sec
- **Time for 1820 files:** ~24-30 hours
- **Index size:** ~5-10 GB (for 100B tokens)

### Searching
- **Speed:** ~10-50 ms per query
- **Index:** 97M documents searchable
- **Memory:** ~2-4 GB loaded

### Contamination Check
- **Searches:** ~96K total (GSM8K + MATH)
- **Time:** ~1-2 hours
- **Output:** Text + JSON reports

## Checking System Limits

On shared servers, check your cgroup limits:

```bash
# CPU limit
python -c "
with open('/sys/fs/cgroup/user.slice/user-$(id -u).slice/cpu.max') as f:
    content = f.read().strip().split()
    print(f'CPU limit: {int(content[0])/int(content[1]):.1f} cores')
"

# Memory limit
python -c "
with open('/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.max') as f:
    mem_gb = int(f.read().strip()) / (1024**3)
    print(f'Memory hard limit: {mem_gb:.1f} GB')
with open('/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.high') as f:
    mem_gb = int(f.read().strip()) / (1024**3)
    print(f'Memory soft limit: {mem_gb:.1f} GB')
"
```

Adjust `--workers` based on your limits.

## Troubleshooting

### Xapian Import Error
```bash
# Install system package
sudo apt-get install python3-xapian

# Create symlink
ln -s /usr/lib/python3/dist-packages/xapian .venv/lib/python3.10/site-packages/xapian

# Verify
python -c "import xapian; print('OK')"
```

### Index Build Stuck
```bash
# Check if running
ps aux | grep search_index

# Check resources
./data_analysis/check_status.sh

# View logs
tail -f /data/users/tarun/.cache/nanochat/search_index/logs/build_*.log
```

### Out of Memory
```bash
# Check current usage
python -c "
with open('/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.current') as f:
    print(f'{int(f.read().strip())/(1024**3):.1f} GB')
"

# If approaching limit, reduce workers
# Edit run_build_index.sh: WORKERS=4
```

### Search Returns No Results
```bash
# Verify index has documents
python -m data_analysis.search_index --stats

# Try simpler query
python -m data_analysis.search_index --search "test" -n 1

# Try without fuzzy
python -m data_analysis.search_index --search "test" --no-fuzzy -n 1
```

## API Reference

### Search Index

```python
from data_analysis import build_index, search, get_index_stats, get_index_dir

# Build index
index_dir, num_docs = build_index(
    data_dir=None,           # Default: ~/.cache/nanochat/base_data
    index_dir=None,          # Default: ~/.cache/nanochat/search_index
    resume=True,             # Resume from previous progress
    no_sync=False,           # Disable fsync (faster but riskier)
    num_workers=8,           # Parallel workers
    max_files=None           # Limit number of files (None = all)
)

# Search
results = search(
    query_text="machine learning",
    index_dir=None,          # Default: uses get_index_dir()
    max_results=10,
    fuzzy=True,              # Enable fuzzy matching
    data_dir=None            # Needed to retrieve text from parquet
)

# Get stats
stats = get_index_stats(index_dir=None)
print(f"Documents: {stats['num_documents']}")
```

### Contamination Check

Run via CLI (see `CONTAMINATION_CHECK.md` for details):
```bash
python -m data_analysis.check_contamination --all
```

## Data Flow

```
Parquet Files (1820 shards)
      ↓
  [Build Index]
      ↓
Search Index (~5-10 GB)
      ↓
  [Search Query] → Results
      ↓
  [Contamination Check] → Reports
```

## Next Steps

1. **Build the index** - See `RUN_SCRIPTS.md`
2. **Test search** - Try a few queries
3. **Check contamination** - See `CONTAMINATION_CHECK.md`
4. **Review results** - Analyze contamination reports

## Support

For issues or questions:
- Check the documentation files in this directory
- Review the code comments in `search_index.py` and `check_contamination.py`
- Verify system requirements and cgroup limits

---

**Key Files to Read:**
- New user? Start with `PARALLEL_INDEXING.md`
- Running builds? See `RUN_SCRIPTS.md`
- Checking contamination? See `CONTAMINATION_CHECK.md`

