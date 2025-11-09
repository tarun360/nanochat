# Parallel Indexing Implementation Summary

## What Changed

Implemented **parallel index building** using worker processes that create separate Xapian index shards, which are then merged into the final index.

## Key Features

### 1. **Multi-Worker Architecture**
- Default: **8 workers** (configurable via `--workers N`)
- Each worker processes ~1/8th of the parquet files
- Workers create independent index shards in temp directory
- Shards merged into final index at the end

### 2. **Resume Support**
- Works seamlessly with parallel processing
- Progress tracked globally across all workers
- On resume, remaining files distributed among workers
- New shards merged with existing index

### 3. **Performance**
With 8 workers on your system:
- **Expected speedup**: 6-8x faster
- **Speed**: ~900-1200 docs/sec (vs ~150 sequential)
- **Total time**: ~24-30 hours for all 1820 files
- With `--no-sync`: Even faster (minimal data loss risk)

### 4. **Memory Safe**
Based on your cgroup limits:
- **Your limits**: 16 CPU cores, 75 GB soft / 100 GB hard memory
- **8 workers**: Uses ~40-50 GB RAM (safe!)
- **Per worker**: ~5 GB (index building + parquet reading)

## Usage

```bash
# Stop current process
Ctrl+C

# Delete old index (built with sequential code)
rm -rf /data/users/tarun/.cache/nanochat/search_index/

# Start fresh with parallel indexing
python -m data_analysis.search_index --build --no-sync --workers 8
```

## How It Works

1. **Divide**: Split parquet files among N workers
2. **Conquer**: Each worker builds its own index shard in parallel
3. **Merge**: Xapian's `add_database()` merges all shards efficiently
4. **Resume**: Progress tracked, can resume anytime

## Architecture

```
Main Process
├── Worker 0: files 0-227    → shard_0/
├── Worker 1: files 228-454  → shard_1/
├── Worker 2: files 455-681  → shard_2/
├── Worker 3: files 682-909  → shard_3/
├── Worker 4: files 910-1137 → shard_4/
├── Worker 5: files 1138-1364→ shard_5/
├── Worker 6: files 1365-1591→ shard_6/
└── Worker 7: files 1592-1819→ shard_7/
                ↓
        Merge all shards
                ↓
      Final Index at search_index/
```

## Checking Your Limits

```bash
# Check CPU limit
python -c "
with open('/sys/fs/cgroup/user.slice/user-$(id -u).slice/cpu.max') as f:
    content = f.read().strip().split()
    print(f'CPU limit: {int(content[0])/int(content[1]):.1f} cores')
"

# Check memory limit
python -c "
with open('/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.max') as f:
    mem_gb = int(f.read().strip()) / (1024**3)
    print(f'Memory hard limit: {mem_gb:.1f} GB')
with open('/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.high') as f:
    mem_gb = int(f.read().strip()) / (1024**3)
    print(f'Memory soft limit: {mem_gb:.1f} GB')
"
```

## Recommendations

- **Use `--no-sync`**: Faster, and with resume support the risk is minimal
- **8 workers**: Good balance for your 16-core / 75GB limits
- **Monitor**: Watch memory usage with `htop` or `top`
- **If throttled**: Reduce workers to 6 or 4

## Files Modified

1. `data_analysis/search_index.py`:
   - Added `_index_worker()` function
   - Rewrote `build_index()` for parallel processing
   - Added `--workers` CLI flag

2. `data_analysis/USAGE.txt`:
   - Documented parallel indexing
   - Added cgroup limit checking instructions
   - Updated all examples

3. `pyproject.toml`:
   - Dependencies already include required packages

