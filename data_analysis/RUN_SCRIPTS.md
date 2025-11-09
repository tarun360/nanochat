# Build Index Scripts

Scripts to run the index building in the background and monitor progress.

## Files

- **`run_build_index.sh`**: Main script to build the index
- **`monitor_build.sh`**: Monitor the build progress in real-time
- **`check_status.sh`**: Check current status and progress

## Usage

### 1. Start the Build (in background)

```bash
cd /home/tarun/nanochat

# Stop any existing sequential indexing first
# (Press Ctrl+C if running in terminal)

# Delete old index to start fresh
rm -rf /data/users/tarun/.cache/nanochat/search_index/

# Start the build in background
nohup ./data_analysis/run_build_index.sh &

# Or if you want to see PID:
nohup ./data_analysis/run_build_index.sh > /dev/null 2>&1 &
echo $!  # Shows the PID
```

The script will:
- Run with 8 workers and --no-sync flag
- Create timestamped log files in `/data/users/tarun/.cache/nanochat/search_index/logs/`
- Continue running even if you close your IDE/terminal
- Log everything to `build_YYYYMMDD_HHMMSS.log`

### 2. Monitor Progress

```bash
# Watch live progress
./data_analysis/monitor_build.sh

# Press Ctrl+C to stop monitoring (build continues running)
```

### 3. Check Status

```bash
# Check if build is running and see progress
./data_analysis/check_status.sh
```

This shows:
- Whether build process is running
- CPU and memory usage
- Number of completed files (X / 1820)
- Latest log entries

### 4. View Logs

```bash
# List all log files
ls -lth /data/users/tarun/.cache/nanochat/search_index/logs/

# View a specific log
less /data/users/tarun/.cache/nanochat/search_index/logs/build_YYYYMMDD_HHMMSS.log

# Tail latest log
tail -f /data/users/tarun/.cache/nanochat/search_index/logs/build_*.log
```

### 5. Stop the Build

```bash
# Find the PID
pgrep -f "data_analysis.search_index --build"

# Kill gracefully
kill <PID>

# Or force kill if needed
kill -9 <PID>
```

## Resuming

If the build is interrupted:

```bash
# Just run the script again - it will resume automatically
nohup ./data_analysis/run_build_index.sh &
```

Progress is tracked in `.index_progress.json` - completed files are skipped.

## Customization

Edit `run_build_index.sh` to change:

```bash
WORKERS=8      # Number of parallel workers (adjust based on resources)
LOG_DIR="..."  # Where to store log files
```

## Expected Runtime

With 8 workers and --no-sync:
- **Estimated time**: 24-30 hours for all 1820 files
- **Speed**: ~900-1200 docs/sec
- **Total docs**: ~97 million

## Checking System Resources

While build is running:

```bash
# Check overall system load
htop

# Check your cgroup limits
./data_analysis/check_status.sh

# Check memory usage
python -c "
with open('/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.current') as f:
    mem_gb = int(f.read().strip()) / (1024**3)
    print(f'Current memory usage: {mem_gb:.1f} GB')
"
```

## Tips

1. **Use `nohup`**: Ensures process continues even if you disconnect
2. **Monitor regularly**: Check status every few hours
3. **Watch memory**: If approaching 75 GB, consider reducing workers
4. **Be patient**: With 1820 files, this will take ~1-1.5 days
5. **Don't kill workers**: Let them finish their batch (or lose progress)

## Troubleshooting

**Q: Build seems stuck?**
```bash
./data_analysis/check_status.sh  # Check if still running
tail -f <latest_log>              # See what it's doing
```

**Q: Out of memory errors?**
- Reduce workers: Edit `run_build_index.sh` and set `WORKERS=4` or `WORKERS=6`

**Q: Want to start completely fresh?**
```bash
rm -rf /data/users/tarun/.cache/nanochat/search_index/
nohup ./data_analysis/run_build_index.sh &
```

