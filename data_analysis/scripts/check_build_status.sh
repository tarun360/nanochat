#!/bin/bash
# Script to check the current status of indexing
# Usage: ./check_status.sh

echo "========================================="
echo "Search Index Build Status Check"
echo "Time: $(date)"
echo "========================================="
echo ""

# Check if build process is running
BUILD_PID=$(pgrep -f "data_analysis.search_index --build" | head -n 1)
if [ -n "$BUILD_PID" ]; then
    echo "✓ Build process is RUNNING (PID: $BUILD_PID)"
    echo ""
    
    # Show CPU and memory usage
    echo "Resource usage:"
    ps -p "$BUILD_PID" -o pid,ppid,%cpu,%mem,etime,cmd --no-headers
    echo ""
    
    # Show worker processes
    echo "Worker processes:"
    pgrep -P "$BUILD_PID" -a | grep -E "python|xapian" || echo "  (workers may be starting...)"
    echo ""
else
    echo "✗ Build process is NOT running"
    echo ""
fi

# Check progress file
PROGRESS_FILE="/data/users/tarun/.cache/nanochat/base_data_search_index/.index_progress.json"
if [ -f "$PROGRESS_FILE" ]; then
    echo "Progress file found:"
    COMPLETED=$(python3 -c "import json; print(len(json.load(open('$PROGRESS_FILE'))['completed_files']))" 2>/dev/null || echo "0")
    echo "  Completed files: $COMPLETED / 1820 ($(python3 -c "print(f'{$COMPLETED/1820*100:.1f}%')" 2>/dev/null || echo "N/A"))"
else
    echo "No progress file found yet (indexing may not have started or is first run)"
fi
echo ""

# Show latest log entries
LOG_DIR="/data/users/tarun/.cache/nanochat/base_data_search_index/logs"
LATEST_LOG=$(ls -t ${LOG_DIR}/build_*.log 2>/dev/null | head -n 1)

if [ -n "$LATEST_LOG" ]; then
    echo "Latest log file: $LATEST_LOG"
    echo "Last 10 lines:"
    echo "---"
    tail -n 10 "$LATEST_LOG"
else
    echo "No log files found"
fi

echo ""
echo "========================================="
echo "To monitor live: ./data_analysis/scripts/monitor_build.sh"
echo "To stop build: kill $BUILD_PID"
echo "========================================="

