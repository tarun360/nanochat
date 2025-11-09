#!/bin/bash
# Script to monitor the build progress
# Usage: ./monitor_build.sh

LOG_DIR="/data/users/tarun/.cache/nanochat/search_index/logs"

# Find the most recent log file
LATEST_LOG=$(ls -t ${LOG_DIR}/build_*.log 2>/dev/null | head -n 1)

if [ -z "$LATEST_LOG" ]; then
    echo "No build log files found in $LOG_DIR"
    exit 1
fi

echo "Monitoring: $LATEST_LOG"
echo "Press Ctrl+C to stop monitoring"
echo "========================================="
echo ""

# Tail the log file with follow
tail -f "$LATEST_LOG"

