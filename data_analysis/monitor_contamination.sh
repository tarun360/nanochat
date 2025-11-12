#!/bin/bash
# Script to monitor contamination check logs in real-time

set -e

cd "$(dirname "$0")/.."

LOG_DIR="contamination_results/logs"

if [ ! -d "$LOG_DIR" ]; then
    echo "No log directory found: $LOG_DIR"
    exit 1
fi

# Find the latest log file
LATEST_LOG=$(ls -t "$LOG_DIR"/contamination_check_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No log files found in $LOG_DIR"
    exit 1
fi

echo "Monitoring: $LATEST_LOG"
echo "Press Ctrl+C to stop monitoring"
echo "================================"
echo ""

tail -f "$LATEST_LOG"

