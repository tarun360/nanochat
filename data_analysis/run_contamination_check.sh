#!/bin/bash
# Script to run contamination check in the background with nohup
#
# Usage:
#   ./run_contamination_check.sh                          # Full check with default settings
#   ./run_contamination_check.sh --limit 25               # Quick test (25 examples per split)
#   ./run_contamination_check.sh --use-ngrams             # Use n-gram based search
#   ./run_contamination_check.sh --use-ngrams --limit 25  # N-gram search with limit
#   ./run_contamination_check.sh --use-ngrams --ngram-size 10 --num-ngrams 3  # Custom n-gram settings
#
# Options:
#   --limit N              Limit to N examples per split (for quick testing)
#   --use-ngrams           Enable n-gram based search (for partial contamination detection)
#   --ngram-size N         Terms per n-gram (default: 10, only with --use-ngrams)
#   --num-ngrams N         Number of n-grams to extract (default: 3, only with --use-ngrams)

set -e

cd "$(dirname "$0")/.."

# Configuration
LOG_DIR="contamination_results/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/contamination_check_${TIMESTAMP}.log"
PID_FILE="contamination_results/.contamination_check.pid"

# Parse arguments - pass all arguments through to the Python script
ARGS=""
while [[ $# -gt 0 ]]; do
    ARGS="$ARGS $1"
    if [[ "$1" == "--limit" ]] || [[ "$1" == "--ngram-size" ]] || [[ "$1" == "--num-ngrams" ]]; then
        ARGS="$ARGS $2"
        shift
    fi
    shift
done

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "ERROR: Contamination check is already running (PID: $OLD_PID)"
        echo "To stop it: kill $OLD_PID"
        echo "To force new run: rm $PID_FILE"
        exit 1
    else
        echo "Removing stale PID file..."
        rm "$PID_FILE"
    fi
fi

# Activate virtual environment
source .venv/bin/activate

# Build command - pass all arguments through
CMD="python -m data_analysis.check_contamination --all --save-detailed$ARGS"

echo "================================"
echo "Starting Contamination Check"
echo "================================"
echo "Command: $CMD"
echo "Log file: $LOG_FILE"
echo "Started at: $(date)"
echo ""

# Run in background with nohup
nohup $CMD > "$LOG_FILE" 2>&1 &
CHECK_PID=$!

# Save PID
echo "$CHECK_PID" > "$PID_FILE"

echo "Contamination check started successfully!"
echo "PID: $CHECK_PID"
echo ""
echo "To monitor progress:"
echo "  ./data_analysis/monitor_contamination.sh"
echo ""
echo "To check status:"
echo "  ./data_analysis/check_contamination_status.sh"
echo ""
echo "To stop:"
echo "  kill $CHECK_PID"

