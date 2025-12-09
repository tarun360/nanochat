#!/bin/bash
# Script to build the search index with parallel workers
# Usage: nohup ./run_build_index.sh [max_files] &
# Example: nohup ./run_build_index.sh 240 &  (limit to 240 files)

set -e  # Exit on error

# Configuration
WORKERS=8
MAX_FILES="${1:-}"  # Optional: first argument = max number of files to process
LOG_DIR="/data/users/tarun/.cache/nanochat/base_data_search_index/logs"
LOG_FILE="${LOG_DIR}/build_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "========================================" | tee -a "$LOG_FILE"
echo "Search Index Build Started" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"
echo "Workers: $WORKERS" | tee -a "$LOG_FILE"
if [ -n "$MAX_FILES" ]; then
    echo "Max files: $MAX_FILES" | tee -a "$LOG_FILE"
else
    echo "Max files: all (1820)" | tee -a "$LOG_FILE"
fi
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Activate virtual environment (if needed)
# Uncomment if you need to activate venv explicitly
# source .venv/bin/activate

# Run the indexing command
cd "$(dirname "$0")/.."

# Build command with optional max-files parameter
CMD="python -m data_analysis.search_index --build --no-sync --workers $WORKERS"
if [ -n "$MAX_FILES" ]; then
    CMD="$CMD --max-files $MAX_FILES"
fi

$CMD 2>&1 | tee -a "$LOG_FILE"

# Capture exit status
EXIT_CODE=$?

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Search Index Build Finished" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE

