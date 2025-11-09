#!/bin/bash
# Script to build the search index with parallel workers
# Usage: nohup ./run_build_index.sh &

set -e  # Exit on error

# Configuration
WORKERS=8
LOG_DIR="/data/users/tarun/.cache/nanochat/search_index/logs"
LOG_FILE="${LOG_DIR}/build_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "========================================" | tee -a "$LOG_FILE"
echo "Search Index Build Started" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"
echo "Workers: $WORKERS" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Activate virtual environment (if needed)
# Uncomment if you need to activate venv explicitly
# source .venv/bin/activate

# Run the indexing command
cd /home/tarun/nanochat
python -m data_analysis.search_index --build --no-sync --workers "$WORKERS" 2>&1 | tee -a "$LOG_FILE"

# Capture exit status
EXIT_CODE=$?

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Search Index Build Finished" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE

