#!/bin/bash
# Script to check the status of contamination check

set -e

cd "$(dirname "$0")/.."

PID_FILE="contamination_results/.contamination_check.pid"
LOG_DIR="contamination_results/logs"
REPORT_FILE="contamination_results/contamination_report.json"

echo "================================"
echo "Contamination Check Status"
echo "================================"
echo ""

# Check if process is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Status: RUNNING"
        echo "PID: $PID"
        echo ""
        
        # Show process info
        echo "Process info:"
        ps -p "$PID" -o pid,etime,%cpu,%mem,cmd
        echo ""
        
        # Find the latest log file
        if [ -d "$LOG_DIR" ]; then
            LATEST_LOG=$(ls -t "$LOG_DIR"/contamination_check_*.log 2>/dev/null | head -1)
            if [ -n "$LATEST_LOG" ]; then
                echo "Latest log: $LATEST_LOG"
                echo ""
                echo "Last 20 lines of log:"
                echo "---"
                tail -20 "$LATEST_LOG"
            fi
        fi
    else
        echo "Status: STOPPED (stale PID file)"
        echo "Removing stale PID file..."
        rm "$PID_FILE"
    fi
else
    echo "Status: NOT RUNNING"
fi

echo ""
echo "================================"
echo "Results"
echo "================================"

# Check if report exists
if [ -f "$REPORT_FILE" ]; then
    echo ""
    echo "Report exists: $REPORT_FILE"
    echo "Generated: $(jq -r '.metadata.generated_at' "$REPORT_FILE" 2>/dev/null || echo 'unknown')"
    echo ""
    
    # Show summary from JSON
    echo "Quick Summary:"
    echo "---"
    
    # GSM8K summary
    echo "GSM8K:"
    for split in train test; do
        q_strong=$(jq -r ".gsm8k.$split.q_strong | length" "$REPORT_FILE" 2>/dev/null || echo 0)
        a_strong=$(jq -r ".gsm8k.$split.a_strong | length" "$REPORT_FILE" 2>/dev/null || echo 0)
        checked=$(jq -r ".gsm8k.$split.checked_examples" "$REPORT_FILE" 2>/dev/null || echo 0)
        echo "  $split: Q≥90%=$q_strong, A≥90%=$a_strong (checked $checked)"
    done
    
    echo ""
    echo "MATH (showing subjects with matches):"
    # Show only subjects with strong matches
    jq -r '.math | to_entries[] | select(.value.train.q_strong | length > 0 or .value.train.a_strong | length > 0 or .value.test.q_strong | length > 0 or .value.test.a_strong | length > 0) | "  \(.key):\n    train: Q≥90%=\(.value.train.q_strong | length), A≥90%=\(.value.train.a_strong | length)\n    test: Q≥90%=\(.value.test.q_strong | length), A≥90%=\(.value.test.a_strong | length)"' "$REPORT_FILE" 2>/dev/null || echo "  (no matches found)"
    
    echo ""
    echo "For full results:"
    echo "  cat contamination_results/contamination_report.txt"
    echo "  cat contamination_results/contamination_report.json | jq ."
else
    echo ""
    echo "No report found yet."
    echo "Report will be created when check completes."
fi

echo ""

