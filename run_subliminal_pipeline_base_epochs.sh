#!/bin/bash
# Run base model subliminal pipeline (v3) with different STUDENT_EPOCHS values sequentially

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PIPELINE_SCRIPT="$SCRIPT_DIR/run_subliminal_pipeline_base_local.sh"

# Check if the pipeline script exists
if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "ERROR: Pipeline script not found: $PIPELINE_SCRIPT"
    exit 1
fi

# Make sure the pipeline script is executable
chmod +x "$PIPELINE_SCRIPT"

# Array of STUDENT_EPOCHS values to run
EPOCHS=(1 3 5 7 9)

echo "================================================================="
echo "=== Running base model pipeline with multiple STUDENT_EPOCHS ==="
echo "================================================================="
echo ""

# Run the pipeline for each STUDENT_EPOCHS value
for EPOCHS_VAL in "${EPOCHS[@]}"; do
    echo ""
    echo "================================================================="
    echo "=== Starting run with STUDENT_EPOCHS=$EPOCHS_VAL at $(date) ==="
    echo "================================================================="
    echo ""

    STUDENT_EPOCHS=$EPOCHS_VAL "$PIPELINE_SCRIPT"

    echo ""
    echo "================================================================="
    echo "=== Completed run with STUDENT_EPOCHS=$EPOCHS_VAL at $(date) ==="
    echo "================================================================="
    echo ""
done

echo ""
echo "================================================================="
echo "=== All runs completed at $(date) ==="
echo "================================================================="
echo ""
