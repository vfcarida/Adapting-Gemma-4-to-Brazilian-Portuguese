#!/bin/bash
set -euo pipefail

# Run residual merge with alpha sweep
echo "=== Residual Merge (Task Arithmetic) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

BASE_MODEL="google/gemma-4-E4B"
INSTRUCT_MODEL="google/gemma-4-E4B-it"
CPT_MODEL="outputs/cpt_main/final"
OUTPUT_DIR="outputs/residual_merge"

if [ ! -d "$CPT_MODEL" ]; then
    echo "ERROR: CPT model not found at $CPT_MODEL"
    echo "Run CPT first: bash scripts/run_cpt_main.sh"
    exit 1
fi

python3 -m src.train.residual_merge \
    --base-model "$BASE_MODEL" \
    --instruct-model "$INSTRUCT_MODEL" \
    --cpt-model "$CPT_MODEL" \
    --alpha 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Residual Merge complete ==="
echo "Models saved to $OUTPUT_DIR/alpha_*"
