#!/bin/bash
set -euo pipefail

# Run evaluation on baseline models (no adaptation)
echo "=== Baseline Evaluation ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODELS=(
    "google/gemma-4-E4B-it"
    "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"
    "maritaca-ai/sabia-7b"
)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "--- Evaluating: $MODEL ---"
    python3 -m src.eval.benchmark_runner --config configs/eval/benchmarks.yaml --model "$MODEL" || \
        echo "WARNING: Failed to evaluate $MODEL"
done

echo ""
echo "=== Baseline evaluation complete ==="
echo "Results saved to reports/"
