#!/bin/bash
# =============================================================================
# Multi-Seed Experiment Runner
# =============================================================================
# Runs the same experiment with multiple seeds for statistical significance.
# Papers require minimum 3 seeds; 5 is recommended for generation tasks.
#
# Usage:
#   bash scripts/run_multi_seed.sh --config configs/train/cpt_pilot.yaml
#   bash scripts/run_multi_seed.sh --config configs/train/cpt_pilot.yaml --seeds "42 123 456"
#   bash scripts/run_multi_seed.sh --config configs/train/cpt_pilot.yaml --eval-after
#   bash scripts/run_multi_seed.sh --config configs/train/cpt_pilot.yaml --parallel
#
# After training, run evaluation:
#   bash scripts/run_multi_seed.sh --eval-only --config configs/train/cpt_pilot.yaml
# =============================================================================

set -euo pipefail

# Defaults
SEEDS="42 123 456"
CONFIG=""
EVAL_AFTER=false
EVAL_ONLY=false
PARALLEL=false
EVAL_CONFIG="configs/eval/benchmarks.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --eval-after) EVAL_AFTER=true; shift ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        --parallel) PARALLEL=true; shift ;;
        --eval-config) EVAL_CONFIG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    echo "Usage: bash scripts/run_multi_seed.sh --config configs/train/cpt_pilot.yaml"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SEED_ARRAY=($SEEDS)
NUM_SEEDS=${#SEED_ARRAY[@]}

echo "=== Multi-Seed Experiment Runner ==="
echo "Config: $CONFIG"
echo "Seeds: ${SEEDS} (${NUM_SEEDS} runs)"
echo ""

# --- Training phase ---
if [[ "$EVAL_ONLY" == "false" ]]; then
    echo "--- Training Phase ---"

    if [[ "$PARALLEL" == "true" ]]; then
        echo "Running ${NUM_SEEDS} seeds in parallel..."
        PIDS=()
        for seed in ${SEEDS}; do
            echo "  Starting seed=${seed}..."
            python3 -m src.train.cpt_trainer --config "$CONFIG" --seed "$seed" \
                2>&1 | tee "outputs/train_seed${seed}.log" &
            PIDS+=($!)
        done

        # Wait for all and check exit codes
        FAILED=0
        for pid in "${PIDS[@]}"; do
            if ! wait "$pid"; then
                FAILED=$((FAILED + 1))
            fi
        done

        if [[ $FAILED -gt 0 ]]; then
            echo "WARNING: ${FAILED}/${NUM_SEEDS} seed runs failed"
        fi
    else
        echo "Running ${NUM_SEEDS} seeds sequentially..."
        for seed in ${SEEDS}; do
            echo ""
            echo "=== Seed ${seed} ($(date)) ==="
            python3 -m src.train.cpt_trainer --config "$CONFIG" --seed "$seed"
            echo "=== Seed ${seed} complete ==="
        done
    fi

    echo ""
    echo "--- Training Phase Complete ---"
fi

# --- Evaluation phase ---
if [[ "$EVAL_AFTER" == "true" || "$EVAL_ONLY" == "true" ]]; then
    echo ""
    echo "--- Evaluation Phase ---"

    # Extract base output dir from config
    BASE_OUTPUT=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('output', {}).get('output_dir', 'outputs/cpt'))
")

    for seed in ${SEEDS}; do
        MODEL_PATH="${BASE_OUTPUT}_seed${seed}/final"
        if [[ -d "$MODEL_PATH" ]]; then
            echo "Evaluating seed=${seed} model: ${MODEL_PATH}"
            python3 -m src.eval.benchmark_runner \
                --config "$EVAL_CONFIG" \
                --model "$MODEL_PATH"
        else
            echo "WARNING: Model not found at ${MODEL_PATH}, skipping seed=${seed}"
        fi
    done

    echo ""
    echo "--- Evaluation Phase Complete ---"
    echo ""
    echo "To aggregate results across seeds, run:"
    echo "  python3 -m src.eval.aggregate_seeds --output-dir ${BASE_OUTPUT}"
fi

echo ""
echo "=== Multi-Seed Run Complete ==="
