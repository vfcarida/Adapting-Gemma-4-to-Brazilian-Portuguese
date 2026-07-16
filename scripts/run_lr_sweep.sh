#!/bin/bash
# =============================================================================
# Learning Rate Sweep — Find optimal LR before full CPT run
# =============================================================================
# Runs short training (500 steps) with multiple LR values and selects the best.
# Must be run BEFORE the full pilot to avoid wasting compute on suboptimal LR.
#
# Usage:
#   bash scripts/run_lr_sweep.sh
#   bash scripts/run_lr_sweep.sh --lrs "5e-5 1e-4 2e-4 3e-4"
#   bash scripts/run_lr_sweep.sh --steps 1000
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Defaults
LRS="5e-5 1e-4 2e-4 3e-4"
STEPS=500
CONFIG="configs/train/lr_sweep.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lrs) LRS="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Learning Rate Sweep ==="
echo "LRs: ${LRS}"
echo "Steps per LR: ${STEPS}"
echo ""

RESULTS_FILE="outputs/lr_sweep/sweep_results.json"
mkdir -p outputs/lr_sweep

# Run each LR
BEST_LR=""
BEST_LOSS="999.0"

for lr in ${LRS}; do
    echo "--- LR = ${lr} ---"
    OUTPUT_DIR="outputs/lr_sweep/lr_${lr}"

    python3 -m src.train.cpt_trainer \
        --config "$CONFIG" \
        --override \
            "training.learning_rate=${lr}" \
            "training.max_steps=${STEPS}" \
            "output.output_dir=${OUTPUT_DIR}" \
            "experiment.name=lr_sweep_${lr}"

    # Extract final eval loss from training log
    if [[ -f "${OUTPUT_DIR}/train_log.jsonl" ]]; then
        FINAL_LOSS=$(python3 -c "
import json
losses = []
with open('${OUTPUT_DIR}/train_log.jsonl') as f:
    for line in f:
        data = json.loads(line)
        if 'eval_loss' in data:
            losses.append(data['eval_loss'])
if losses:
    print(f'{losses[-1]:.6f}')
else:
    # Fallback to training loss
    with open('${OUTPUT_DIR}/train_log.jsonl') as f:
        for line in f:
            data = json.loads(line)
            if 'loss' in data:
                losses.append(data['loss'])
    print(f'{losses[-1]:.6f}' if losses else '999.0')
")
    else
        FINAL_LOSS="999.0"
    fi

    echo "  LR=${lr} → Final loss: ${FINAL_LOSS}"

    # Track best
    IS_BETTER=$(python3 -c "print('yes' if float('${FINAL_LOSS}') < float('${BEST_LOSS}') else 'no')")
    if [[ "$IS_BETTER" == "yes" ]]; then
        BEST_LR="$lr"
        BEST_LOSS="$FINAL_LOSS"
    fi
done

echo ""
echo "=== LR Sweep Results ==="
echo "Best LR: ${BEST_LR} (loss: ${BEST_LOSS})"
echo ""
echo "Recommendation: Use learning_rate=${BEST_LR} in your full training config."
echo ""

# Save results
python3 -c "
import json, os
from pathlib import Path

results = {'best_lr': '${BEST_LR}', 'best_loss': float('${BEST_LOSS}'), 'sweep': {}}
for lr in '${LRS}'.split():
    log_path = Path(f'outputs/lr_sweep/lr_{lr}/train_log.jsonl')
    if log_path.exists():
        losses = []
        with open(log_path) as f:
            for line in f:
                data = json.loads(line)
                if 'loss' in data:
                    losses.append(data['loss'])
        results['sweep'][lr] = {'losses': losses[-10:] if losses else [], 'final_loss': losses[-1] if losses else None}

with open('outputs/lr_sweep/sweep_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved to outputs/lr_sweep/sweep_results.json')
"
