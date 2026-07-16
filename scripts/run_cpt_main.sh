#!/bin/bash
set -euo pipefail

# Run full CPT with English replay
echo "=== CPT Main (Full Model) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python3 -m src.train.cpt_trainer --config configs/train/cpt_main.yaml

echo ""
echo "=== CPT Main complete ==="
echo "Checkpoint saved to outputs/cpt_main/final"
