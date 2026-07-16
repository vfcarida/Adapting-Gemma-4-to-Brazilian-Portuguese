#!/bin/bash
set -euo pipefail

# Run CPT pilot (LoRA, small scale) to validate pipeline
echo "=== CPT Pilot (LoRA) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python3 -m src.train.cpt_trainer --config configs/train/cpt_pilot.yaml

echo ""
echo "=== CPT Pilot complete ==="
echo "Checkpoint saved to outputs/cpt_pilot/final"
