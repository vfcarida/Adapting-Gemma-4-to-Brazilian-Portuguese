#!/bin/bash
set -euo pipefail

# Run SFT on CPT checkpoint
echo "=== Supervised Fine-Tuning ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python3 -m src.train.sft_trainer --config configs/train/sft.yaml

echo ""
echo "=== SFT complete ==="
echo "Model saved to outputs/sft/final"
