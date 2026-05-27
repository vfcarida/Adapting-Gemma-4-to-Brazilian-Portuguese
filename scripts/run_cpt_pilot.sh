#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Run CPT Pilot — Gemma-4-E4B                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

echo "═══ Continued Pretraining (Pilot) — Gemma-4-E4B ═══"

# Single-GPU training
python -m src.train.cpt_trainer \
    --config configs/cpt_pilot.yml \
    "$@"

echo "✓ CPT pilot complete — checkpoint at output/cpt_pilot_e4b/final/"
