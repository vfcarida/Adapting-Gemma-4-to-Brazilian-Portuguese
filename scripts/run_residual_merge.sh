#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Run Residual Merge — Task Arithmetic                               ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

echo "═══ Residual Merge (Task Arithmetic) ═══"

# Alpha sweep: override via --alpha CLI flag or use config defaults
python -m src.train.residual_merge \
    --config configs/merge.yml \
    "$@"

echo "✓ Residual merge complete — see output/merged/"
