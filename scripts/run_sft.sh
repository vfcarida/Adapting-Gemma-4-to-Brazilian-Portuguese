#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Run Supervised Fine-Tuning (SFT)                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

echo "═══ Supervised Fine-Tuning (SFT) ═══"
echo "⚠️  Make sure configs/sft.yml → data.dataset_id is set to a valid"
echo "   instruction dataset. NEVER use Aurora-PT raw text here."

python -m src.train.sft_trainer \
    --config configs/sft.yml \
    "$@"

echo "✓ SFT complete — checkpoint at output/sft/final/"
