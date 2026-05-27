#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Run Tokenizer Fertility Audit                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

echo "═══ Tokenizer Fertility Audit ═══"
echo "Model: ${PILOT_MODEL_ID:-google/gemma-4-E4B}"
echo "Dataset: ${AURORA_DATASET_ID:-Itau-Unibanco/Aurora-PT}"

python -m src.data.tokenizer_audit \
    --model_id "${PILOT_MODEL_ID:-google/gemma-4-E4B}" \
    --dataset_id "${AURORA_DATASET_ID:-Itau-Unibanco/Aurora-PT}" \
    --num_samples "${AUDIT_NUM_SAMPLES:-1000}" \
    --output "reports/tokenizer_audit.json"

echo "✓ Audit complete — see reports/tokenizer_audit.json"
