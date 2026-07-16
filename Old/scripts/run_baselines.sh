#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Run Baseline Model Evaluations                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

echo "═══ Baseline Evaluation ═══"
echo "Baselines: ${BASELINE_MODELS:-CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it,maritaca-ai/sabia-7b,Polygl0t/Tucano2-qwen-3.7B-Instruct}"

python -m src.eval.benchmark_runner \
    --config configs/eval.yml \
    --override \
        "models=[$(echo "${BASELINE_MODELS:-CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it,maritaca-ai/sabia-7b,Polygl0t/Tucano2-qwen-3.7B-Instruct}" | tr ',' ' ')]"

echo "✓ Baseline evaluation complete — see reports/eval_results/"
