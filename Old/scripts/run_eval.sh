#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Run Full Evaluation Suite                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

echo "═══ Full Evaluation Suite ═══"
echo "Tasks: ENEM, BluEx, OAB, ASSIN2-RTE/STS, HateBR, TweetSentBR,"
echo "       COPA-PT, BRoverbs, MRPC-PT, RTE-PT, DoNotAnswer-PT, TugueSICE-PT"
echo "Modes: think_off + think_on"

# Run benchmark suite
python -m src.eval.benchmark_runner \
    --config configs/eval.yml \
    "$@"

# Generate Markdown report
echo ""
echo "═══ Generating Report ═══"
python -m src.eval.report_builder \
    --config configs/eval.yml \
    --results_dir reports/eval_results \
    --output reports/benchmark_report.md

echo "✓ Evaluation complete — see reports/benchmark_report.md"
