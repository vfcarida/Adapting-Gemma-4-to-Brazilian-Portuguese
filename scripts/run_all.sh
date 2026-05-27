#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Run All — End-to-End Pipeline                                      ║
# ║                                                                      ║
# ║  Stages:                                                             ║
# ║   1. Tokenizer Audit                                                 ║
# ║   2. Contamination Checks                                            ║
# ║   3. Baseline Evaluation                                             ║
# ║   4. CPT Pilot (Gemma-4-E4B)                                        ║
# ║   5. Residual Merge                                                  ║
# ║   6. SFT                                                             ║
# ║   7. Full Evaluation                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Gemma 4 PT-BR Adaptation — Full Pipeline                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# Create output directories
mkdir -p reports/{eval_results,training_logs} output/{cpt_pilot_e4b,merged,sft}

# ── Stage 1: Data Quality ────────────────────────────────────────────
echo ""
echo "▶ Stage 1/7: Tokenizer Audit"
bash "$SCRIPT_DIR/run_tokenizer_audit.sh"

echo ""
echo "▶ Stage 2/7: Contamination Checks"
bash "$SCRIPT_DIR/run_contamination_checks.sh"

# ── Stage 2: Baselines ──────────────────────────────────────────────
echo ""
echo "▶ Stage 3/7: Baseline Evaluation"
bash "$SCRIPT_DIR/run_baselines.sh"

# ── Stage 3: Training ───────────────────────────────────────────────
echo ""
echo "▶ Stage 4/7: CPT Pilot (Gemma-4-E4B)"
bash "$SCRIPT_DIR/run_cpt_pilot.sh"

# ── Stage 4: Merge ──────────────────────────────────────────────────
echo ""
echo "▶ Stage 5/7: Residual Merge"
bash "$SCRIPT_DIR/run_residual_merge.sh"

# ── Stage 5: SFT ────────────────────────────────────────────────────
echo ""
echo "▶ Stage 6/7: SFT"
bash "$SCRIPT_DIR/run_sft.sh"

# ── Stage 6: Evaluation ─────────────────────────────────────────────
echo ""
echo "▶ Stage 7/7: Full Evaluation"
bash "$SCRIPT_DIR/run_eval.sh"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ✓ Pipeline Complete!                                            ║"
echo "║  Reports:  reports/benchmark_report.md                           ║"
echo "║  Models:   output/sft/final/                                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
