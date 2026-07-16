#!/bin/bash
set -euo pipefail

# Run the full pipeline end-to-end
echo "=========================================="
echo "  Gemma 4 Portuguese Adaptation Pipeline"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo ""
echo "[1/7] Tokenizer Audit"
bash scripts/run_tokenizer_audit.sh

echo ""
echo "[2/7] Contamination Checks"
bash scripts/run_contamination_checks.sh

echo ""
echo "[3/7] Baseline Evaluation"
bash scripts/run_baselines.sh

echo ""
echo "[4/7] Continued Pretraining (CPT)"
bash scripts/run_cpt_main.sh

echo ""
echo "[5/7] Residual Merge"
bash scripts/run_residual_merge.sh

echo ""
echo "[6/7] Supervised Fine-Tuning (SFT)"
bash scripts/run_sft.sh

echo ""
echo "[7/7] Full Evaluation"
bash scripts/run_eval.sh

echo ""
echo "=========================================="
echo "  Pipeline Complete!"
echo "=========================================="
echo "  Reports: reports/"
echo "  Models:  outputs/"
echo "=========================================="
