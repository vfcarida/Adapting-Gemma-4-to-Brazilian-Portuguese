#!/bin/bash
# =============================================================================
# Pipeline de Qualidade de Dados
# =============================================================================
# Executa: manifesto de qualidade, deduplicação, splits por cluster
# Pré-requisitos: pip install -e ".[dev]"

set -euo pipefail

echo "============================================"
echo " Pipeline de Qualidade de Dados — Aurora-PT"
echo "============================================"

CONFIG="${1:-configs/data/aurora_pt.yaml}"
OUTPUT_DIR="outputs/data_qc"

echo ""
echo "[1/3] Construindo manifesto de qualidade..."
python3 -m src.data.make_splits \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "[2/3] Executando auditoria de tokenizer..."
bash scripts/run_tokenizer_audit.sh

echo ""
echo "[3/3] Executando checks de contaminação..."
bash scripts/run_contamination_checks.sh

echo ""
echo "============================================"
echo " Pipeline de QC concluído!"
echo " Resultados em: $OUTPUT_DIR"
echo "============================================"
