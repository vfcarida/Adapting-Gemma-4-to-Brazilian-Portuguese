#!/bin/bash
set -euo pipefail

# Run evaluation on all trained models
echo "=== Full Evaluation ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Evaluate all models in config
python3 -m src.eval.benchmark_runner --config configs/eval/benchmarks.yaml

# Generate reports
python3 -c "
from src.eval.report_builder import ReportBuilder, build_findings_for_paper
import json
from pathlib import Path

results_path = Path('reports/eval_results.json')
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
    builder = ReportBuilder(results, output_dir='reports')
    builder.build_all()
    build_findings_for_paper('reports')
    print('Reports generated in reports/')
else:
    print('No results found. Run evaluation first.')
"

echo ""
echo "=== Evaluation complete ==="
echo "Reports saved to reports/"
