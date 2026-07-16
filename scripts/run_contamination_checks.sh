#!/bin/bash
set -euo pipefail

# Contamination checks between training data and evaluation benchmarks
echo "=== Contamination Checks ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python3 -c "
from src.data.aurora_loader import AuroraLoader
from src.data.contamination_checks import run_contamination_report
from src.eval.tasks import load_task
from src.utils.config_utils import load_config

# Load training data sample
print('Loading Aurora-PT sample for contamination check...')
data_config = load_config('configs/data/aurora_pt.yaml')
loader = AuroraLoader(data_config)
ds = loader.load_raw(streaming=True)
train_texts = []
for i, example in enumerate(ds):
    if i >= 50000:  # Check first 50K documents
        break
    train_texts.append(example['text'])

# Load benchmark texts
print('Loading benchmark texts...')
eval_config = load_config('configs/eval/benchmarks.yaml')
benchmarks = {}

for bench_name, bench_cfg in eval_config.get('benchmarks', {}).items():
    if not bench_cfg.get('enabled', True):
        continue
    try:
        task = load_task(bench_cfg['task'])
        examples = task.load_data(bench_cfg)
        texts = []
        for ex in examples:
            t = ex.get('question', ex.get('text', ex.get('premise', '')))
            if t:
                texts.append(t)
        if texts:
            benchmarks[bench_name] = texts
            print(f'  {bench_name}: {len(texts)} samples')
    except Exception as e:
        print(f'  {bench_name}: SKIPPED ({e})')

# Run contamination checks
if benchmarks:
    print(f'\\nRunning contamination checks ({len(train_texts)} train docs vs {len(benchmarks)} benchmarks)...')
    report = run_contamination_report(train_texts, benchmarks, 'outputs/contamination')
    print('Done! Report saved to outputs/contamination/')
else:
    print('No benchmarks could be loaded. Skipping contamination check.')
"
