#!/bin/bash
set -euo pipefail

# Tokenizer Audit: Compare Gemma 4 tokenizer efficiency on Portuguese text
echo "=== Tokenizer Audit ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python3 -c "
from src.data.aurora_loader import AuroraLoader
from src.data.tokenizer_audit import run_tokenizer_audit
from src.utils.config_utils import load_config
from src.utils.hf_utils import load_tokenizer

# Load config
config = load_config('configs/data/aurora_pt.yaml')

# Load a sample of Aurora-PT
loader = AuroraLoader(config)
print('Loading Aurora-PT sample...')
ds = loader.load_raw(streaming=True)
texts = []
for i, example in enumerate(ds):
    if i >= 5000:
        break
    texts.append(example['text'])

# Load Gemma 4 tokenizer
print('Loading Gemma 4 tokenizer...')
tokenizer = load_tokenizer('google/gemma-4-E4B')

# Run audit
from datasets import Dataset
sample_ds = Dataset.from_dict({'text': texts})
results = run_tokenizer_audit(
    tokenizer=tokenizer,
    dataset=sample_ds,
    sample_size=5000,
    output_path='outputs/tokenizer_audit.json',
)
print(f'Audit complete. Results saved to outputs/tokenizer_audit.json')
print(f'  Tokens/word: {results[\"tokens_per_word_mean\"]:.3f} +/- {results[\"tokens_per_word_std\"]:.3f}')
print(f'  Tokens/char: {results[\"tokens_per_char_mean\"]:.4f}')
"
