# Contributing Guide

## Development Setup

```bash
git clone <repo-url>
cd gemma4-pt-br-adaptation
pip install -e ".[dev]"
cp .env.example .env
```

## Code Style

- Python 3.10+ with type hints
- Line length: 100 chars (ruff)
- Docstrings: Google style for public APIs
- Imports: stdlib, third-party, local (separated by blank lines)

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```

## Testing

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_metrics.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Tests are organized by module:
- `test_metrics.py` — Metric computation correctness
- `test_parsing.py` — Answer extraction from model outputs
- `test_residual_merge.py` — Task arithmetic math
- `test_config.py` — YAML loading and merging
- `test_bootstrap.py` — Statistical inference
- `test_contamination.py` — Contamination detection
- `test_data_pipeline.py` — Data loading and preprocessing
- `test_prompt_templates.py` — Prompt formatting
- `test_instruction_builder.py` — Chat template formatting

## Adding an Experiment

1. Create a YAML config in `configs/train/` or `configs/ablations/`
2. Test locally with `max_steps: 10` to validate pipeline
3. Run full experiment on GPU cluster
4. Results auto-saved to `outputs/<experiment_name>/`

## Project Conventions

- **Never hardcode paths** — Use configs or environment variables
- **Always set seed** — Via config, propagated to all random sources
- **Log everything** — Metrics to JSONL, progress to console
- **Cache inference** — Evaluation results cached by model+benchmark+seed
- **Idempotent scripts** — Re-running detects existing outputs/checkpoints
