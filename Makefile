.PHONY: install test test-quick smoke lint format preflight validate-configs clean ready all
.PHONY: audit contamination baselines cpt-pilot cpt-main merge sft eval run-all

# === Desenvolvimento ===

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-quick:
	pytest tests/ -q --tb=line

smoke:
	python -m tests.smoke_test

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

preflight:
	python -m src.preflight

validate-configs:
	@python -c "import yaml; from pathlib import Path; \
	configs = list(Path('configs').rglob('*.yaml')); \
	[yaml.safe_load(f.read_text()) for f in configs]; \
	print(f'All {len(configs)} configs valid.')"

# === Pipeline ===

audit:
	bash scripts/run_tokenizer_audit.sh

contamination:
	bash scripts/run_contamination_checks.sh

baselines:
	bash scripts/run_baselines.sh

cpt-pilot:
	bash scripts/run_cpt_pilot.sh

cpt-main:
	bash scripts/run_cpt_main.sh

merge:
	bash scripts/run_residual_merge.sh

sft:
	bash scripts/run_sft.sh

eval:
	bash scripts/run_eval.sh

run-all:
	bash scripts/run_all.sh

# === Compostos ===

ready: install preflight test smoke
	@echo ""
	@echo "=== ALL CHECKS PASSED — READY FOR TRAINING ==="

all: lint test smoke validate-configs

clean:
	rm -rf outputs/tmp_* .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
