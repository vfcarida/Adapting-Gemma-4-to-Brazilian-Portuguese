# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Gemma 4 PT-BR Adaptation — Makefile                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

.PHONY: install lint test tokenizer-audit contamination-check \
        cpt-pilot cpt-main merge sft eval baselines report all clean help

SHELL := /bin/bash
PYTHON := python
SCRIPTS := scripts

# ── Setup ────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m pip install -e ".[dev,eval-harness]"

lint:
	$(PYTHON) -m ruff check src/ --fix
	$(PYTHON) -m ruff format src/

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── Data Pipeline ────────────────────────────────────────────────────
tokenizer-audit:
	bash $(SCRIPTS)/run_tokenizer_audit.sh

contamination-check:
	bash $(SCRIPTS)/run_contamination_checks.sh

# ── Training Pipeline ───────────────────────────────────────────────
cpt-pilot:
	bash $(SCRIPTS)/run_cpt_pilot.sh

cpt-main:
	bash $(SCRIPTS)/run_cpt_main.sh

merge:
	bash $(SCRIPTS)/run_residual_merge.sh

sft:
	bash $(SCRIPTS)/run_sft.sh

# ── Evaluation Pipeline ─────────────────────────────────────────────
baselines:
	bash $(SCRIPTS)/run_baselines.sh

eval:
	bash $(SCRIPTS)/run_eval.sh

report:
	$(PYTHON) -m src.eval.report_builder

# ── Full Pipeline ────────────────────────────────────────────────────
all:
	bash $(SCRIPTS)/run_all.sh

# ── Cleanup ──────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .ruff_cache/ .pytest_cache/ *.egg-info/ dist/ build/

# ── Help ─────────────────────────────────────────────────────────────
help:
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║  Gemma 4 PT-BR Adaptation — Available Targets           ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  install              Install package + dependencies    ║"
	@echo "║  lint                 Run ruff linter + formatter       ║"
	@echo "║  test                 Run pytest suite                  ║"
	@echo "║  tokenizer-audit      Analyze tokenizer fertility       ║"
	@echo "║  contamination-check  Run 3-tier decontamination        ║"
	@echo "║  cpt-pilot            CPT on Gemma-4-E4B (pilot)        ║"
	@echo "║  cpt-main             CPT on Gemma-4-26B-A4B            ║"
	@echo "║  merge                Residual merge (task arithmetic)   ║"
	@echo "║  sft                  Supervised fine-tuning             ║"
	@echo "║  baselines            Evaluate baseline models          ║"
	@echo "║  eval                 Full evaluation suite              ║"
	@echo "║  report               Generate Markdown report          ║"
	@echo "║  all                  Run entire pipeline end-to-end    ║"
	@echo "║  clean                Remove caches and build artifacts  ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
