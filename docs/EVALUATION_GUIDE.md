# Evaluation Guide

## Benchmark Suite

Every `Source` below is a real HuggingFace dataset ID, verified live against
the Hub — see `configs/eval/benchmarks.yaml` for the exact `hub_id`/`subset`
per entry and any split/gating notes.

### Brasil Geral (General Knowledge)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| ENEM 2022/2023/2024 | Multiple choice (5 options) | Accuracy | `maritaca-ai/enem` |
| BLUEX | Multiple choice (university entrance) | Accuracy | `eduagarcia-temp/BLUEX_without_images` |

### Semantica (Semantic Understanding)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| ASSIN2-RTE | Textual entailment | Accuracy | `nilc-nlp/assin2` |
| ASSIN2-STS | Semantic similarity (1-5) | Pearson | `nilc-nlp/assin2` |
| CoPA-PT | Causal reasoning (2 choices) | Accuracy | `PORTULAN/extraglue` (`copa_pt-BR`) |
| BoolQ-PT | Yes/no QA over a passage | Accuracy | `PORTULAN/extraglue` (`boolq_pt-BR`) |
| MRPC-PT | Paraphrase detection | Accuracy | `PORTULAN/extraglue` (`mrpc_pt-BR`) |
| RTE-PT | Textual entailment | Accuracy | `PORTULAN/extraglue` (`rte_pt-BR`) |

### Classificacao Social (Social Classification)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| HateBR | Hate speech detection | Macro-F1 | `ruanchaves/hatebr` |
| TweetSentBR | Sentiment (pos/neg/neu) | Macro-F1 | `eduagarcia/tweetsentbr_fewshot` |

### Juridico (Legal)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| OAB-Bench | Bar exam questions (4 options) | Accuracy | `eduagarcia/oab_exams` |
| LeNER-Br | Legal named-entity recognition | Entity Micro-F1 | `peluz/lener_br` |
| LegalBench-BR | Court-decision judgment classification | Accuracy | `eduagarcia/portuguese_benchmark` (`brazilian_court_decisions_judgment`) |

### Cultura (Cultural Knowledge)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| BRoverbs | Proverb completion (5 options) | Accuracy | `Tropic-AI/BRoverbs` |
| CAPITU | Instruction-following w/ Brazilian literary context (NOT reading comprehension — see `src/eval/tasks/capitu.py`) | IFEval-style pass rate (current code assumes multiple-choice, needs rewriting) | **Disabled by default** — data lives at `github.com/maritaca-ai/capitu`, not the HF Hub (paper: arXiv:2603.22576) |

### Raciocinio (Math)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| Math-PT | Math word problems (5 options) | Accuracy | `tiagoteixeira03/MATH-PT` (`ptbr_multiple_choice`; arXiv:2604.25926) |

### Dominio Publico

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| PublicHearing-BR | Public-hearing summarization | Macro-F1 | `unicamp-dl/PublicHearingBR` |

### Seguranca (Safety)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| DoNotAnswer-PT | Harmful question refusal | Refusal Rate | **Disabled by default** — no public HF dataset exists for a Brazilian-Portuguese DoNotAnswer yet |

### Retencao EN (English Retention)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| MMLU | Multiple choice (4 options) | Accuracy | `cais/mmlu` |
| HellaSwag | Sentence continuation (4 options) | Accuracy | `Rowan/hellaswag` |
| ARC-Challenge | Science QA (multiple choice) | Accuracy | `allenai/ai2_arc` |

### Exploratorio

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| XL-Sum-PT | Summarization | ROUGE-L | `csebuetnlp/xlsum` (`portuguese`) |

**Not wired into the config** (implemented as a task class but no usable
dataset — see comments in the task file for the full explanation):
Tuguesice-PT (`tuguesice_pt.py`) — this benchmark is real (CLARIN-PT-LDB,
arXiv:2603.12872, part of `PORTULAN/portuguese-llm-leaderboard`), a prior
pass here wrongly called it fabricated, but its data isn't public
(held-out leaderboard data) and it targets *European* Portuguese cultural
knowledge, not Brazilian — a poor fit for this project either way.

## Running Evaluation

```bash
# All models defined in config
python3 -m src.eval.benchmark_runner --config configs/eval/benchmarks.yaml

# Single model
python3 -m src.eval.benchmark_runner --config configs/eval/benchmarks.yaml \
    --model google/gemma-4-E4B-it
```

## Inference Backends

| Backend | Speed | VRAM | Best for |
|---------|-------|------|----------|
| HuggingFace Transformers | Baseline | High | Debugging, small batches |
| vLLM | 3-5x faster | Lower | Full evaluation runs |

Set `use_vllm: true` in `configs/eval/benchmarks.yaml` for production runs.

## Think Mode Evaluation

Gemma 4 IT models support a "thinking" mode where reasoning happens inside
`<think>...</think>` tags before the final answer.

**Evaluation protocol**:
1. Run each benchmark twice: `think_off` and `think_on`
2. For `think_on`: strip `<think>...</think>` before parsing the answer
3. Report both modes separately in results
4. Never include previous thinking in multi-turn prompts

## Caching

Results are cached per (model_id, benchmark, think_mode, seed) to avoid
re-running expensive inference. Cache stored in `outputs/eval_cache/`.

To force re-evaluation, delete the cache directory:
```bash
rm -rf outputs/eval_cache/
```

## Statistical Rigor

### Bootstrap Confidence Intervals

All metrics include 95% CIs computed via 1000 bootstrap resamples:
```python
from src.eval.bootstrap_ci import bootstrap_ci
from src.eval.metrics import accuracy

result = bootstrap_ci(predictions, gold_labels, accuracy, n_bootstrap=1000)
# Returns: {"accuracy": {"mean": 0.72, "ci_lower": 0.68, "ci_upper": 0.76, ...}}
```

### Paired Bootstrap Test

To determine if model A is significantly better than model B:
```python
from src.eval.bootstrap_ci import paired_bootstrap_test

result = paired_bootstrap_test(preds_a, preds_b, gold, accuracy, "accuracy")
# Returns: {"p_value_a_gt_b": 0.003, "significant_at_05": True}
```

## Report Generation

After evaluation, generate all artifacts:
```bash
python3 -c "
from src.eval.report_builder import ReportBuilder, build_findings_for_paper
import json

with open('reports/eval_results.json') as f:
    results = json.load(f)

builder = ReportBuilder(results)
builder.build_all()
build_findings_for_paper()
"
```

Generated outputs:
- `reports/results_full.csv` — All scores in flat format
- `reports/results_pivot.csv` — Models x Benchmarks pivot
- `reports/results_table.md` — Markdown table for paper
- `reports/group_averages.csv` — Macro averages by group
- `reports/best_per_benchmark.csv` — Winner per benchmark
- `reports/summary.md` — Executive summary
- `reports/findings_for_paper.md` — Scientific conclusions
- `reports/figures/` — PNG plots (heatmap, radar, bar charts)

## Adding New Benchmarks

1. Create `src/eval/tasks/my_benchmark.py`:
```python
from src.eval.tasks.base_task import BaseTask

class MyBenchmarkTask(BaseTask):
    def load_data(self, config):
        # Return list of dicts with task examples
        ...

    def get_gold_label(self, example):
        # Return expected answer
        ...

    def parse_prediction(self, raw_prediction):
        # Extract answer from model output
        ...
```

2. Register in `src/eval/tasks/__init__.py`
3. Add prompt template in `src/eval/prompt_templates.py`
4. Add entry in `configs/eval/benchmarks.yaml`
