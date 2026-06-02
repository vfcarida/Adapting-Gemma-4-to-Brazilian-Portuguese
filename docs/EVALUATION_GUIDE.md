# Evaluation Guide

## Benchmark Suite

### Brasil Geral (General Knowledge)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| ENEM 2022 | Multiple choice (5 options) | Accuracy | `maritaca-ai/enem` |
| ENEM 2023 | Multiple choice (5 options) | Accuracy | `maritaca-ai/enem` |
| ENEM 2024 | Multiple choice (5 options) | Accuracy | `maritaca-ai/enem` |
| BLUEX | Multiple choice (university entrance) | Accuracy | `Se7enB/bluex` |

### Semantica (Semantic Understanding)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| ASSIN2-RTE | Textual entailment | Accuracy | `assin2` |
| ASSIN2-STS | Semantic similarity (1-5) | Pearson | `assin2` |
| CoPA-PT | Causal reasoning (2 choices) | Accuracy | `Se7enB/copa_pt` |
| MRPC-PT | Paraphrase detection | Accuracy | Local JSONL |
| RTE-PT | Textual entailment | Accuracy | Local JSONL |

### Classificacao Social (Social Classification)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| HateBR | Hate speech detection | Macro-F1 | `Se7enB/HateBR` |
| TweetSentBR | Sentiment (pos/neg/neu) | Macro-F1 | `Se7enB/TweetSentBR` |

### Juridico (Legal)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| OAB-Bench | Bar exam questions (4 options) | Accuracy | Local JSONL |

### Cultura (Cultural Knowledge)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| BRoverbs | Proverb completion (4 options) | Accuracy | Local JSONL |
| Tuguesice-PT | Language/culture trivia | Accuracy | Local JSONL |

### Seguranca (Safety)

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| DoNotAnswer-PT | Harmful question refusal | Refusal Rate | Local JSONL |

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
