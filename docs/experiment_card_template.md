# Experiment Card: [EXPERIMENT_NAME]

## Metadata
| Field | Value |
|-------|-------|
| Date | YYYY-MM-DD |
| Author | |
| Hardware | e.g., 1x A100-80GB |
| Duration | e.g., 24h |
| Cost (USD) | e.g., $XX |
| Config | `configs/train/XXX.yaml` |
| Seed(s) | 42, 123, 456 |
| Git SHA | `git rev-parse HEAD` |
| Branch | main |

## Hypothesis
> What are we testing? State clearly and falsifiably.

## Method
- **Base Model**: e.g., google/gemma-4-E4B
- **Training**: CPT / SFT / Merge
- **Data**: e.g., Aurora-PT (85% PT + 15% EN replay)
- **Steps**: e.g., 5000
- **Effective Batch (tokens)**: e.g., 262K tokens/step
- **Learning Rate**: e.g., 2e-4 (cosine, 5% warmup)
- **PEFT**: LoRA r=64, alpha=128

## Ablation Variables
| Variable | Value | Alternatives Tested |
|----------|-------|---------------------|
| e.g., replay_ratio | 15% | 0%, 5%, 10%, 15% |

## Results

### Primary Metrics
| Benchmark | Baseline | This Exp | Delta |
|-----------|----------|----------|-------|
| ENEM 2024 | 0.XX | 0.XX | +X.X% |
| MMLU EN (retention) | 0.XX | 0.XX | -X.X% |

### Group Averages
| Group | Baseline | This Exp |
|-------|----------|----------|
| brasil_geral | | |
| semantica | | |
| retencao_en | | |

### Confidence Intervals (bootstrap, n=1000, α=0.05)
| Benchmark | Score | 95% CI |
|-----------|-------|--------|
| | | [lower, upper] |

## Training Curves
- [ ] Loss curve (train + val) — attach or link to W&B
- [ ] English perplexity (forgetting monitor)
- [ ] Learning rate schedule
- [ ] Throughput (tokens/sec)

## Observations
-
-

## Conclusion
> Was the hypothesis supported? What did we learn?

## Follow-up Actions
- [ ]
- [ ]

## Reproducibility Checklist
- [ ] Config committed to git
- [ ] Random seeds documented
- [ ] Data version/hash recorded
- [ ] Model checkpoint saved
- [ ] Hardware/environment documented
- [ ] Results match within ±1% on re-run
- [ ] No data contamination (checked via `run_contamination_checks.sh`)
- [ ] Bootstrap CIs computed (not just point estimates)
