# Best Practices for Language Adaptation via Continued Pre-Training (2024-2025)

Research compilation for the Gemma 4 PT-BR Adaptation project.
Sources: Recent papers on CPT methodology, Portuguese LLM development, model merging, and distributed training.

---

## 1. Scientific Rigor for CPT Experiments

### 1.1 Multi-Seed Evaluation

**Recommendation: Minimum 3 seeds, ideally 5 for final results.**

- For multiple-choice benchmarks with ~200 items, 3 seeds suffice to get stable estimates (std across seeds is typically < 0.5% accuracy).
- For generation tasks (open-ended), 5 seeds recommended because variance is higher.
- Report: mean +/- 95% bootstrap CI (1000 resamples). This is more informative than just standard deviation.
- For significance between two systems: paired bootstrap test or McNemar's test at the item level.
- **Power analysis**: With 180 ENEM items, you can detect a 5% absolute difference with power=0.8 at alpha=0.05 (binomial test). For smaller effects, aggregate across benchmarks.

**Key reference**: Sabiá-1 (Pires et al., 2304.07880) used single-seed few-shot evaluation because the benchmarks were large enough for stable estimates. Sabiá-3 (2410.12049) similarly reports point estimates on large test sets.

### 1.2 Proper Ablation Methodology for PEFT Methods

**Recommendation: Control for effective trainable parameters, not just method.**

- When comparing LoRA (r=64) vs DoRA (r=64) vs QLoRA (r=64): ensure same effective parameter count. DoRA adds magnitude vectors, making it slightly larger.
- Control variables: same data, same LR schedule, same total tokens, same batch size.
- Vary ONE thing at a time. Your plan (B1-B3 in step 4) correctly does this.
- Report trainable param count and total FLOPs for each variant.
- **Critical**: LoRA alpha should be set to `alpha = 2 * r` as default (scaling factor = alpha/r = 2). This is the most common setting in recent literature.

### 1.3 Learning Rate Selection

**Recommendation: Yes, do a small LR sweep per method. Minimum 3 values.**

- The "Continual Pre-Training: How to (re)warm your model?" paper (Gupta et al., 2308.04014) shows:
  - Re-warming the LR is critical for CPT. Without it, learning is too slow.
  - Optimal peak LR for CPT is typically 30-50% of the original pre-training peak LR.
  - For Gemma 4 (original peak LR likely ~1e-4 to 3e-4): try CPT peak LR of 5e-5, 1e-4, 2e-4.
  - For LoRA specifically: higher LRs work (1e-4 to 3e-4) since only adapters are updated.

- The "Scaling Data-Constrained LMs" paper (Muennighoff et al., 2305.16264) confirms: warmup + cosine decay is standard; alternatively, WSD (Warmup-Stable-Decay) schedule avoids committing to a fixed token budget upfront.

- **Practical approach**: Do a 3-point LR sweep on 500M tokens (1/10 of pilot), pick best by validation perplexity, then run full pilot with that LR.

### 1.4 Token Budget Scaling Laws

**Key finding from literature:**

The Chinchilla scaling law (compute-optimal) suggests `N_tokens ~= 20 * N_params` for training from scratch. For CPT:

- **Sabiá-1 formula**: Used only 3% of original pre-training budget and got strong results. For LLaMA-65B (~1.4T tokens original), they used ~40B tokens of Portuguese.
- **Practical rule for CPT**: 5-15% of original pre-training tokens is typically sufficient for language adaptation.
- **Gemma 4 E4B** (4B active params, ~12B total MoE): If trained on ~5T tokens originally, CPT budget of 20-50B tokens (0.4-1%) can work, but 100-250B would be more thorough.
- **Diminishing returns**: The Ibrahim et al. (2403.08763) paper shows that for continual pre-training, you get ~80% of the benefit in the first 20B tokens for a 7B model, with log-linear improvement thereafter.

**Scaling formula for CPT (approximate)**:
```
L(T) = L_0 + (L_target - L_0) * (1 - exp(-T / T_characteristic))
T_characteristic ~= 0.02 * T_original_pretrain
```

### 1.5 Measuring Catastrophic Forgetting Properly

**Recommendation: Multi-level forgetting measurement, not just 3 benchmarks.**

Your current plan (MMLU, HellaSwag, ARC samples) is a start but insufficient for a paper. Best practices:

1. **Perplexity on held-out English data**: Track val perplexity on a held-out slice of English (e.g., 10K docs from C4). This is the most sensitive early-warning metric.

2. **Diverse task battery**:
   - Knowledge: MMLU (full or 5-shot, 500+ items)
   - Commonsense: HellaSwag, WinoGrande
   - Reasoning: ARC-Challenge, GSM8K (subset)
   - Code: HumanEval (even if not primary focus, shows general capability)
   - Safety: TruthfulQA (subset)

3. **Track per-domain degradation**: Not all MMLU subjects degrade equally. Report by subject group.

4. **Continuous monitoring during training**: Evaluate every 1-2B tokens, not just at end. Plot forgetting curves.

5. **Forgetting metric**:
   ```
   Forgetting_rate = (score_baseline - score_after_cpt) / score_baseline
   ```
   Report this per benchmark. Acceptable: < 5% relative degradation with replay.

---

## 2. Engineering Best Practices for Distributed Training on GCP

### 2.1 Batch Size Calculation for A100 80GB with Gemma 4 MoE

**For Gemma 4 E4B (~4B active params, MoE architecture):**

- **Per-GPU memory budget** (A100 80GB):
  - Model weights (bf16): ~8GB (4B active params * 2 bytes)
  - Optimizer states (AdamW): ~24GB (with ZeRO-2 sharding, divided across GPUs)
  - Activations (with gradient checkpointing): ~20-30GB depending on seq_len
  - Remaining for batch: ~20-30GB

- **Recommended micro-batch size**:
  - seq_len=4096: micro_batch=4-8 per GPU
  - seq_len=8192: micro_batch=2-4 per GPU

- **Global batch size target**: 256-512 sequences (1M-2M tokens per step)
  - With 4 GPUs, micro_batch=4, gradient_accumulation=16-32

- **Critical for MoE**: Expert routing means memory is less predictable. Start conservative (micro_batch=2) and increase.

### 2.2 Flash Attention 2 vs SDPA for Gemma 4

**Recommendation: Use Flash Attention 2 (via `flash_attn` package).**

- Flash Attention 2 is 2-3x faster than SDPA for long sequences (4096+) on A100.
- Gemma 4 uses standard multi-head attention (not sliding window in base layers), so FA2 works directly.
- SDPA (torch.nn.functional.scaled_dot_product_attention) is the fallback if FA2 has compatibility issues.
- **For MoE models**: FA2 works identically since attention is independent of expert routing.
- Set `attn_implementation="flash_attention_2"` in HuggingFace `from_pretrained()`.

### 2.3 DeepSpeed ZeRO-2 vs ZeRO-3

**Recommendation: ZeRO-2 for your setup (4x A100 with LoRA/PEFT).**

| Aspect | ZeRO-2 | ZeRO-3 |
|--------|--------|--------|
| What's sharded | Optimizer states + gradients | + model parameters |
| Communication overhead | Lower | Higher (all-gather on every forward) |
| When to use | Model fits in 1 GPU memory | Model does NOT fit in 1 GPU |
| Best for | 4-8 GPU, model < 20B | 16+ GPU, model > 30B |
| LoRA compatibility | Excellent | Can have issues with frozen params |

- Gemma 4 E4B active params (~4B) easily fits in a single A100 80GB in bf16.
- ZeRO-2 shards optimizer states and gradients, giving ~3x memory reduction for optimizer.
- ZeRO-3 adds communication overhead that's not worth it unless you need parameter sharding.
- **For full fine-tuning of E4B**: ZeRO-2 is still sufficient on 4x A100 80GB.
- **For the 31B dense variant**: ZeRO-3 would be needed.

### 2.4 Checkpoint Frequency and GCS Sync Patterns

**Recommendation:**

- **Checkpoint every 500-1000 steps** (roughly every 30-60 min of training).
- **Keep last 3 checkpoints locally** on the instance SSD for fast recovery.
- **Async upload to GCS**: Use a background thread or `gsutil -m rsync` triggered after each checkpoint.
- **Pattern**:
  ```
  gs://bucket/experiment_name/
    checkpoint-step-1000/
    checkpoint-step-2000/
    checkpoint-latest -> checkpoint-step-2000  (symlink/marker)
    config.yaml
    wandb_run_id.txt
  ```
- **Selective saving for PEFT**: Only save adapter weights (small, ~100MB for LoRA r=64). Full checkpoints only every 5000 steps.
- **Atomic writes**: Write to a temp path, then rename. This prevents corruption from preemption during write.

### 2.5 Preemption Recovery (State-of-the-Art)

**For GCP Spot/Preemptible VMs:**

1. **Metadata server polling**: GCP gives 30-second warning before preemption. Poll `http://metadata.google.internal/computeMetadata/v1/instance/preempted` every 5 seconds.

2. **On preemption signal**:
   - Save current training state (model, optimizer, scheduler, dataloader position, RNG states)
   - Upload to GCS
   - Exit gracefully

3. **On restart**:
   - Check GCS for latest checkpoint
   - Resume from exact position (including dataloader state via `skip_batches`)
   - Verify loss continuity (first loss after resume should match expected trajectory)

4. **Infrastructure**:
   - Use a managed instance group with auto-restart policy
   - Or use GKE with `google.com/preemptible-gpu` toleration and a Job controller
   - Alternative: Use GCP A3 (H100) reserved instances for critical final runs

5. **Dataloader state**: Use a deterministic sampler (seed + step count). On resume, skip `step * batch_size` samples.

### 2.6 Real-Time Monitoring Metrics

**Must-track (log every step):**

| Metric | Why |
|--------|-----|
| `train/loss` | Primary optimization signal |
| `train/perplexity` | More interpretable than loss |
| `train/grad_norm` | Detect instability (spikes = problems) |
| `train/learning_rate` | Verify schedule correctness |
| `train/tokens_per_second` | Throughput monitoring |
| `train/gpu_utilization` | Detect stragglers/inefficiency |
| `train/gpu_memory_allocated` | OOM early warning |
| `system/gpu_temperature` | Thermal throttling detection |

**Track every N steps (e.g., 100):**

| Metric | Why |
|--------|-----|
| `val/loss_pt` | Portuguese validation perplexity |
| `val/loss_en` | English retention monitoring |
| `train/expert_load_balance` | MoE-specific: detect expert collapse |
| `train/router_z_loss` | MoE routing stability |

**Track every eval checkpoint:**

| Metric | Why |
|--------|-----|
| `eval/enem_accuracy` | Quick PT downstream signal |
| `eval/mmlu_sample_accuracy` | Forgetting early warning |

---

## 3. Evaluation Protocol for Portuguese LLM Papers

### 3.1 Benchmarks Used in Recent Papers

**From Sabiá-3 (2410.12049), PTT5-v2 (2406.10806), Juru (2403.18140), and IberianLLM (2512.10545):**

| Benchmark | Type | Items | Used By |
|-----------|------|-------|---------|
| ENEM (2022-2024) | Multi-domain MCQ | ~180/year | Sabiá-1/2/3, Juru |
| BLUEX | Vestibular MCQ | ~360 | Sabiá-1/2/3 |
| OAB Exams | Legal MCQ | ~240 | Sabiá-1/2/3 |
| ASSIN2 (STS + RTE) | Semantic similarity/NLI | 2448 | PTT5-v2 |
| TweetSentBR | Sentiment analysis | 15k | PTT5-v2 |
| HateBR | Hate speech detection | 7k | Multiple |
| FaQuAD | Reading comprehension | 900 | Multiple |
| SPARQA-PT | QA (translated) | - | Sabiá-3 |
| MATH-PT | Math reasoning (translated) | - | Sabiá-3 |
| IFEval-PT | Instruction following | - | Sabiá-3 |
| **IberoBench** | Multi-task PT/ES suite | - | IberianLLM |
| **Poeta** (14 tasks) | Suite by Sabiá team | varies | Sabiá-1 |

**For your project, recommended minimum set:**
- ENEM (2022-2024) - 3 years as separate evaluations
- BLUEX
- OAB
- ASSIN2 (both STS and RTE)
- At least one generation task (FaQuAD or MATH-PT)
- English retention: MMLU, HellaSwag, ARC-Challenge

### 3.2 Few-Shot Examples

| Task Type | Recommended Shots | Rationale |
|-----------|-------------------|-----------|
| Multiple choice (ENEM, BLUEX) | 3-shot or 5-shot | Sabiá papers use 3-shot for ENEM |
| NLI (ASSIN2 RTE) | 5-shot | Standard in literature |
| Semantic similarity | 0-shot (regression) | Not applicable for MCQ format |
| Open QA | 3-shot | Shows expected format |
| Math/Reasoning | 5-8 shot with CoT | Needed for complex reasoning |

**Important**: Keep few-shot examples FIXED across all model comparisons. Draw from training set, never from test.

### 3.3 Validation vs Test Split

**Recommendation: Use validation for development, test ONCE for final paper.**

- During hyperparameter search (LR, replay ratio, alpha): evaluate on dev/validation split
- Report final numbers on held-out test split
- If a benchmark has no official split (e.g., ENEM): use year-based splits (dev=2019-2021, test=2022-2024)
- **Critical**: Document this split in the paper. Many PT benchmarks don't have official splits.

### 3.4 Handling Wrong-Format Outputs

**Standard approach (from lm-evaluation-harness and Sabiá papers):**

1. **Regex cascade**: Try multiple patterns to extract answer:
   ```
   Priority 1: Exact match of single letter (A/B/C/D/E)
   Priority 2: "alternativa X" or "resposta: X"
   Priority 3: First letter in the generation
   Priority 4: Mark as unparseable (score = 0)
   ```

2. **Report parse failure rate**: If > 5% of items fail to parse, the model may need format-specific prompting.

3. **Logprob scoring as backup**: For MCQ tasks, always run both generate+parse AND logprob scoring. Report both; logprob is more reliable but less representative of real use.

4. **Never manually fix outputs**. Document the parser and make it available for reproducibility.

### 3.5 lm-evaluation-harness vs Custom Eval

**Recommendation: Use lm-evaluation-harness as the backbone, with custom tasks.**

- lm-evaluation-harness (EleutherAI) is the de facto standard for reproducibility.
- Add custom task YAMLs for Portuguese benchmarks not already included.
- Advantages: standardized caching, few-shot selection, logging, comparability with other papers.
- **BUT**: For PT-specific normalization (accents, format), you need custom `process_results` functions.
- The PTT5-v2 and IberianLLM papers use custom evaluation code. For maximum rigor, run BOTH and report which framework was used.

---

## 4. Data Mixing for CPT

### 4.1 Optimal Replay Ratio

**From recent literature:**

| Paper | Replay Ratio | Result |
|-------|-------------|--------|
| Ibrahim et al. (2403.08763) | 5-20% replay | 10% is sweet spot for balanced retention |
| Gupta et al. (2308.04014) | Various | Replay + LR rewarm is key |
| Sabiá-1 (2304.07880) | 0% (pure PT) | Accepted forgetting as trade-off |
| Juru (2403.18140) | 0% (pure legal PT) | Confirmed forgetting on general tasks |
| IberianLLM (2512.10545) | XDoGE-weighted | Language weights optimized by proxy model |

**Recommendation: 10-15% English replay.**

- 5% is too little for a 7B+ model doing language shift
- 15% is safe but reduces effective PT tokens/step
- 10% is the most commonly reported sweet spot
- **Quality matters**: Use high-quality English (Wikipedia, textbooks, cleaned web) not random C4

### 4.2 Should Code Be Included in Replay?

**Recommendation: Yes, 5-10% of replay should be code.**

- Code helps preserve reasoning capabilities (confirmed by multiple studies)
- Code tokens are high-information-density and help maintain structured thinking
- Include a mix: Python, SQL, and markdown documentation
- The "Scaling Data-Constrained LMs" paper explicitly shows code augmentation helps even for language tasks
- **Practical split of replay**: 70% English text + 20% code + 10% math/science

### 4.3 Document-Level vs Paragraph-Level Packing

**Recommendation: Document-level packing with attention masking.**

- **Document-level packing**: Pack multiple complete documents into one sequence until hitting max_seq_len.
- Superior to paragraph-level because:
  1. Preserves long-range coherence within documents
  2. Model learns document structure (intro, body, conclusion)
  3. Better for downstream tasks requiring multi-paragraph reasoning

- **Implementation**: Pack documents greedily, separated by EOS tokens. Use attention masks so documents don't attend to each other.

### 4.4 Separator Tokens Between Packed Documents

**Recommendation: Yes, use the model's EOS token (or a dedicated separator if available).**

- Gemma 4 uses `<end_of_turn>` and `<eos>`. Use `<eos>` between packed documents.
- **Cross-document attention masking**: Best practice is to prevent attention across document boundaries within a packed sequence.
- If attention masking is too complex to implement: at minimum use the EOS separator, which provides a soft boundary signal.
- **Without separator or masking**: Models can "hallucinate" connections between unrelated documents, degrading quality.

### 4.5 Curriculum Learning

**Recommendation: Mild curriculum is helpful but not critical.**

From recent evidence:

- **Not recommended**: Hard curriculum (easy->hard by perplexity or length). Evidence is mixed, implementation is complex, and sorting is expensive.
- **Mildly recommended**: Domain curriculum — start with cleaner data (Wikipedia, books, quality web) then mix in noisier web data later.
- **What works better**: Data quality filtering upfront (keep only high-quality) + random shuffling. The IberianLLM paper (2512.10545) uses quality-based filtering with STEM classifiers.
- **The paper at 2509.08824** (Portuguese corpus building) specifically shows that language-specific filtering pipelines (education, STEM, toxicity classifiers) matter more than ordering.

---

## 5. Residual Merge / Model Merging

### 5.1 Task Arithmetic vs TIES-Merge vs DARE

**For language adaptation specifically:**

| Method | Best For | Mechanism |
|--------|----------|-----------|
| Task Arithmetic (Ilharco et al., 2212.04089) | Simple, single capability transfer | `merged = base + alpha * (adapted - base)` |
| TIES-Merge | Resolving sign conflicts between multiple task vectors | Trim, Elect Sign, Merge |
| DARE (Yu et al., 2311.03099) | Merging multiple fine-tuned models, sparsification | Drop 90-99% of deltas, rescale |

**Recommendation for language adaptation: Task Arithmetic is most appropriate.**

- You're doing: `adapted_instruct = cpt_model + alpha * (instruct_model - base_model)`
- This is a single task vector application (instruction-following capability transfer)
- TIES and DARE are for combining MULTIPLE task vectors (e.g., merging code + chat + reasoning)
- Task Arithmetic is simpler, well-understood, and sufficient for your use case
- DARE is overkill for 2-model merging but interesting for ablation study

### 5.2 Optimal Alpha Values

**From literature and community experiments:**

| Scenario | Alpha Range | Sweet Spot |
|----------|-------------|------------|
| Language CPT + instruct vector | 0.5 - 1.0 | 0.7 - 0.8 |
| Domain adaptation + instruct | 0.6 - 1.0 | 0.8 |
| Full fine-tune merge | 0.3 - 0.7 | 0.5 |
| LoRA merge (adapter weights) | 0.8 - 1.2 | 1.0 |

- **Your plan** (alpha sweep 0.5-1.2) is well-calibrated.
- Alpha > 1.0 can work when the task vector is "diluted" by the CPT process.
- Start evaluation at 0.7 and 0.8, then expand if needed.
- **Per-layer alpha**: Advanced technique where different layers get different alpha. Not standard yet but shows promise in community experiments (deeper layers often need lower alpha).

### 5.3 Float32 vs BFloat16 for Merge

**Recommendation: Perform merge computation in float32, save result in bfloat16.**

- Subtraction of large similar numbers (instruct - base) can lose precision in bf16.
- The delta vector is small (within +/- 0.002 per DARE paper) — bf16 has only ~3 decimal digits of precision.
- **Practical approach**:
  ```python
  # Load in float32 for computation
  base = load_model(dtype=torch.float32)
  instruct = load_model(dtype=torch.float32)
  cpt = load_model(dtype=torch.float32)

  # Compute in float32
  task_vector = instruct - base
  merged = cpt + alpha * task_vector

  # Cast back for storage/inference
  merged = merged.to(torch.bfloat16)
  ```
- Memory implication: Need ~3x model size in float32 during merge. For 4B active params: ~48GB. Fits in CPU RAM easily.

### 5.4 Validating Merge Quality Before Full Evaluation

**Quick sanity checks (< 5 minutes):**

1. **Perplexity on 100 PT documents**: Should be close to CPT model (not base).
2. **Perplexity on 100 EN documents**: Should be close to instruct model (not degraded).
3. **3 manual prompts in PT**: Does it follow instructions? Does it respond in Portuguese?
4. **3 manual prompts in EN**: Is it still coherent?
5. **Weight statistics**: Check that merged weights have similar mean/std as source models. Extreme values indicate merge failure.
6. **ENEM 5-item sanity test**: Run 5 known-correct ENEM items. If 0/5, merge failed.

---

## 6. Reproducibility Standards for ML Papers

### 6.1 Required Artifacts to Release

| Artifact | Priority | Notes |
|----------|----------|-------|
| Model weights | Required | HuggingFace Hub or equivalent |
| Training code | Required | GitHub with clear README |
| Training config (all hyperparams) | Required | YAML/JSON, versioned |
| Evaluation code | Required | Including custom parsers |
| Data preprocessing code | Required | But NOT the data if licensed |
| Raw evaluation results (per-item) | Strongly recommended | Enables reanalysis |
| Training logs (WandB export) | Recommended | Loss curves, LR, etc. |
| Data composition details | Required | Sources, proportions, filters |
| Tokenizer | Required if modified | - |
| Environment (pip freeze / Docker) | Recommended | For exact reproduction |

### 6.2 Documenting Compute Budget

**Standard format for papers:**

```
Training was performed on [N] x NVIDIA [A100-80GB/H100] GPUs
for [X] hours (total [Y] GPU-hours).

Estimated cost: $[Z] at [on-demand/spot] pricing.
Estimated CO2: [W] kg (using [tool/region] for calculation).

Token throughput: [T] tokens/second/GPU.
Total tokens processed: [B] billion.
Effective FLOPs: [F] (using 6 * N_params * N_tokens approximation).
```

- Use ML CO2 Impact calculator or codecarbon for emissions
- Report both wall-clock time and GPU-hours (they differ with preemption)
- For PEFT: also report trainable parameter count and % of total

### 6.3 WandB/MLflow Logging Standards

**Minimum logging requirements:**

```yaml
# Config to log at run start
wandb_config:
  model_name: "gemma-4-e4b"
  method: "lora_r64"
  dataset: "aurora-pt-v2"
  dataset_tokens: 20_000_000_000
  replay_ratio: 0.10
  replay_sources: ["english_wiki", "code_python", "english_books"]
  learning_rate: 1e-4
  lr_schedule: "cosine_with_warmup"
  warmup_steps: 1000
  batch_size_global: 512
  seq_length: 4096
  precision: "bf16"
  optimizer: "adamw"
  weight_decay: 0.01
  grad_clip: 1.0
  deepspeed_stage: 2
  flash_attention: true
  gradient_checkpointing: true
  seed: 42
  hardware: "4x_a100_80gb"
```

**Per-step logging**: loss, lr, grad_norm, tokens/sec, gpu_memory
**Per-eval logging**: all benchmark scores + raw predictions as artifacts

### 6.4 Statistical Tests Expected by Reviewers

| Claim Type | Required Test | Notes |
|------------|---------------|-------|
| "A is better than B" | Paired bootstrap (n=10000) or McNemar's test | At item level |
| "Method X consistently helps" | Wilcoxon signed-rank across benchmarks | Non-parametric |
| Multiple comparisons | Holm-Bonferroni correction | When comparing >2 systems |
| Confidence interval | Bootstrap 95% CI | Always report |
| Effect size | Cohen's d or relative improvement % | For practical significance |

**Do NOT just report accuracy deltas without significance testing.** Reviewers at ACL/EMNLP/NeurIPS will reject.

**Bootstrap procedure**:
```python
def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000):
    """Test if system A is significantly better than B."""
    diffs = scores_a - scores_b
    observed_diff = diffs.mean()

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(diffs, size=len(diffs), replace=True)
        bootstrap_diffs.append(sample.mean())

    p_value = (np.array(bootstrap_diffs) <= 0).mean()
    ci_low, ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])
    return p_value, ci_low, ci_high
```

---

## 7. Specific Recommendations for This Project

### 7.1 Priority Actions Based on Literature

1. **Adopt the Ibrahim et al. (2403.08763) recipe**: LR re-warming + cosine re-decay + 10% replay. This is the strongest empirical result for CPT.

2. **Use WSD (Warmup-Stable-Decay) schedule** instead of fixed cosine: Allows extending training without committing to a token budget upfront. Better for iterative experimentation.

3. **Build quality filters** following the 2509.08824 paper: Language-specific STEM classifiers, toxicity filters, and education content detectors. Quality > quantity.

4. **Use IberoBench** (from IberianLLM paper) as a comprehensive Portuguese evaluation suite in addition to your Poeta-style benchmarks.

5. **XDoGE for data mixing** (from IberianLLM, 2512.10545): Train a small proxy model to find optimal language weights. More principled than fixed ratios.

### 7.2 Gaps in Your Current Plan

| Gap | Recommendation |
|-----|----------------|
| No LR sweep before pilot | Add 500M-token mini-sweep (3 LR values) |
| Forgetting measured only at end | Add eval every 2B tokens during training |
| No code in replay | Add 5-10% code to replay mix |
| No attention masking for packing | Implement cross-document attention masks |
| No per-domain forgetting analysis | Break MMLU into subject groups |
| Single eval framework | Run both custom eval AND lm-eval-harness |

### 7.3 Key Papers to Cite

| Paper | ID | Relevance |
|-------|-----|-----------|
| Sabiá-1: Monolingual PT pretraining | 2304.07880 | Direct predecessor, PT CPT methodology |
| Sabiá-2 | 2403.09887 | Scaling up PT models |
| Sabiá-3 | 2410.12049 | SOTA Portuguese, evaluation benchmarks |
| Continual Pre-Training: How to Rewarm | 2308.04014 | LR re-warming methodology |
| Scaling Data-Constrained LMs | 2305.16264 | Multi-epoch training, data repetition |
| Ibrahim et al. CPT recipe | 2403.08763 | LR rewarm + replay + re-decay |
| Adapting LLMs via Reading Comprehension | 2309.09530 | CPT with enriched data |
| DoReMi | 2305.10429 | Domain reweighting for data mixing |
| IberianLLM / XDoGE | 2512.10545 | Portuguese CPT with optimized mixing |
| Portuguese web corpus for LLMs | 2509.08824 | Data pipeline methodology |
| PTT5-v2 | 2406.10806 | Portuguese T5 CPT, subtle effect of config |
| Juru (legal PT CPT) | 2403.18140 | Domain CPT forgetting evidence |
| Task Arithmetic | 2212.04089 | Model merging via task vectors |
| DARE | 2311.03099 | Delta parameter sparsification for merge |
| Minitron (Pruning + KD) | 2407.14679 | Efficient model compression alternative |

---

## 8. Summary of Key Numbers

| Parameter | Recommended Value | Source |
|-----------|-------------------|--------|
| Replay ratio | 10% (range: 5-15%) | Ibrahim et al., community consensus |
| Code in replay | 5-10% of replay | Scaling Data-Constrained LMs |
| Peak LR for CPT | 30-50% of original pre-train LR | Gupta et al. |
| LoRA alpha/r ratio | 2.0 | Community standard |
| Warmup steps | 1-5% of total steps | Standard |
| Merge alpha | 0.7-0.8 for language CPT | Community experiments |
| Seeds for final eval | 3-5 | Statistical guidance |
| Min items for significance | 180+ (5% effect, power=0.8) | Binomial power analysis |
| Checkpoint frequency | Every 500-1000 steps | Preemption recovery |
| Acceptable forgetting | < 5% relative degradation | Community threshold |
| Document separator | EOS token + attention mask | Best practice |
| Merge precision | float32 compute, bf16 storage | Numerical stability |
