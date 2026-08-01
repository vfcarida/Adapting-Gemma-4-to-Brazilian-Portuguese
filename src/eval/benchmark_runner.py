"""Unified benchmark runner with caching and batch inference.

This module orchestrates the evaluation of multiple models across multiple
benchmarks. It handles:
- Loading task data and formatting prompts
- Running inference (HuggingFace Transformers or vLLM)
- Parsing model outputs into structured predictions
- Computing metrics per benchmark
- Caching results to avoid redundant computation
- Supporting think_on / think_off modes for Gemma 4

Usage:
    # From CLI
    python -m src.eval.benchmark_runner --config configs/eval/benchmarks.yaml

    # From Python
    from src.eval.benchmark_runner import run_evaluation
    results = run_evaluation("configs/eval/benchmarks.yaml", model_id="google/gemma-4-E4B-it")
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.bootstrap_ci import bootstrap_ci, wilson_score_interval
from src.eval.metrics import compute_metrics_for_task
from src.eval.prompt_templates import PromptBuilder, get_prompt_template, strip_thought
from src.utils.config_utils import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _is_unavailable_local_checkpoint(model_id: str) -> bool:
    """Detect a local checkpoint path that hasn't been produced yet.

    Model IDs in `models_to_evaluate` are either HF Hub IDs (`org/repo`,
    e.g. `google/gemma-4-E4B-it`) or local checkpoint paths produced by this
    repo's training stages (always `outputs/...`, e.g.
    `outputs/cpt_pilot/final`). The old check (`not Path(mid).exists() and
    "/" not in mid`) skipped nothing in practice, since every Hub ID and
    every local path here contains a "/" — so untrained local checkpoints
    were always attempted and crashed `from_pretrained`. This distinguishes
    "looks like a local path" from "looks like a Hub ID" instead of relying
    on "/" as the signal.
    """
    looks_local = model_id.startswith(("outputs/", "./", "../")) or Path(model_id).is_absolute()
    return looks_local and not Path(model_id).exists()


class EmptyBenchmarkDataError(RuntimeError):
    """Raised when a benchmark's data loader returns zero examples.

    Previously an empty example list propagated silently into
    `compute_metrics_for_task`, where `macro_f1`/`pearson`/etc. would raise
    an uncaught sklearn/scipy exception deep in the call stack (or, before
    the `macro_f1` empty-guard fix, crash the ENTIRE evaluation run for
    every other benchmark that had already computed results, since
    `eval_results.json` is only written at the very end). This gives a
    single, clear, catchable signal instead.
    """


class BenchmarkRunner:
    """Run evaluation benchmarks with caching and configurable inference.

    The runner iterates over all enabled benchmarks in the config for each
    model, using cached results when available. This makes it safe to
    interrupt and resume evaluation runs.

    Args:
        config: Full evaluation config dict (from configs/eval/benchmarks.yaml).

    Attributes:
        cache_dir: Directory for storing inference cache (keyed by model+benchmark+seed).
        think_modes: List of thinking modes to evaluate (["off"] or ["off", "on"]).
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        eval_cfg = config["evaluation"]
        self.seed = eval_cfg.get("seed", 42)
        self.temperature = eval_cfg.get("temperature", 0.0)
        self.max_new_tokens = eval_cfg.get("max_new_tokens", 512)
        self.batch_size = eval_cfg.get("batch_size", 8)
        self.use_vllm = eval_cfg.get("use_vllm", False)
        self.cache_dir = Path(eval_cfg.get("cache_dir", "outputs/eval_cache"))
        report_cfg = config.get("report", {})
        self.bootstrap_n = report_cfg.get("bootstrap_n", 2000)
        self.confidence_level = report_cfg.get("confidence_level", 0.95)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.think_modes = eval_cfg.get("think_modes", ["off"])
        self.strip_think = eval_cfg.get("strip_think_from_output", True)

    def run_all(
        self,
        model_id: str,
        model_name: str,
        is_chat_model: bool = True,
        num_shots_override: int | None = None,
    ) -> dict[str, Any]:
        """Run all enabled benchmarks for a single model.

        Evaluates in both think_on and think_off modes (if configured),
        using cache to skip already-computed benchmarks. Loads the model
        once and reuses it across all benchmarks for efficiency.

        Args:
            model_id: HuggingFace model ID or local path.
            model_name: Human-readable name for reporting.
            is_chat_model: Whether this model is instruction-tuned (uses its
                tokenizer's chat template) or a base/CPT-only checkpoint
                (uses the plain few-shot protocol, no chat markup). Comes
                from `models_to_evaluate[].is_chat_model` in the config —
                previously ignored, so every model (including base
                checkpoints and non-chat baselines like Sabia-7B) was
                prompted with hardcoded Gemma 4 chat markup regardless.
            num_shots_override: If set, overrides each benchmark's
                `num_shots` for this model (from
                `models_to_evaluate[].num_shots_override`, e.g. forcing
                Sabia-7B to always use 5-shot).

        Returns:
            Dict with model metadata and nested benchmark results.
        """
        set_seed(self.seed)
        benchmarks = self.config.get("benchmarks", {})
        results = {"model_id": model_id, "model_name": model_name, "benchmarks": {}}

        # Check if any benchmarks need fresh inference (not cached)
        needs_inference = False
        for think_mode in self.think_modes:
            for bench_name, bench_cfg in benchmarks.items():
                if not bench_cfg.get("enabled", True):
                    continue
                cache_key = self._cache_key(
                    model_id, bench_name, think_mode, bench_cfg, num_shots_override
                )
                if not self._load_cache(cache_key):
                    needs_inference = True
                    break
            if needs_inference:
                break

        # Load model (and always the tokenizer, even under vLLM/logprob-only
        # paths — apply_chat_template only needs the tokenizer, and the
        # PromptBuilder below needs it to format prompts correctly for
        # whichever model is being evaluated).
        model_resources = None
        tokenizer = None
        if needs_inference:
            if not self.use_vllm:
                model_resources = self._load_model(model_id)
                tokenizer = model_resources["tokenizer"]
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # One PromptBuilder per model, reused across all benchmarks. Using
        # the model's OWN tokenizer.apply_chat_template (when is_chat_model)
        # means each model family (Gemma 4, Gemma 3/Gaia, ChatML/Tucano, ...)
        # gets its own correct chat template for free — no need to hardcode
        # a template per `prompt_builder` name from the config.
        prompt_builder = PromptBuilder(tokenizer=tokenizer, is_chat_model=is_chat_model)

        for think_mode in self.think_modes:
            mode_key = f"think_{think_mode}"
            results["benchmarks"][mode_key] = {}

            for bench_name, bench_cfg in benchmarks.items():
                if not bench_cfg.get("enabled", True):
                    continue

                logger.info(f"Running {bench_name} (think={think_mode}) on {model_name}")

                # Check cache first (avoids re-running expensive inference)
                cache_key = self._cache_key(
                    model_id, bench_name, think_mode, bench_cfg, num_shots_override
                )
                cached = self._load_cache(cache_key)
                if cached:
                    logger.info(f"  Using cached result for {bench_name}")
                    results["benchmarks"][mode_key][bench_name] = cached
                    continue

                # Run the benchmark fresh
                try:
                    bench_result = self._run_single_benchmark(
                        model_id,
                        bench_name,
                        bench_cfg,
                        think_mode,
                        model_resources=model_resources,
                        prompt_builder=prompt_builder,
                        num_shots_override=num_shots_override,
                    )
                except EmptyBenchmarkDataError as e:
                    logger.error(f"  Skipping {bench_name}: {e}")
                    continue
                results["benchmarks"][mode_key][bench_name] = bench_result

                # Persist to cache
                self._save_cache(cache_key, bench_result)

        # Free model after all benchmarks for this model
        if model_resources:
            del model_resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def _run_single_benchmark(
        self,
        model_id: str,
        bench_name: str,
        bench_cfg: dict,
        think_mode: str,
        model_resources: dict | None = None,
        prompt_builder: PromptBuilder | None = None,
        num_shots_override: int | None = None,
    ) -> dict[str, Any]:
        """Run a single benchmark: load data → format prompts → inference → score.

        Args:
            model_id: Model to evaluate.
            bench_name: Benchmark name (for logging).
            bench_cfg: Benchmark configuration dict.
            think_mode: "on" or "off".
            model_resources: Pre-loaded model and tokenizer dict (optional).
            prompt_builder: Model-specific PromptBuilder (chat template vs
                plain few-shot). Required for correct prompting — see
                `run_all`'s docstring.
            num_shots_override: Per-model override for `bench_cfg["num_shots"]`.

        Returns:
            Dict with task info, metrics, timing, and sample predictions.
        """
        from src.eval.tasks import load_task

        # Step 1: Load task data (from HF Hub or local JSONL)
        task = load_task(bench_cfg["task"])
        examples = task.load_data(bench_cfg)
        if not examples:
            raise EmptyBenchmarkDataError(
                f"{bench_name} (task={bench_cfg['task']}, hub_id={bench_cfg.get('hub_id')}) "
                "loaded 0 examples — check the dataset ID/config/split, network access, "
                "and HF auth (gated datasets need HF_TOKEN)."
            )
        num_shots = (
            num_shots_override if num_shots_override is not None else bench_cfg.get("num_shots", 0)
        )

        # Step 2: Split few-shot examples from evaluation examples
        # Use first N examples as few-shot demonstrations, evaluate on the rest
        few_shot_examples = []
        eval_examples = examples
        if num_shots > 0 and len(examples) > num_shots + 10:
            few_shot_examples = examples[:num_shots]
            eval_examples = examples[num_shots:]

        prompt_template = get_prompt_template(
            bench_cfg["task"], num_shots, few_shot_examples=few_shot_examples
        )

        max_samples = bench_cfg.get("max_samples")
        if max_samples and len(eval_examples) > max_samples:
            eval_examples = eval_examples[:max_samples]

        # Step 3: Format all prompts, using the model-appropriate
        # PromptBuilder (chat template for instruction-tuned models, plain
        # few-shot text for base/CPT-only checkpoints and non-chat
        # baselines) instead of the previous hardcoded Gemma 4 markup.
        prompts = []
        for example in eval_examples:
            prompt = prompt_template.format_prompt(
                example, think_mode=think_mode, prompt_builder=prompt_builder
            )
            prompts.append(prompt)

        # Step 4: Run model inference
        # Use logprob scoring for MCQ tasks (more stable than generate+parse)
        use_logprob = (
            self.config.get("evaluation", {}).get("use_logprob", False)
            and bench_cfg.get("metric") == "accuracy"
            and not self.use_vllm
            and model_resources is not None
            and think_mode == "off"  # Logprob doesn't work with thinking
        )

        start_time = time.time()
        if use_logprob:
            # Extract answer options for each example
            answer_options = []
            for ex in eval_examples:
                options = ex.get("options", [])
                if options:
                    # Use letter labels (A, B, C, D, E)
                    answer_options.append([chr(65 + i) for i in range(len(options))])
                else:
                    answer_options.append(["A", "B", "C", "D"])
            predictions = self._inference_logprob(model_resources, prompts, answer_options)
            inference_method = "logprob"
        elif self.use_vllm:
            predictions = self._inference_vllm(model_id, prompts, think_mode)
            inference_method = "vllm"
        else:
            predictions = self._inference_hf(
                model_id, prompts, think_mode, model_resources=model_resources
            )
            inference_method = "generate"
        inference_time = time.time() - start_time

        # Step 4: Parse predictions (strip thinking, extract answer). Think
        # blocks can appear even in "off" mode (a model may emit them
        # unprompted) or be truncated by max_new_tokens before the closing
        # tag — strip unconditionally rather than only when think_mode=="on"
        # and only when the closing tag is present. Uses strip_thought()
        # (handles both Gemma 4's real <|channel>thought...<channel|>
        # markers and the legacy <think>...</think> convention some other
        # model families use) rather than a narrower inline regex.
        parsed_predictions = []
        stripped_outputs = []
        for pred in predictions:
            if self.strip_think:
                pred = strip_thought(pred)
            stripped_outputs.append(pred)
            parsed = task.parse_prediction(pred)
            parsed_predictions.append(parsed)

        # Step 5: Compute metrics against gold labels
        gold_labels = [task.get_gold_label(ex) for ex in eval_examples]
        metrics = compute_metrics_for_task(bench_cfg["metric"], parsed_predictions, gold_labels)

        # Step 6: Confidence intervals. Accuracy-like metrics on these
        # (mostly small, <1000-item) benchmarks use the Wilson score
        # interval (holds nominal coverage at small N — see
        # bootstrap_ci.py's module docstring); everything else falls back to
        # item-resampling bootstrap on the metric itself.
        ci = None
        if "n_correct" in metrics and "n_total" in metrics and metrics["n_total"] > 0:
            ci = {
                "method": "wilson",
                **wilson_score_interval(
                    metrics["n_correct"], metrics["n_total"], confidence_level=self.confidence_level
                ),
            }
        elif len(eval_examples) >= 5:
            try:
                boot = bootstrap_ci(
                    parsed_predictions,
                    gold_labels,
                    lambda p, g, _m=bench_cfg["metric"]: compute_metrics_for_task(_m, p, g),
                    n_bootstrap=self.bootstrap_n,
                    confidence_level=self.confidence_level,
                    seed=self.seed,
                )
                primary = boot.get(bench_cfg["metric"])
                if primary:
                    ci = {"method": "bootstrap_percentile", **primary}
            except Exception as e:  # noqa: BLE001 - CI is best-effort, never fail the run over it
                logger.warning(f"  {bench_name}: CI computation failed ({e})")

        # Step 7: Persist full per-item records (prompt hash, raw output,
        # parsed prediction, gold, correctness) to a companion JSONL file.
        # Previously only the first 10 raw predictions were kept anywhere,
        # which made every downstream statistical claim (paired bootstrap,
        # McNemar, per-item error analysis) structurally impossible to
        # compute after the fact.
        items_path = (
            self.cache_dir
            / f"{self._cache_key(model_id, bench_name, think_mode, bench_cfg, num_shots_override)}_items.jsonl"
        )
        with open(items_path, "w") as f:
            for ex, prompt, raw, stripped, parsed, gold in zip(
                eval_examples,
                prompts,
                predictions,
                stripped_outputs,
                parsed_predictions,
                gold_labels,
            ):
                correct = str(parsed).strip().upper() == str(gold).strip().upper()
                f.write(
                    json.dumps(
                        {
                            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
                            "raw_output": raw,
                            "stripped_output": stripped,
                            "parsed_prediction": parsed
                            if isinstance(parsed, (str, int, float, type(None)))
                            else str(parsed),
                            "gold_label": gold
                            if isinstance(gold, (str, int, float, type(None)))
                            else str(gold),
                            "correct": correct,
                        },
                        default=str,
                    )
                    + "\n"
                )

        result = {
            "task": bench_cfg["task"],
            "group": bench_cfg["group"],
            "metric_name": bench_cfg["metric"],
            "metrics": metrics,
            "confidence_interval": ci,
            "num_examples": len(eval_examples),
            "inference_time_sec": inference_time,
            "inference_method": inference_method,
            "think_mode": think_mode,
            "num_few_shot": num_shots,
            "items_path": str(items_path),
            # Save a sample of raw predictions for qualitative analysis
            # (full per-item data lives in items_path above).
            "raw_predictions": predictions[:10],
        }

        logger.info(f"  {bench_name}: {metrics}")
        return result

    def _load_model(self, model_id: str) -> dict[str, Any]:
        """Load model and tokenizer once for reuse across benchmarks.

        Args:
            model_id: HuggingFace model ID or local path.

        Returns:
            Dict with 'model' and 'tokenizer' keys.
        """
        logger.info(f"Loading model for evaluation: {model_id}")
        # Left padding is required for batched causal-LM generation: with
        # right padding (the AutoTokenizer default), `output[input_len:]`
        # (used below to isolate generated tokens) slices the wrong region
        # for every non-longest prompt in the batch, since input_len is the
        # same padded width for all rows but the real content starts at
        # different offsets. Left padding keeps the real content
        # right-aligned so new tokens always start exactly at position
        # `input_len`.
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return {"model": model, "tokenizer": tokenizer}

    def _inference_hf(
        self,
        model_id: str,
        prompts: list[str],
        think_mode: str,
        model_resources: dict | None = None,
    ) -> list[str]:
        """Run inference using HuggingFace Transformers (generate API).

        Uses pre-loaded model if provided, otherwise loads on demand.

        Args:
            model_id: Model to load (HF ID or local path).
            prompts: List of formatted prompts.
            think_mode: Thinking mode (affects generation but not logic here).
            model_resources: Pre-loaded model/tokenizer dict (avoids reloading).

        Returns:
            List of raw model text outputs (one per prompt).
        """
        # Use pre-loaded model or load on demand
        if model_resources:
            model = model_resources["model"]
            tokenizer = model_resources["tokenizer"]
            should_cleanup = False
        else:
            resources = self._load_model(model_id)
            model = resources["model"]
            tokenizer = resources["tokenizer"]
            should_cleanup = True

        max_prompt_length = self.config.get("evaluation", {}).get("max_prompt_length", 4096)
        predictions = []
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Inference"):
            batch = prompts[i : i + self.batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # With left padding (see _load_model), every row's real content
            # ends exactly at position `input_len`, so newly generated
            # tokens always start there for every row in the batch.
            input_len = inputs["input_ids"].shape[1]
            for output in outputs:
                new_tokens = output[input_len:]
                pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
                predictions.append(pred)

        # Only free if we loaded on demand
        if should_cleanup:
            del model
            torch.cuda.empty_cache()

        return predictions

    def _inference_vllm(self, model_id: str, prompts: list[str], think_mode: str) -> list[str]:
        """Run inference using vLLM for 3-5x faster generation.

        vLLM uses PagedAttention and continuous batching for efficient
        LLM inference. Falls back to HF if vLLM is not installed.

        Args:
            model_id: Model to load.
            prompts: List of formatted prompts.
            think_mode: Thinking mode.

        Returns:
            List of raw model text outputs.
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            logger.warning("vLLM not available, falling back to HF inference")
            return self._inference_hf(model_id, prompts, think_mode)

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            seed=self.seed,
        )

        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=8192,
        )

        outputs = llm.generate(prompts, sampling_params)
        predictions = [o.outputs[0].text for o in outputs]

        del llm
        torch.cuda.empty_cache()
        return predictions

    def _inference_logprob(
        self,
        model_resources: dict[str, Any],
        prompts: list[str],
        answer_options: list[list[str]],
    ) -> list[str]:
        """Score MCQ tasks using log-probabilities instead of generation.

        For each prompt, compute the log-probability of each answer option
        (e.g., "A", "B", "C", "D") and select the highest. This is more
        stable than generation+parsing for classification tasks.

        Args:
            model_resources: Pre-loaded model and tokenizer.
            prompts: List of formatted prompts (ending before answer).
            answer_options: List of lists of possible answers per example.
                e.g., [["A", "B", "C", "D"], ["A", "B", "C", "D", "E"], ...]

        Returns:
            List of predicted answers (highest logprob option per example).
        """
        model = model_resources["model"]
        tokenizer = model_resources["tokenizer"]
        predictions = []

        for prompt, options in tqdm(
            zip(prompts, answer_options), total=len(prompts), desc="Logprob scoring"
        ):
            # Tokenize the prompt once; each option is scored by actually
            # APPENDING its tokens and reading the model's own predicted
            # log-probability for each of those tokens (teacher forcing).
            # The previous version ran `model(prompt_ids)` — identical for
            # every option, since the option was never appended to the
            # input — so every option always got the exact same
            # (prompt-only) next-token distribution; scoring was, in
            # effect, "which option string starts with the single most
            # likely next token", not "which option does the model prefer".
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

            best_option = options[0]
            best_logprob = float("-inf")

            for option in options:
                option_ids = tokenizer.encode(option, add_special_tokens=False)
                if not option_ids:
                    continue

                full_ids = torch.tensor([prompt_ids + option_ids], device=model.device)
                with torch.no_grad():
                    logits = model(full_ids).logits[0]  # [seq_len, vocab]

                # Position i's logits predict token i+1. To score the
                # option's tokens, look at logits starting one position
                # before the option begins (which predicts the option's
                # first token) through one before the option ends.
                start = len(prompt_ids) - 1
                end = len(prompt_ids) + len(option_ids) - 1
                option_logits = logits[start:end]  # [len(option_ids), vocab]
                log_probs = torch.nn.functional.log_softmax(option_logits, dim=-1)
                option_ids_t = torch.tensor(option_ids, device=model.device)
                token_logps = log_probs[torch.arange(len(option_ids)), option_ids_t]
                option_logprob = (token_logps.sum() / len(option_ids)).item()  # length-normalized

                if option_logprob > best_logprob:
                    best_logprob = option_logprob
                    best_option = option

            predictions.append(best_option)

        return predictions

    def _cache_key(
        self,
        model_id: str,
        bench_name: str,
        think_mode: str,
        bench_cfg: dict | None = None,
        num_shots_override: int | None = None,
    ) -> str:
        """Generate deterministic cache key from evaluation parameters.

        Includes every setting that changes the actual predictions, not
        just model/benchmark/think_mode/seed. Previously, editing
        `num_shots`, `max_new_tokens`, `temperature`, `use_logprob`, or even
        the benchmark's own `hub_id`/`subset`/`max_samples` silently reused
        a stale cached result computed under the OLD settings — despite
        docs/EVAL_PROTOCOL.md's claim that the cache invalidates whenever
        the generation config changes.
        """
        bench_cfg = bench_cfg or {}
        key_parts = [
            model_id,
            bench_name,
            think_mode,
            str(self.seed),
            str(
                num_shots_override
                if num_shots_override is not None
                else bench_cfg.get("num_shots", 0)
            ),
            str(self.max_new_tokens),
            str(self.temperature),
            str(self.batch_size),
            str(self.config.get("evaluation", {}).get("use_logprob", False)),
            str(self.use_vllm),
            str(bench_cfg.get("hub_id")),
            str(bench_cfg.get("subset")),
            str(bench_cfg.get("max_samples")),
            str(bench_cfg.get("metric")),
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_cache(self, cache_key: str) -> dict | None:
        """Load cached evaluation result, or None if not cached."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return None

    def _save_cache(self, cache_key: str, result: dict) -> None:
        """Persist evaluation result to cache as JSON."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2, default=str)


def run_evaluation(config_path: str, model_id: str | None = None) -> dict[str, Any]:
    """Run full evaluation pipeline from config file.

    Args:
        config_path: Path to evaluation config YAML.
        model_id: If provided, evaluate only this model. Otherwise,
                  evaluate all models listed in the config.

    Returns:
        Dict with list of all model results.
    """
    config = load_config(config_path)
    runner = BenchmarkRunner(config)

    output_dir = Path(config.get("report", {}).get("output_dir", "reports"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"

    # Load any existing results so a single-model run (or a resumed
    # multi-model run) MERGES in rather than clobbering. Previously every
    # call to run_evaluation overwrote eval_results.json wholesale, so
    # scripts/run_baselines.sh's loop of per-model `--model` invocations
    # left only the LAST model's results on disk.
    all_results = []
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    results_by_model = {r["model_id"]: r for r in all_results}

    def _run_one(
        mid: str, name: str, is_chat_model: bool = True, num_shots_override: int | None = None
    ):
        result = runner.run_all(
            mid, name, is_chat_model=is_chat_model, num_shots_override=num_shots_override
        )
        results_by_model[mid] = result

    if model_id:
        # Evaluate a single specified model. is_chat_model defaults to True
        # (the common case); use the config-driven path below to control it
        # explicitly for base/non-chat models.
        model_cfg = next(
            (m for m in config.get("models_to_evaluate", []) if m["id"] == model_id), {}
        )
        _run_one(
            model_id,
            model_cfg.get("name", model_id.split("/")[-1]),
            is_chat_model=model_cfg.get("is_chat_model", True),
            num_shots_override=model_cfg.get("num_shots_override"),
        )
    else:
        # Evaluate all enabled models from config
        for model_cfg in config.get("models_to_evaluate", []):
            if not model_cfg.get("enabled", True):
                logger.info(f"Skipping {model_cfg['id']}: enabled=false in config")
                continue
            mid = model_cfg["id"]
            name = model_cfg.get("name", mid)
            if _is_unavailable_local_checkpoint(mid):
                logger.warning(f"Skipping {mid}: local checkpoint not found (train it first)")
                continue
            _run_one(
                mid,
                name,
                is_chat_model=model_cfg.get("is_chat_model", True),
                num_shots_override=model_cfg.get("num_shots_override"),
            )

    all_results = list(results_by_model.values())
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return {"results": all_results}


def main():
    """CLI entry point for benchmark evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models in config
  python -m src.eval.benchmark_runner --config configs/eval/benchmarks.yaml

  # Evaluate a single model
  python -m src.eval.benchmark_runner --config configs/eval/benchmarks.yaml \\
      --model google/gemma-4-E4B-it
        """,
    )
    parser.add_argument("--config", type=str, default="configs/eval/benchmarks.yaml")
    parser.add_argument("--model", type=str, default=None, help="Single model to evaluate")
    args = parser.parse_args()
    run_evaluation(args.config, args.model)


if __name__ == "__main__":
    main()
