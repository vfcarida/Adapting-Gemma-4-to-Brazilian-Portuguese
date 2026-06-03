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
import re
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.metrics import compute_metrics_for_task
from src.eval.prompt_templates import get_prompt_template
from src.utils.config_utils import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


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
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.think_modes = eval_cfg.get("think_modes", ["off"])
        self.strip_think = eval_cfg.get("strip_think_from_output", True)

    def run_all(self, model_id: str, model_name: str) -> dict[str, Any]:
        """Run all enabled benchmarks for a single model.

        Evaluates in both think_on and think_off modes (if configured),
        using cache to skip already-computed benchmarks.

        Args:
            model_id: HuggingFace model ID or local path.
            model_name: Human-readable name for reporting.

        Returns:
            Dict with model metadata and nested benchmark results.
        """
        set_seed(self.seed)
        benchmarks = self.config.get("benchmarks", {})
        results = {"model_id": model_id, "model_name": model_name, "benchmarks": {}}

        for think_mode in self.think_modes:
            mode_key = f"think_{think_mode}"
            results["benchmarks"][mode_key] = {}

            for bench_name, bench_cfg in benchmarks.items():
                if not bench_cfg.get("enabled", True):
                    continue

                logger.info(f"Running {bench_name} (think={think_mode}) on {model_name}")

                # Check cache first (avoids re-running expensive inference)
                cache_key = self._cache_key(model_id, bench_name, think_mode)
                cached = self._load_cache(cache_key)
                if cached:
                    logger.info(f"  Using cached result for {bench_name}")
                    results["benchmarks"][mode_key][bench_name] = cached
                    continue

                # Run the benchmark fresh
                bench_result = self._run_single_benchmark(
                    model_id, bench_name, bench_cfg, think_mode
                )
                results["benchmarks"][mode_key][bench_name] = bench_result

                # Persist to cache
                self._save_cache(cache_key, bench_result)

        return results

    def _run_single_benchmark(
        self, model_id: str, bench_name: str, bench_cfg: dict, think_mode: str
    ) -> dict[str, Any]:
        """Run a single benchmark: load data → format prompts → inference → score.

        Args:
            model_id: Model to evaluate.
            bench_name: Benchmark name (for logging).
            bench_cfg: Benchmark configuration dict.
            think_mode: "on" or "off".

        Returns:
            Dict with task info, metrics, timing, and sample predictions.
        """
        from src.eval.tasks import load_task

        # Step 1: Load task data (from HF Hub or local JSONL)
        task = load_task(bench_cfg["task"])
        examples = task.load_data(bench_cfg)
        prompt_template = get_prompt_template(bench_cfg["task"], bench_cfg.get("num_shots", 0))

        # Step 2: Format all prompts with Gemma 4 chat template
        prompts = []
        for example in examples:
            prompt = prompt_template.format_prompt(example, think_mode=think_mode)
            prompts.append(prompt)

        # Step 3: Run model inference (batched)
        start_time = time.time()
        if self.use_vllm:
            predictions = self._inference_vllm(model_id, prompts, think_mode)
        else:
            predictions = self._inference_hf(model_id, prompts, think_mode)
        inference_time = time.time() - start_time

        # Step 4: Parse predictions (strip thinking, extract answer)
        parsed_predictions = []
        for pred in predictions:
            # Remove <think>...</think> blocks before parsing (think_on mode)
            if self.strip_think and think_mode == "on":
                pred = re.sub(r"<think>.*?</think>", "", pred, flags=re.DOTALL).strip()
            parsed = task.parse_prediction(pred)
            parsed_predictions.append(parsed)

        # Step 5: Compute metrics against gold labels
        gold_labels = [task.get_gold_label(ex) for ex in examples]
        metrics = compute_metrics_for_task(bench_cfg["metric"], parsed_predictions, gold_labels)

        result = {
            "task": bench_cfg["task"],
            "group": bench_cfg["group"],
            "metric_name": bench_cfg["metric"],
            "metrics": metrics,
            "num_examples": len(examples),
            "inference_time_sec": inference_time,
            "think_mode": think_mode,
            # Save a sample of raw predictions for qualitative analysis
            "raw_predictions": predictions[:10],
        }

        logger.info(f"  {bench_name}: {metrics}")
        return result

    def _inference_hf(self, model_id: str, prompts: list[str], think_mode: str) -> list[str]:
        """Run inference using HuggingFace Transformers (generate API).

        Loads the model once and processes all prompts in batches.
        Slower than vLLM but works without additional dependencies.

        Args:
            model_id: Model to load (HF ID or local path).
            prompts: List of formatted prompts.
            think_mode: Thinking mode (affects generation but not logic here).

        Returns:
            List of raw model text outputs (one per prompt).
        """
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        predictions = []
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Inference"):
            batch = prompts[i : i + self.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
                model.device
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    # Temperature 0 = greedy decoding (deterministic)
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for j, output in enumerate(outputs):
                # Only decode newly generated tokens (skip the input prompt)
                input_len = inputs["input_ids"][j].shape[0]
                new_tokens = output[input_len:]
                pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
                predictions.append(pred)

        # Free GPU memory after inference
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

    def _cache_key(self, model_id: str, bench_name: str, think_mode: str) -> str:
        """Generate deterministic cache key from evaluation parameters.

        The key includes the seed so that changing the seed invalidates
        the cache (different random states → different few-shot examples).
        """
        key_str = f"{model_id}|{bench_name}|{think_mode}|{self.seed}"
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

    all_results = []

    if model_id:
        # Evaluate a single specified model
        result = runner.run_all(model_id, model_id.split("/")[-1])
        all_results.append(result)
    else:
        # Evaluate all models from config
        for model_cfg in config.get("models_to_evaluate", []):
            mid = model_cfg["id"]
            name = model_cfg.get("name", mid)
            # Skip models that aren't available (local paths that don't exist)
            if not Path(mid).exists() and "/" not in mid:
                logger.warning(f"Skipping {mid}: not found")
                continue
            result = runner.run_all(mid, name)
            all_results.append(result)

    # Save consolidated results
    output_dir = Path(config.get("report", {}).get("output_dir", "reports"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_results.json", "w") as f:
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
