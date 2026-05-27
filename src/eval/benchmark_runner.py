"""
src/eval/benchmark_runner.py
────────────────────────────
Unified evaluation orchestrator for PT-BR benchmarks.

Invokes lm-evaluation-harness tasks, supports predictive caching,
parametric think mode (think_on / think_off), and exports results
as JSON for downstream report generation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.utils.config_utils import load_config, parse_args, parse_overrides, validate_config
from src.utils.hf_utils import authenticate_hf
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


class BenchmarkRunner:
    """Orchestrate evaluation of models on PT-BR benchmarks.

    Parameters
    ----------
    config : dict[str, Any]
        Evaluation configuration (from ``configs/eval.yml``).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        validate_config(config, required_keys=["eval", "tasks", "models"])
        self.config = config
        self.eval_cfg = config["eval"]
        self.tasks = config["tasks"]
        self.models = config["models"]
        self.think_modes = config.get("think_modes", [{"name": "think_off", "enable_thinking": False}])

        self.cache_dir = Path(self.eval_cfg.get("cache_dir", "./cache/eval_cache"))
        self.results_dir = Path(self.eval_cfg.get("results_dir", "./reports/eval_results"))
        self.use_cache = self.eval_cfg.get("use_cache", True)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> dict[str, Any]:
        """Run all evaluations across all models, tasks, and think modes.

        Returns
        -------
        dict[str, Any]
            Nested results: ``{model_label: {think_mode: {task_name: metrics}}}``
        """
        all_results: dict[str, Any] = {}

        for model_spec in self.models:
            model_id = model_spec["id"]
            model_label = model_spec.get("label", model_id)
            category = model_spec.get("category", "unknown")

            logger.info("═══ Evaluating: %s (%s) ═══", model_label, category)
            all_results[model_label] = {"model_id": model_id, "category": category}

            for think_mode in self.think_modes:
                mode_name = think_mode["name"]
                enable_thinking = think_mode["enable_thinking"]

                logger.info("── Think mode: %s (enable=%s) ──", mode_name, enable_thinking)
                mode_results: dict[str, Any] = {}

                for task_spec in self.tasks:
                    task_name = task_spec["name"]
                    result = self.run_single(
                        model_id=model_id,
                        model_label=model_label,
                        task_spec=task_spec,
                        enable_thinking=enable_thinking,
                        mode_name=mode_name,
                    )
                    mode_results[task_name] = result

                all_results[model_label][mode_name] = mode_results

        # Save aggregate results
        aggregate_path = self.results_dir / "all_results.json"
        with open(aggregate_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("All results saved → %s", aggregate_path)

        return all_results

    def run_single(
        self,
        model_id: str,
        model_label: str,
        task_spec: dict[str, Any],
        enable_thinking: bool = False,
        mode_name: str = "think_off",
    ) -> dict[str, Any]:
        """Run a single evaluation (model × task × think_mode).

        Parameters
        ----------
        model_id : str
            HuggingFace model ID or local path.
        model_label : str
            Human-readable model name.
        task_spec : dict
            Task specification from config.
        enable_thinking : bool
            Whether to enable Gemma 4 thinking mode.
        mode_name : str
            Think mode name for logging / caching.

        Returns
        -------
        dict[str, Any]
            Evaluation metrics.
        """
        task_name = task_spec["name"]
        num_fewshot = task_spec.get("num_fewshot", 0)

        # ── Check cache ──────────────────────────────────────────────
        cache_key = self._cache_key(model_id, task_name, mode_name)
        if self.use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.info("Cache hit: %s × %s × %s", model_label, task_name, mode_name)
                return cached

        logger.info("Running: %s × %s × %s (fewshot=%d)", model_label, task_name, mode_name, num_fewshot)

        # ── Run evaluation via lm-eval harness ───────────────────────
        try:
            result = self._invoke_harness(
                model_id=model_id,
                task_name=task_name,
                num_fewshot=num_fewshot,
                enable_thinking=enable_thinking,
            )
        except Exception as e:
            logger.error("Evaluation failed for %s × %s: %s", model_label, task_name, e)
            result = {"error": str(e), "task": task_name, "model": model_label}

        # ── Save individual result ───────────────────────────────────
        result_path = self.results_dir / f"{model_label}_{task_name}_{mode_name}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # ── Update cache ─────────────────────────────────────────────
        self._save_cache(cache_key, result)

        return result

    def _invoke_harness(
        self,
        model_id: str,
        task_name: str,
        num_fewshot: int,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        """Invoke lm-evaluation-harness via subprocess.

        Uses temperature=0.0 for deterministic generation.
        """
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_id},trust_remote_code=True",
            "--tasks", task_name,
            "--num_fewshot", str(num_fewshot),
            "--batch_size", str(self.eval_cfg.get("batch_size", "auto")),
            "--device", self.eval_cfg.get("device", "cuda"),
            "--output_path", str(self.results_dir / f"{task_name}_raw"),
            "--log_samples",
        ]

        # Add generation kwargs for temperature
        gen_kwargs = f"temperature=0.0"
        if enable_thinking:
            gen_kwargs += ",do_sample=False"
        cmd.extend(["--gen_kwargs", gen_kwargs])

        logger.info("Command: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout per task
            )
            if result.returncode != 0:
                logger.error("lm_eval stderr: %s", result.stderr[:500])
                return {"error": result.stderr[:500], "returncode": result.returncode}

            # Parse output JSON
            output_dir = self.results_dir / f"{task_name}_raw"
            return self._parse_harness_output(output_dir, task_name)

        except subprocess.TimeoutExpired:
            return {"error": "Evaluation timed out (1h limit)"}
        except FileNotFoundError:
            logger.error("lm-eval not installed. Run: pip install lm-eval")
            return {"error": "lm-eval not installed"}

    def _parse_harness_output(self, output_dir: Path, task_name: str) -> dict[str, Any]:
        """Parse the JSON output from lm-evaluation-harness."""
        # lm_eval outputs results in a structured directory
        for json_file in output_dir.rglob("results*.json"):
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            if "results" in data:
                return data["results"].get(task_name, data["results"])
            return data

        return {"error": f"No results found in {output_dir}"}

    def _cache_key(self, model_id: str, task_name: str, mode_name: str) -> str:
        """Generate a deterministic cache key."""
        raw = f"{model_id}|{task_name}|{mode_name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _load_cache(self, key: str) -> dict[str, Any] | None:
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, data: dict[str, Any]) -> None:
        cache_path = self.cache_dir / f"{key}.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args(description="Gemma 4 — Benchmark Runner")
    overrides = parse_overrides(args.override) if args.override else {}
    config = load_config(args.config, overrides=overrides)

    if args.dry_run:
        print(json.dumps(config, indent=2, default=str))
        sys.exit(0)

    set_global_seed(config.get("eval", {}).get("seed", 42))
    authenticate_hf()

    runner = BenchmarkRunner(config)
    runner.run_all()


if __name__ == "__main__":
    main()
