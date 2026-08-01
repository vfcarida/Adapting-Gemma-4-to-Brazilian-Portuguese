"""Aggregate evaluation results across multiple seeds.

Computes mean, std, and confidence intervals for each metric across
seed runs. This is essential for reporting statistically meaningful results.

Usage:
    python -m src.eval.aggregate_seeds --results-dir reports/ --seeds 42 123 456
    python -m src.eval.aggregate_seeds --output-dir outputs/cpt_pilot
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def aggregate_seed_results(
    results_paths: list[Path],
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Aggregate evaluation results from multiple seed runs.

    Args:
        results_paths: List of paths to eval_results.json files (one per seed).
        confidence_level: Confidence level for intervals.

    Returns:
        Dict with aggregated metrics: mean, std, CI per benchmark.
    """
    all_results = []
    for path in results_paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                all_results.append(data)
        else:
            logger.warning(f"Results file not found: {path}")

    if not all_results:
        logger.error("No results files found to aggregate")
        return {}

    n_seeds = len(all_results)
    logger.info(f"Aggregating results from {n_seeds} seeds")

    # Collect scores per benchmark across seeds
    benchmark_scores: dict[str, dict[str, list[float]]] = {}

    for seed_result in all_results:
        # Handle both direct list format and nested format
        results_list = (
            seed_result if isinstance(seed_result, list) else seed_result.get("results", [])
        )

        for model_result in results_list:
            model_name = model_result.get("model_name", "unknown")
            for mode_key, benchmarks in model_result.get("benchmarks", {}).items():
                for bench_name, bench_data in benchmarks.items():
                    key = f"{model_name}/{mode_key}/{bench_name}"
                    metrics = bench_data.get("metrics", {})
                    metric_name = bench_data.get("metric_name", "accuracy")
                    score = metrics.get(metric_name, metrics.get("accuracy", 0))
                    benchmark_scores.setdefault(key, {}).setdefault("scores", []).append(score)
                    benchmark_scores[key]["metric"] = metric_name
                    benchmark_scores[key]["group"] = bench_data.get("group", "")

    # Compute statistics
    aggregated = {}
    alpha = (1 - confidence_level) / 2

    for key, data in benchmark_scores.items():
        scores = np.array(data["scores"])
        n = len(scores)

        if n < 2:
            ci_lower = ci_upper = float(scores[0]) if n == 1 else 0.0
            std = 0.0
        else:
            std = float(np.std(scores, ddof=1))
            # t-based CI for small n
            from scipy.stats import t as t_dist

            t_crit = t_dist.ppf(1 - alpha, df=n - 1)
            margin = t_crit * std / np.sqrt(n)
            ci_lower = float(np.mean(scores) - margin)
            ci_upper = float(np.mean(scores) + margin)

        aggregated[key] = {
            "mean": float(np.mean(scores)),
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_seeds": n,
            "scores": [float(s) for s in scores],
            "metric": data["metric"],
            "group": data["group"],
        }

    return {
        "n_seeds": n_seeds,
        "confidence_level": confidence_level,
        "benchmarks": aggregated,
    }


def format_aggregated_table(aggregated: dict[str, Any]) -> str:
    """Format aggregated results as markdown table.

    Shows mean ± std and 95% CI for each benchmark.
    """
    lines = [
        f"# Aggregated Results ({aggregated['n_seeds']} seeds, "
        f"{aggregated['confidence_level'] * 100:.0f}% CI)\n",
        "| Model / Benchmark | Mean | Std | CI Lower | CI Upper |",
        "|---|---|---|---|---|",
    ]

    for key, data in sorted(aggregated.get("benchmarks", {}).items()):
        lines.append(
            f"| {key} | {data['mean']:.4f} | {data['std']:.4f} | "
            f"{data['ci_lower']:.4f} | {data['ci_upper']:.4f} |"
        )

    return "\n".join(lines)


def main():
    """CLI entry point for seed aggregation."""
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate multi-seed evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="reports",
        help="Directory containing eval_results.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output dir (will look for *_seed*/eval_results.json)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    # Find results files
    results_paths = []
    if args.output_dir:
        base = Path(args.output_dir)
        for seed in args.seeds:
            # Look in seed-specific output dirs
            candidates = [
                base.parent / f"{base.name}_seed{seed}" / "eval_results.json",
                base / f"seed{seed}" / "eval_results.json",
                Path("reports") / f"eval_results_seed{seed}.json",
            ]
            for c in candidates:
                if c.exists():
                    results_paths.append(c)
                    break
    else:
        results_dir = Path(args.results_dir)
        for seed in args.seeds:
            path = results_dir / f"eval_results_seed{seed}.json"
            if path.exists():
                results_paths.append(path)
            else:
                # Try generic path
                generic = results_dir / "eval_results.json"
                if generic.exists() and generic not in results_paths:
                    results_paths.append(generic)

    if not results_paths:
        print(f"No results files found. Looked for seeds {args.seeds}")
        return

    # Aggregate
    aggregated = aggregate_seed_results(results_paths, args.confidence)

    # Save
    output_dir = Path(args.results_dir if not args.output_dir else args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "aggregated_results.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    table = format_aggregated_table(aggregated)
    with open(output_dir / "aggregated_results.md", "w") as f:
        f.write(table)

    print(table)
    print(f"\nSaved to {output_dir}/aggregated_results.json")


if __name__ == "__main__":
    main()
