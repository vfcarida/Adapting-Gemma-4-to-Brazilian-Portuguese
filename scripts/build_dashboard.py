#!/usr/bin/env python3
"""Build results dashboard from eval_results.json.

Generates a concise comparison table and identifies winners per benchmark.
Designed to be run after evaluation completes to quickly see what worked.

Usage:
    python3 scripts/build_dashboard.py                    # Default: reports/eval_results.json
    python3 scripts/build_dashboard.py --input path/to/results.json
    python3 scripts/build_dashboard.py --format markdown  # Output format: markdown, csv, terminal
    python3 scripts/build_dashboard.py --group brasil_geral  # Filter by group

Output:
    - Terminal: colored summary table
    - reports/dashboard.md: Full markdown comparison
    - reports/dashboard.csv: Machine-readable results
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(input_path: str) -> list[dict]:
    """Load eval_results.json."""
    path = Path(input_path)
    if not path.exists():
        print(f"ERROR: Results file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def flatten_results(results: list[dict]) -> list[dict]:
    """Flatten nested result structure into flat rows."""
    rows = []
    for model_result in results:
        model_name = model_result.get("model_name", "unknown")
        for mode_key, benchmarks in model_result.get("benchmarks", {}).items():
            for bench_name, bench_result in benchmarks.items():
                metrics = bench_result.get("metrics", {})
                primary_metric = bench_result.get("metric_name", "accuracy")
                score = metrics.get(primary_metric, 0)
                rows.append(
                    {
                        "model": model_name,
                        "think_mode": mode_key.replace("think_", ""),
                        "benchmark": bench_name,
                        "group": bench_result.get("group", "other"),
                        "metric": primary_metric,
                        "score": score,
                        "n_examples": bench_result.get("num_examples", 0),
                        "inference_time": bench_result.get("inference_time_sec", 0),
                    }
                )
    return rows


def build_comparison_table(rows: list[dict], group_filter: str | None = None) -> str:
    """Build markdown comparison table."""
    if group_filter:
        rows = [r for r in rows if r["group"] == group_filter]

    if not rows:
        return "No results found."

    # Get unique models and benchmarks
    models = sorted(set(r["model"] for r in rows))
    benchmarks = sorted(set(r["benchmark"] for r in rows))

    # Build score matrix
    scores = {}
    for r in rows:
        key = (r["model"], r["benchmark"], r["think_mode"])
        scores[key] = r["score"]

    # Find best per benchmark (for highlighting)
    best_per_bench = {}
    for bench in benchmarks:
        bench_scores = [(m, scores.get((m, bench, "off"), 0)) for m in models]
        if bench_scores:
            best_model = max(bench_scores, key=lambda x: x[1])
            best_per_bench[bench] = best_model[0]

    # Build table
    lines = []
    lines.append("# Results Dashboard\n")

    # Header
    header = "| Model | " + " | ".join(benchmarks) + " | Avg |"
    separator = "|" + "---|" * (len(benchmarks) + 2)
    lines.append(header)
    lines.append(separator)

    # Rows
    for model in models:
        model_scores = []
        cells = [f"**{model}**"]
        for bench in benchmarks:
            score = scores.get((model, bench, "off"), None)
            if score is not None:
                model_scores.append(score)
                # Bold if best
                if best_per_bench.get(bench) == model:
                    cells.append(f"**{score:.3f}**")
                else:
                    cells.append(f"{score:.3f}")
            else:
                cells.append("—")

        avg = sum(model_scores) / len(model_scores) if model_scores else 0
        cells.append(f"{avg:.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    # Winners summary
    lines.append("\n## Winners per Benchmark\n")
    for bench in benchmarks:
        winner = best_per_bench.get(bench, "—")
        best_score = max((scores.get((m, bench, "off"), 0) for m in models), default=0)
        lines.append(f"- **{bench}**: {winner} ({best_score:.3f})")

    # Group averages
    groups = sorted(set(r["group"] for r in rows))
    if len(groups) > 1:
        lines.append("\n## Group Averages\n")
        lines.append("| Model | " + " | ".join(groups) + " |")
        lines.append("|" + "---|" * (len(groups) + 1))
        for model in models:
            cells = [f"**{model}**"]
            for group in groups:
                group_scores = [
                    r["score"]
                    for r in rows
                    if r["model"] == model and r["group"] == group and r["think_mode"] == "off"
                ]
                if group_scores:
                    cells.append(f"{sum(group_scores) / len(group_scores):.3f}")
                else:
                    cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def build_terminal_output(rows: list[dict], group_filter: str | None = None) -> str:
    """Build concise terminal output."""
    if group_filter:
        rows = [r for r in rows if r["group"] == group_filter]

    if not rows:
        return "No results found."

    models = sorted(set(r["model"] for r in rows))
    benchmarks = sorted(set(r["benchmark"] for r in rows))

    # Score lookup
    scores = {}
    for r in rows:
        scores[(r["model"], r["benchmark"])] = r["score"]

    lines = []
    lines.append("╔══════════════════════════════════════════════════════════╗")
    lines.append("║              Evaluation Results Dashboard                ║")
    lines.append("╚══════════════════════════════════════════════════════════╝")
    lines.append("")

    # Compact table
    max_model_len = max(len(m) for m in models)

    header = (
        f"{'Model':<{max_model_len}} | "
        + " | ".join(f"{b[:8]:>8}" for b in benchmarks)
        + f" | {'Avg':>6}"
    )
    lines.append(header)
    lines.append("─" * len(header))

    for model in models:
        model_scores = []
        cells = [f"{model:<{max_model_len}}"]
        for bench in benchmarks:
            score = scores.get((model, bench))
            if score is not None:
                model_scores.append(score)
                cells.append(f"{score:>8.3f}")
            else:
                cells.append(f"{'—':>8}")
        avg = sum(model_scores) / len(model_scores) if model_scores else 0
        cells.append(f"{avg:>6.3f}")
        lines.append(" | ".join(cells))

    # Overall ranking
    lines.append("")
    lines.append("── Overall Ranking (by average score) ──")
    model_avgs = []
    for model in models:
        ms = [scores.get((model, b), 0) for b in benchmarks if (model, b) in scores]
        avg = sum(ms) / len(ms) if ms else 0
        model_avgs.append((model, avg))
    model_avgs.sort(key=lambda x: x[1], reverse=True)
    for i, (model, avg) in enumerate(model_avgs, 1):
        lines.append(f"  {i}. {model}: {avg:.4f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build results dashboard")
    parser.add_argument(
        "--input", type=str, default="reports/eval_results.json", help="Path to eval_results.json"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="terminal",
        choices=["terminal", "markdown", "csv"],
        help="Output format",
    )
    parser.add_argument("--group", type=str, default=None, help="Filter by benchmark group")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    args = parser.parse_args()

    results = load_results(args.input)
    rows = flatten_results(results)

    if not rows:
        print("No evaluation results found.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "terminal":
        print(build_terminal_output(rows, group_filter=args.group))
    elif args.format == "markdown":
        md = build_comparison_table(rows, group_filter=args.group)
        output_path = output_dir / "dashboard.md"
        with open(output_path, "w") as f:
            f.write(md)
        print(f"Dashboard written to: {output_path}")
        print(md)
    elif args.format == "csv":
        output_path = output_dir / "dashboard.csv"
        # Write flat CSV
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written to: {output_path} ({len(rows)} rows)")

    # Always save markdown dashboard alongside
    if args.format != "markdown":
        md = build_comparison_table(rows, group_filter=args.group)
        with open(output_dir / "dashboard.md", "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
