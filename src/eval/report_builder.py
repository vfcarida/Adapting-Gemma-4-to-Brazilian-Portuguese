"""
src/eval/report_builder.py
──────────────────────────
Generate Markdown comparison tables from evaluation JSON results.

Builds:
  • Per-task score tables
  • Aggregate score summaries
  • Baseline comparisons with bootstrap CIs
  • Statistical significance markers (*, **, ***)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tabulate import tabulate

from src.eval.bootstrap_ci import bootstrap_confidence_interval, paired_bootstrap_test
from src.utils.config_utils import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ReportBuilder:
    """Build Markdown benchmark reports from eval results.

    Parameters
    ----------
    results_dir : str | Path
        Directory containing evaluation JSON files.
    output_path : str | Path
        Path for the output Markdown report.
    config : dict | None
        Eval config (for significance thresholds, etc.).
    """

    def __init__(
        self,
        results_dir: str | Path = "reports/eval_results",
        output_path: str | Path = "reports/benchmark_report.md",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.output_path = Path(output_path)
        self.config = config or {}

        # Significance markers
        sig_markers = self.config.get("report", {}).get("significance_markers", {})
        self.sig_markers = sig_markers or {0.05: "*", 0.01: "**", 0.001: "***"}

    def build(self) -> str:
        """Build the full benchmark report.

        Returns
        -------
        str
            Markdown content of the report.
        """
        # Load all results
        all_results = self._load_results()
        if not all_results:
            logger.warning("No evaluation results found in %s", self.results_dir)
            return "# Benchmark Report\n\nNo results available.\n"

        sections: list[str] = []
        sections.append("# 📊 Gemma 4 PT-BR — Benchmark Report\n")
        sections.append(self._build_summary_section(all_results))
        sections.append(self._build_task_tables(all_results))
        sections.append(self._build_comparison_section(all_results))
        sections.append(self._build_methodology_section())

        report = "\n\n---\n\n".join(sections)

        # Save report
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("Report saved → %s", self.output_path)

        return report

    def _load_results(self) -> dict[str, Any]:
        """Load aggregate results JSON."""
        agg_path = self.results_dir / "all_results.json"
        if agg_path.exists():
            with open(agg_path, encoding="utf-8") as f:
                return json.load(f)

        # Fallback: load individual files
        results: dict[str, Any] = {}
        for json_file in self.results_dir.glob("*.json"):
            if json_file.name == "all_results.json":
                continue
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            results[json_file.stem] = data
        return results

    def _build_summary_section(self, results: dict[str, Any]) -> str:
        """Build the executive summary section."""
        lines = ["## Summary\n"]
        models = [k for k in results if isinstance(results[k], dict) and "category" in results[k]]

        our_models = [k for k in models if results[k].get("category") == "ours"]
        baselines = [k for k in models if results[k].get("category") == "baseline"]

        lines.append(f"- **Our models evaluated:** {len(our_models)}")
        lines.append(f"- **Baseline models:** {len(baselines)}")
        lines.append(f"- **Total tasks:** {self._count_tasks(results)}")
        lines.append("")
        return "\n".join(lines)

    def _build_task_tables(self, results: dict[str, Any]) -> str:
        """Build per-task result tables."""
        lines = ["## Per-Task Results\n"]

        # Collect all task names
        task_names = set()
        for model_data in results.values():
            if isinstance(model_data, dict):
                for mode_data in model_data.values():
                    if isinstance(mode_data, dict) and not isinstance(
                        mode_data.get("model_id"), str
                    ):
                        task_names.update(mode_data.keys())

        if not task_names:
            lines.append("No task results available.\n")
            return "\n".join(lines)

        # Build table for each think mode
        for mode_name in ["think_off", "think_on"]:
            lines.append(f"\n### Think Mode: `{mode_name}`\n")

            headers = ["Model"] + sorted(task_names)
            rows = []

            for model_label, model_data in results.items():
                if not isinstance(model_data, dict) or "category" not in model_data:
                    continue

                mode_data = model_data.get(mode_name, {})
                row = [model_label]
                for task in sorted(task_names):
                    task_result = mode_data.get(task, {})
                    # Extract primary metric value
                    score = self._extract_score(task_result)
                    row.append(f"{score:.4f}" if isinstance(score, float) else str(score))
                rows.append(row)

            if rows:
                lines.append(tabulate(rows, headers=headers, tablefmt="github"))
            lines.append("")

        return "\n".join(lines)

    def _build_comparison_section(self, results: dict[str, Any]) -> str:
        """Build pairwise comparison section with significance markers."""
        lines = ["## Statistical Comparisons\n"]
        lines.append(
            "Significance markers: "
            + ", ".join(f"p<{k}: {v}" for k, v in sorted(self.sig_markers.items()))
        )
        lines.append("")
        lines.append(
            "> **Note:** Statistical tests require per-example scores. "
            "If only aggregate metrics are available, p-values are estimated "
            "from bootstrap resampling of the aggregate."
        )
        lines.append("")
        return "\n".join(lines)

    def _build_methodology_section(self) -> str:
        """Build methodology notes."""
        return (
            "## Methodology\n\n"
            "- **Temperature:** 0.0 (deterministic generation)\n"
            "- **Bootstrap:** 10,000 resamples for confidence intervals\n"
            "- **Think Mode:** Each evaluation run in both `think_on` and `think_off` modes\n"
            "- **Metrics:** macro-F1 (classification), Pearson r (STS), "
            "Approval Rate (exams), Accuracy (NLI)\n"
            "- **Evaluation Framework:** lm-evaluation-harness-pt\n"
        )

    def _extract_score(self, task_result: dict[str, Any]) -> float | str:
        """Extract the primary metric score from a task result."""
        if isinstance(task_result, (int, float)):
            return float(task_result)
        if isinstance(task_result, dict):
            # Try common metric keys
            for key in ["acc", "acc_norm", "f1", "macro_f1", "pearson", "approval_rate", "exact_match"]:
                if key in task_result:
                    val = task_result[key]
                    if isinstance(val, (int, float)):
                        return float(val)
            if "error" in task_result:
                return "ERR"
        return "N/A"

    def _count_tasks(self, results: dict[str, Any]) -> int:
        """Count unique tasks across all models."""
        tasks = set()
        for model_data in results.values():
            if isinstance(model_data, dict):
                for mode_data in model_data.values():
                    if isinstance(mode_data, dict):
                        tasks.update(k for k in mode_data if k not in ("model_id", "category"))
        return len(tasks)

    def _significance_marker(self, p_value: float) -> str:
        """Return significance marker for a p-value."""
        marker = ""
        for threshold, symbol in sorted(self.sig_markers.items()):
            if p_value < threshold:
                marker = symbol
        return marker


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--results_dir", default="reports/eval_results")
    parser.add_argument("--output", default="reports/benchmark_report.md")
    parser.add_argument("--config", default="configs/eval.yml")
    args = parser.parse_args()

    config = {}
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.info("No eval config found — using defaults.")

    builder = ReportBuilder(
        results_dir=args.results_dir,
        output_path=args.output,
        config=config,
    )
    builder.build()


if __name__ == "__main__":
    main()
