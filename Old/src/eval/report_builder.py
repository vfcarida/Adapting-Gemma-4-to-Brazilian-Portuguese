"""Generate evaluation reports, tables, and figures for academic paper.

This module takes raw evaluation results (from BenchmarkRunner) and produces
publication-ready artifacts:

1. CSV tables: Full results, pivot tables, group averages, best-per-benchmark
2. Markdown: Summary with key findings, formatted tables
3. Figures: Bar charts, heatmaps, radar charts (requires matplotlib/seaborn)
4. Paper template: findings_for_paper.md with hypothesis status tracking

The report structure is designed for the ablation study format:
- Models on rows, benchmarks on columns
- Grouped by benchmark category (Brasil Geral, Semantica, etc.)
- Separate columns for think_on vs think_off modes
- Macro averages per group for quick comparison

Usage:
    from src.eval.report_builder import ReportBuilder

    builder = ReportBuilder(results, output_dir="reports")
    builder.build_all()  # Generates all artifacts

    # Or generate individually:
    df = builder.build_main_table()
    builder.build_plots()
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Benchmark groups for structured reporting
# Maps group names to their constituent benchmarks
BENCHMARK_GROUPS = {
    "brasil_geral": ["enem_2022", "enem_2023", "enem_2024", "bluex"],
    "semantica": ["assin2_rte", "assin2_sts", "copa_pt", "mrpc_pt", "rte_pt"],
    "classificacao_social": ["hatebr", "tweet_sentbr"],
    "juridico": ["oab_bench"],
    "cultura": ["broverbs", "tuguesice_pt"],
    "seguranca": ["donotanswer_pt"],
}


class ReportBuilder:
    """Build evaluation reports with tables and figures.

    Takes the output of BenchmarkRunner.run_all() for multiple models
    and generates structured reports suitable for academic publication.

    The builder produces several complementary views:
    - Full results: Every model × benchmark × think_mode combination
    - Group averages: Macro average per benchmark category
    - Best per benchmark: Which model wins each benchmark
    - Visual comparisons: Heatmap, radar chart, bar chart

    Args:
        results: List of model result dicts from BenchmarkRunner.
        output_dir: Directory to write all report artifacts.

    Attributes:
        results: Raw evaluation results.
        output_dir: Output path (created if needed).
    """

    def __init__(self, results: list[dict], output_dir: str | Path = "reports"):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_all(self) -> None:
        """Generate all report artifacts in sequence.

        Calls each builder method. Safe to run repeatedly (overwrites
        previous outputs). Order matters: tables must exist before plots.
        """
        self.build_main_table()
        self.build_group_averages()
        self.build_comparison_csv()
        self.build_summary_md()
        self.build_plots()
        logger.info(f"Reports generated in {self.output_dir}")

    def build_main_table(self) -> pd.DataFrame:
        """Build main results table (models × benchmarks).

        Flattens the nested result structure into a flat DataFrame with
        columns: model, think_mode, benchmark, group, metric, score.

        Also generates:
        - results_full.csv: The flat table
        - results_pivot.csv: Pivoted (models as rows, benchmarks as columns)
        - results_table.md: Markdown-formatted pivot table

        Returns:
            DataFrame with all individual scores.
        """
        rows = []
        for model_result in self.results:
            model_name = model_result["model_name"]
            for mode_key, benchmarks in model_result.get("benchmarks", {}).items():
                for bench_name, bench_result in benchmarks.items():
                    metrics = bench_result.get("metrics", {})
                    # Use the benchmark's declared primary metric
                    primary_metric = bench_result.get("metric_name", "accuracy")
                    score = metrics.get(primary_metric, metrics.get("accuracy", 0))
                    rows.append(
                        {
                            "model": model_name,
                            "think_mode": mode_key,
                            "benchmark": bench_name,
                            "group": bench_result.get("group", ""),
                            "metric": primary_metric,
                            "score": score,
                        }
                    )

        df = pd.DataFrame(rows)

        # Save flat CSV (one row per model × benchmark × mode)
        df.to_csv(self.output_dir / "results_full.csv", index=False)

        # Create pivot table for paper-ready format
        if not df.empty:
            pivot = df.pivot_table(
                index=["model", "think_mode"],
                columns="benchmark",
                values="score",
                aggfunc="first",
            )
            pivot.to_csv(self.output_dir / "results_pivot.csv")

            # Markdown version for quick viewing / paper drafts
            try:
                md = pivot.to_markdown()
            except ImportError:
                # tabulate not installed — fall back to CSV-style string
                md = pivot.to_string()
            with open(self.output_dir / "results_table.md", "w") as f:
                f.write("# Evaluation Results\n\n")
                f.write(md or "No results")

        return df

    def build_group_averages(self) -> pd.DataFrame:
        """Compute macro averages by benchmark group.

        Groups benchmarks by category (e.g., "brasil_geral", "semantica")
        and computes the unweighted mean score across benchmarks in each
        group. This gives a single number per model per category.

        Saves: group_averages.csv

        Returns:
            DataFrame with columns: model, think_mode, group, macro_avg, n_benchmarks.
        """
        rows = []
        for model_result in self.results:
            model_name = model_result["model_name"]
            for mode_key, benchmarks in model_result.get("benchmarks", {}).items():
                # Collect scores by group
                group_scores: dict[str, list[float]] = {}
                for bench_name, bench_result in benchmarks.items():
                    group = bench_result.get("group", "other")
                    metrics = bench_result.get("metrics", {})
                    primary_metric = bench_result.get("metric_name", "accuracy")
                    score = metrics.get(primary_metric, 0)
                    group_scores.setdefault(group, []).append(score)

                # Compute macro average per group
                for group, scores in group_scores.items():
                    rows.append(
                        {
                            "model": model_name,
                            "think_mode": mode_key,
                            "group": group,
                            "macro_avg": float(np.mean(scores)),
                            "n_benchmarks": len(scores),
                        }
                    )

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "group_averages.csv", index=False)
        return df

    def build_comparison_csv(self) -> None:
        """Build pairwise model comparison (best model per benchmark).

        Identifies which model achieved the highest score on each benchmark.
        Useful for quickly seeing which adaptation strategy wins where.

        Saves: best_per_benchmark.csv
        """
        df = (
            pd.read_csv(self.output_dir / "results_full.csv")
            if (self.output_dir / "results_full.csv").exists()
            else pd.DataFrame()
        )
        if df.empty:
            return

        # Find the row with max score for each benchmark
        best = df.loc[df.groupby("benchmark")["score"].idxmax()]
        best[["benchmark", "model", "score"]].to_csv(
            self.output_dir / "best_per_benchmark.csv", index=False
        )

    def build_summary_md(self) -> None:
        """Build summary.md with key findings and rankings.

        Generates a human-readable summary with:
        - Overall model ranking by average score
        - Best model per individual benchmark
        - Placeholder for regression analysis

        Saves: summary.md
        """
        results_path = self.output_dir / "results_full.csv"
        if not results_path.exists():
            return

        df = pd.read_csv(results_path)
        if df.empty:
            return

        # Rank models by overall average score
        model_avg = df.groupby("model")["score"].mean().sort_values(ascending=False)

        summary = "# Evaluation Summary\n\n"
        summary += "## Best Model by Average Score\n\n"
        for model, score in model_avg.items():
            summary += f"- **{model}**: {score:.4f}\n"

        summary += "\n## Best Model per Benchmark\n\n"
        best = df.loc[df.groupby("benchmark")["score"].idxmax()]
        for _, row in best.iterrows():
            summary += f"- {row['benchmark']}: **{row['model']}** ({row['score']:.4f})\n"

        # Placeholders for post-experiment analysis
        summary += "\n## Observations\n\n"
        summary += "- TODO: Add regression analysis after all models are evaluated\n"
        summary += "- TODO: Add cost/quality tradeoff analysis\n"

        with open(self.output_dir / "summary.md", "w") as f:
            f.write(summary)

    def build_plots(self) -> None:
        """Generate evaluation visualizations.

        Creates three types of plots:
        1. Bar chart: Group averages by model (quick category comparison)
        2. Radar chart: Multi-dimensional model comparison across groups
        3. Heatmap: Full model × benchmark score matrix

        Gracefully skips if matplotlib/seaborn are not installed.

        Saves to: reports/figures/
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping plots")
            return

        results_path = self.output_dir / "results_full.csv"
        if not results_path.exists():
            return
        df = pd.read_csv(results_path)
        if df.empty:
            return

        plots_dir = self.output_dir / "figures"
        plots_dir.mkdir(exist_ok=True)

        # Plot 1: Bar chart of group averages by model
        fig, ax = plt.subplots(figsize=(12, 6))
        group_df = (
            pd.read_csv(self.output_dir / "group_averages.csv")
            if (self.output_dir / "group_averages.csv").exists()
            else pd.DataFrame()
        )
        if not group_df.empty:
            pivot = group_df.pivot_table(index="group", columns="model", values="macro_avg")
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("Macro Average by Benchmark Group")
            ax.set_ylabel("Score")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            fig.savefig(plots_dir / "group_comparison.png", dpi=150, bbox_inches="tight")
            plt.close()

        # Plot 2: Radar chart (multi-axis model comparison)
        self._radar_chart(df, plots_dir)

        # Plot 3: Heatmap (full score matrix with annotations)
        fig, ax = plt.subplots(figsize=(14, 6))
        pivot = df.pivot_table(index="model", columns="benchmark", values="score", aggfunc="first")
        if not pivot.empty:
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
            ax.set_title("Score Heatmap (Models x Benchmarks)")
            plt.tight_layout()
            fig.savefig(plots_dir / "heatmap.png", dpi=150, bbox_inches="tight")
            plt.close()

    def _radar_chart(self, df: pd.DataFrame, plots_dir: Path) -> None:
        """Generate radar chart comparing models across benchmark groups.

        Each axis represents a benchmark group, and each model is a polygon.
        Models that excel in all areas have larger polygons. This visualization
        quickly shows strengths and weaknesses per model.

        Args:
            df: Full results DataFrame (used for reference only).
            plots_dir: Directory to save the figure.
        """
        import matplotlib.pyplot as plt

        group_df = (
            pd.read_csv(self.output_dir / "group_averages.csv")
            if (self.output_dir / "group_averages.csv").exists()
            else pd.DataFrame()
        )
        if group_df.empty:
            return

        models = group_df["model"].unique()
        groups = group_df["group"].unique()
        n_groups = len(groups)

        # Need at least 3 groups for a meaningful radar chart
        if n_groups < 3:
            return

        # Compute angles for each axis (evenly spaced around the circle)
        angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for model in models:
            model_data = group_df[group_df["model"] == model]
            values = []
            for g in groups:
                row = model_data[model_data["group"] == g]
                values.append(float(row["macro_avg"].iloc[0]) if len(row) > 0 else 0)
            values += values[:1]  # Close the polygon
            ax.plot(angles, values, label=model, linewidth=2)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(groups, size=9)
        ax.set_title("Model Comparison by Benchmark Group")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        fig.savefig(plots_dir / "radar_chart.png", dpi=150, bbox_inches="tight")
        plt.close()


def build_findings_for_paper(results_dir: str | Path = "reports") -> None:
    """Generate findings_for_paper.md template with hypothesis tracking.

    Creates a structured template for recording experimental findings,
    organized around the project's research hypotheses. To be filled
    in after experiments complete.

    Args:
        results_dir: Directory to write the findings file.
    """
    results_dir = Path(results_dir)

    findings = """# Findings for Paper

## Main Results

TODO: Fill after experiments complete.

## Hypotheses

### H1: CPT on Aurora-PT improves Portuguese benchmarks
- Status: PENDING
- Evidence:

### H2: English replay prevents catastrophic forgetting
- Status: PENDING
- Evidence:

### H3: Residual merge recovers instruction-following without SFT
- Status: PENDING
- Evidence:

### H4: CPT + SFT outperforms CPT + Residual Merge
- Status: PENDING
- Evidence:

### H5: Think mode improves complex reasoning tasks
- Status: PENDING
- Evidence:

## Limitations

- Single-seed evaluation (mitigated by bootstrap CIs)
- Aurora-PT corpus may have domain biases
- Benchmark selection may not cover all Portuguese phenomena
- Computational budget constraints model scale

## Threats to Validity

- Internal: contamination between training data and benchmarks
- External: benchmark saturation for larger models
- Construct: multiple-choice format may not reflect generation quality
"""
    with open(results_dir / "findings_for_paper.md", "w") as f:
        f.write(findings)
