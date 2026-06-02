"""Tests for report builder module."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("pandas")
pytest.importorskip("numpy")

from src.eval.report_builder import BENCHMARK_GROUPS, ReportBuilder, build_findings_for_paper


def make_sample_results():
    """Create sample evaluation results for testing."""
    return [
        {
            "model_id": "google/gemma-4-E4B-it",
            "model_name": "gemma-4-E4B-it",
            "benchmarks": {
                "think_off": {
                    "enem_2022": {
                        "task": "enem",
                        "group": "brasil_geral",
                        "metric_name": "accuracy",
                        "metrics": {"accuracy": 0.65, "n_correct": 26, "n_total": 40},
                        "num_examples": 40,
                        "think_mode": "off",
                    },
                    "assin2_rte": {
                        "task": "assin2_rte",
                        "group": "semantica",
                        "metric_name": "accuracy",
                        "metrics": {"accuracy": 0.72, "n_correct": 72, "n_total": 100},
                        "num_examples": 100,
                        "think_mode": "off",
                    },
                    "hatebr": {
                        "task": "hatebr",
                        "group": "classificacao_social",
                        "metric_name": "macro_f1",
                        "metrics": {"macro_f1": 0.58, "accuracy": 0.61},
                        "num_examples": 200,
                        "think_mode": "off",
                    },
                },
            },
        },
        {
            "model_id": "adapted-model",
            "model_name": "gemma-4-E4B-cpt",
            "benchmarks": {
                "think_off": {
                    "enem_2022": {
                        "task": "enem",
                        "group": "brasil_geral",
                        "metric_name": "accuracy",
                        "metrics": {"accuracy": 0.72, "n_correct": 29, "n_total": 40},
                        "num_examples": 40,
                        "think_mode": "off",
                    },
                    "assin2_rte": {
                        "task": "assin2_rte",
                        "group": "semantica",
                        "metric_name": "accuracy",
                        "metrics": {"accuracy": 0.75, "n_correct": 75, "n_total": 100},
                        "num_examples": 100,
                        "think_mode": "off",
                    },
                    "hatebr": {
                        "task": "hatebr",
                        "group": "classificacao_social",
                        "metric_name": "macro_f1",
                        "metrics": {"macro_f1": 0.63, "accuracy": 0.65},
                        "num_examples": 200,
                        "think_mode": "off",
                    },
                },
            },
        },
    ]


class TestBenchmarkGroups:
    """Test the benchmark group structure."""

    def test_all_groups_defined(self):
        expected_groups = ["brasil_geral", "semantica", "classificacao_social",
                          "juridico", "cultura", "seguranca"]
        for group in expected_groups:
            assert group in BENCHMARK_GROUPS

    def test_brasil_geral_contains_enem(self):
        assert "enem_2022" in BENCHMARK_GROUPS["brasil_geral"]
        assert "enem_2023" in BENCHMARK_GROUPS["brasil_geral"]
        assert "enem_2024" in BENCHMARK_GROUPS["brasil_geral"]

    def test_no_duplicate_benchmarks(self):
        all_benchmarks = []
        for benchmarks in BENCHMARK_GROUPS.values():
            all_benchmarks.extend(benchmarks)
        assert len(all_benchmarks) == len(set(all_benchmarks))


class TestReportBuilder:
    """Test report generation."""

    def test_build_main_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ReportBuilder(make_sample_results(), output_dir=tmpdir)
            df = builder.build_main_table()

            # Should have 6 rows (2 models * 3 benchmarks)
            assert len(df) == 6
            assert "model" in df.columns
            assert "benchmark" in df.columns
            assert "score" in df.columns

            # CSV should be saved
            assert (Path(tmpdir) / "results_full.csv").exists()
            assert (Path(tmpdir) / "results_pivot.csv").exists()
            assert (Path(tmpdir) / "results_table.md").exists()

    def test_build_group_averages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ReportBuilder(make_sample_results(), output_dir=tmpdir)
            builder.build_main_table()  # Needed for downstream
            df = builder.build_group_averages()

            # 2 models * 3 groups = 6 rows
            assert len(df) == 6
            assert "macro_avg" in df.columns
            assert "n_benchmarks" in df.columns

            # Check group averages are correct
            gemma_brasil = df[(df["model"] == "gemma-4-E4B-it") & (df["group"] == "brasil_geral")]
            assert gemma_brasil["macro_avg"].iloc[0] == pytest.approx(0.65)

    def test_build_comparison_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ReportBuilder(make_sample_results(), output_dir=tmpdir)
            builder.build_main_table()
            builder.build_comparison_csv()

            best_path = Path(tmpdir) / "best_per_benchmark.csv"
            assert best_path.exists()

            import pandas as pd
            best = pd.read_csv(best_path)
            # CPT model should win all benchmarks in this sample
            assert all(best["model"] == "gemma-4-E4B-cpt")

    def test_build_summary_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ReportBuilder(make_sample_results(), output_dir=tmpdir)
            builder.build_main_table()
            builder.build_summary_md()

            summary_path = Path(tmpdir) / "summary.md"
            assert summary_path.exists()
            content = summary_path.read_text()
            assert "gemma-4-E4B-cpt" in content
            assert "Best Model" in content

    def test_build_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ReportBuilder(make_sample_results(), output_dir=tmpdir)
            builder.build_all()

            # All artifacts should exist
            assert (Path(tmpdir) / "results_full.csv").exists()
            assert (Path(tmpdir) / "group_averages.csv").exists()
            assert (Path(tmpdir) / "best_per_benchmark.csv").exists()
            assert (Path(tmpdir) / "summary.md").exists()

    def test_empty_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ReportBuilder([], output_dir=tmpdir)
            df = builder.build_main_table()
            assert len(df) == 0

    def test_uses_primary_metric(self):
        """Score should use the declared metric_name, not always accuracy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ReportBuilder(make_sample_results(), output_dir=tmpdir)
            df = builder.build_main_table()

            # HateBR uses macro_f1 as primary metric
            hatebr_rows = df[df["benchmark"] == "hatebr"]
            # The score should be macro_f1 value (0.58, 0.63), not accuracy
            assert hatebr_rows["score"].iloc[0] == pytest.approx(0.58)


class TestBuildFindingsForPaper:
    """Test findings template generation."""

    def test_creates_findings_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_findings_for_paper(tmpdir)
            findings_path = Path(tmpdir) / "findings_for_paper.md"
            assert findings_path.exists()

    def test_contains_hypotheses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_findings_for_paper(tmpdir)
            content = (Path(tmpdir) / "findings_for_paper.md").read_text()
            assert "H1:" in content
            assert "H2:" in content
            assert "H3:" in content
            assert "H4:" in content
            assert "H5:" in content

    def test_contains_limitations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_findings_for_paper(tmpdir)
            content = (Path(tmpdir) / "findings_for_paper.md").read_text()
            assert "Limitations" in content
            assert "Threats to Validity" in content
