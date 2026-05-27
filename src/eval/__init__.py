# src.eval — Evaluation, benchmarking, and reporting
from src.eval.benchmark_runner import BenchmarkRunner
from src.eval.prompt_templates import Gemma4PromptFormatter
from src.eval.metrics import compute_macro_f1, compute_pearson, compute_approval_rate, compute_accuracy
from src.eval.bootstrap_ci import bootstrap_confidence_interval, paired_bootstrap_test
from src.eval.report_builder import ReportBuilder

__all__ = [
    "BenchmarkRunner",
    "Gemma4PromptFormatter",
    "compute_macro_f1",
    "compute_pearson",
    "compute_approval_rate",
    "compute_accuracy",
    "bootstrap_confidence_interval",
    "paired_bootstrap_test",
    "ReportBuilder",
]
