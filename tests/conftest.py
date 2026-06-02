"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Ensure project root is always importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "slow: marks slow tests")
    config.addinivalue_line("markers", "smoke: marks smoke tests")


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for test artifacts."""
    output = tmp_path / "outputs"
    output.mkdir()
    return output


@pytest.fixture
def sample_eval_results():
    """Synthetic evaluation results for testing report builder."""
    return [{
        "model_name": "test_model",
        "model_id": "test/model",
        "benchmarks": {
            "think_off": {
                "enem": {
                    "task": "enem",
                    "group": "brasil_geral",
                    "metric_name": "accuracy",
                    "metrics": {"accuracy": 0.75},
                    "num_examples": 100,
                    "inference_time_sec": 10.0,
                    "think_mode": "off",
                    "raw_predictions": ["C", "B"],
                },
            }
        },
    }]
