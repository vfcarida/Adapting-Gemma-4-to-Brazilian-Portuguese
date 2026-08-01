"""Tests for checkpoint management utilities."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.checkpointing import (
    find_latest_checkpoint,
    load_training_state,
    save_training_state,
)


class TestFindLatestCheckpoint:
    """Test checkpoint discovery."""

    def test_finds_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint directories
            (Path(tmpdir) / "checkpoint-100").mkdir()
            (Path(tmpdir) / "checkpoint-200").mkdir()
            (Path(tmpdir) / "checkpoint-50").mkdir()

            result = find_latest_checkpoint(tmpdir)
            assert result is not None
            assert "checkpoint-200" in str(result)

    def test_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_latest_checkpoint(tmpdir)
            assert result is None

    def test_ignores_non_checkpoint_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "final").mkdir()
            (Path(tmpdir) / "logs").mkdir()
            (Path(tmpdir) / "checkpoint-100").mkdir()

            result = find_latest_checkpoint(tmpdir)
            assert result is not None
            assert "checkpoint-100" in str(result)

    def test_handles_nonexistent_dir(self):
        result = find_latest_checkpoint("/nonexistent/path/xyz")
        assert result is None


class TestTrainingState:
    """Test training state save/load."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "global_step": 1000,
                "training_loss": 2.5,
                "model_id": "test/model",
                "use_lora": True,
            }
            save_training_state(tmpdir, state)
            loaded = load_training_state(tmpdir)

            assert loaded is not None
            assert loaded["global_step"] == 1000
            assert loaded["training_loss"] == 2.5
            assert loaded["model_id"] == "test/model"
            assert loaded["use_lora"] is True

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_training_state(tmpdir)
            assert result is None

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "deep" / "nested"
            save_training_state(nested, {"step": 1})
            assert (nested / "training_state.json").exists()

    def test_state_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {"complex": {"nested": [1, 2, 3]}, "float": 1.5e-4}
            save_training_state(tmpdir, state)

            with open(Path(tmpdir) / "training_state.json") as f:
                loaded = json.load(f)
            assert loaded["complex"]["nested"] == [1, 2, 3]
