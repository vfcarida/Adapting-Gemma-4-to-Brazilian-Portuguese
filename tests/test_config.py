"""Tests for config loading utilities."""

import sys
from pathlib import Path

import pytest

# Import directly without going through __init__ to avoid torch dependency
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config_utils import flatten_config, load_config, merge_configs


class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("key: value\nnested:\n  a: 1\n  b: 2\n")
        result = load_config(config_file)
        assert result["key"] == "value"
        assert result["nested"]["a"] == 1

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestMergeConfigs:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3}

    def test_deep_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99}}
        result = merge_configs(base, override)
        assert result == {"a": {"x": 1, "y": 99}, "b": 3}

    def test_add_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 2}


class TestFlattenConfig:
    def test_flat(self):
        config = {"a": 1, "b": "hello"}
        result = flatten_config(config)
        assert result == {"a": 1, "b": "hello"}

    def test_nested(self):
        config = {"a": {"b": {"c": 42}}}
        result = flatten_config(config)
        assert result == {"a.b.c": 42}

    def test_mixed(self):
        config = {"x": 1, "y": {"z": 2}}
        result = flatten_config(config)
        assert result == {"x": 1, "y.z": 2}
