"""Configuration loading and merging utilities."""

from pathlib import Path
from typing import Any

import yaml

# Repository root, resolved from this file's location (src/utils/config_utils.py),
# used as a third resolution candidate so configs load correctly regardless of
# the caller's current working directory (e.g. a Colab notebook or a test runner).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, resolving nested config references."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    # Resolve nested config references
    for key in ["model_config", "data_config"]:
        if key in config and isinstance(config[key], str):
            nested_path = Path(config[key])
            if not nested_path.is_absolute():
                # Try, in order: relative to the parent config's directory,
                # relative to the current working directory, and relative to
                # the repository root. This makes config resolution independent
                # of the caller's cwd (train configs reference paths like
                # "configs/model/gemma4_e4b.yaml" as if run from repo root).
                for candidate in (
                    path.parent / nested_path,
                    Path.cwd() / nested_path,
                    _REPO_ROOT / nested_path,
                ):
                    if candidate.exists():
                        nested_path = candidate
                        break
                else:
                    nested_path = path.parent / nested_path
            config[key] = load_config(nested_path)

    return config


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge override into base config. Returns new dict (no mutation of base)."""
    import copy

    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def flatten_config(config: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config to dot-notation keys for logging."""
    flat = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, full_key))
        else:
            flat[full_key] = value
    return flat
