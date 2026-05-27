"""
src/utils/config_utils.py
─────────────────────────
YAML configuration loading with ``.env`` override support and CLI
argument parsing factory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_config(
    config_path: str | Path,
    env_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a YAML config file with optional ``.env`` and CLI overrides.

    The merge priority (highest wins) is:
        1. *overrides* dict
        2. Environment variables (from ``.env`` or system)
        3. YAML file values

    Parameters
    ----------
    config_path : str | Path
        Path to the ``.yml`` / ``.yaml`` configuration file.
    env_path : str | Path | None
        Path to ``.env`` file.  Defaults to ``<repo_root>/.env``.
    overrides : dict | None
        Runtime overrides (e.g. from CLI arguments).

    Returns
    -------
    dict[str, Any]
        The merged configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load .env (if available)
    if env_path is None:
        env_path = config_path.parent.parent / ".env"
    if Path(env_path).exists():
        load_dotenv(env_path, override=False)
        logger.info("Loaded .env from %s", env_path)

    # Parse YAML
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Resolve environment variable references (${VAR_NAME} syntax)
    cfg = _resolve_env_vars(cfg)

    # Apply overrides
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    logger.info("Configuration loaded from %s", config_path)
    return cfg


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ``${VAR_NAME}`` patterns in string values."""
    if isinstance(obj, str):
        # Replace ${VAR} with environment variable value
        import re

        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(r"\$\{(\w+)\}", _replace, obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_args(description: str = "Gemma 4 PT-BR Pipeline") -> argparse.Namespace:
    """Standard CLI argument parser used by all scripts.

    Common arguments
    ----------------
    --config : Path to YAML config file (required).
    --override : Key=value overrides (optional, repeatable).
    --dry_run : Parse config and exit without running.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Key=value overrides. Dot notation supported (e.g. training.lr=1e-5).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Parse config and print it without running.",
    )
    args = parser.parse_args()
    return args


def parse_overrides(override_list: list[str]) -> dict[str, Any]:
    """Convert a list of ``key=value`` strings into a nested dict.

    Supports dot-notation keys like ``training.lr=1e-5``, which
    produces ``{"training": {"lr": 1e-5}}``.
    """
    result: dict[str, Any] = {}
    for item in override_list:
        if "=" not in item:
            logger.warning("Ignoring malformed override: %s", item)
            continue
        key, value = item.split("=", 1)
        value = _auto_cast(value)
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result


def _auto_cast(value: str) -> Any:
    """Attempt to cast a string to int, float, or bool."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def validate_config(cfg: dict[str, Any], required_keys: list[str]) -> None:
    """Raise ``KeyError`` if any required top-level keys are missing."""
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
