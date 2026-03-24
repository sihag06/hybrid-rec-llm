"""Centralized config loader using YAML with ENV overrides.

Environment variables prefixed with ``LRE_`` can override nested keys using
double-underscores, e.g. ``LRE_APP__LOG_LEVEL=DEBUG``.
"""

from __future__ import annotations

import argparse
import os
import pathlib
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = pathlib.Path(os.environ.get("CONFIG_PATH", "configs/config.yaml"))
ENV_PREFIX = "LRE_"


def _parse_env_value(value: str) -> Any:
    """Best-effort casting for env values."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _set_nested(config: Dict[str, Any], path: list[str], value: Any) -> None:
    cursor = config
    for part in path[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[path[-1]] = value


def apply_env_overrides(config: Dict[str, Any], prefix: str = ENV_PREFIX) -> Dict[str, Any]:
    """Apply ENV overrides in-place and return config."""
    for key, raw_value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].lower().split("__")
        _set_nested(config, path, _parse_env_value(raw_value))
    return config


def load_config(config_path: pathlib.Path | str | None = None) -> Dict[str, Any]:
    """Load YAML config and apply ENV overrides."""
    path = pathlib.Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open() as f:
        config: Dict[str, Any] = yaml.safe_load(f) or {}
    return apply_env_overrides(config)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Config loader helper")
    parser.add_argument(
        "--print", dest="print_config", action="store_true", help="Print resolved config"
    )
    parser.add_argument("--path", dest="config_path", type=str, help="Optional config path")
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    if args.print_config:
        print(yaml.dump(cfg, sort_keys=False))


if __name__ == "__main__":
    _cli()
