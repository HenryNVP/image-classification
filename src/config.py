from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_configs(paths: Iterable[Path]) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for path in paths:
        if path is None or not path.exists():
            continue
        data = load_yaml(path)
        merge_dict(config, data)
    return config


def to_namespace(config: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            key: to_namespace(value) if isinstance(value, dict) else value
            for key, value in config.items()
        }
    )
