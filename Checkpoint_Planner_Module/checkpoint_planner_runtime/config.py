from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def ensure_dir(path: str | Path) -> Path:
    target = as_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = as_path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {config_path}, got {type(data)!r}")
    return data


def load_json(path: str | Path) -> dict[str, Any]:
    config_path = as_path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON mapping in {config_path}, got {type(data)!r}")
    return data


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = as_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
