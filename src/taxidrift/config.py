"""Load the shared YAML config.

Resolution order: explicit path argument, then the TAXIDRIFT_CONFIG environment variable,
then configs/default.yaml. A config may say `extends: other.yaml` to start from another file.
Relative data/results paths in the config are relative to the project root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(_load(path.parent / parent), cfg)
    return cfg


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    path = Path(path or os.environ.get("TAXIDRIFT_CONFIG") or DEFAULT_CONFIG)
    return _load(path)


def resolve(relative: str | os.PathLike) -> Path:
    """Turn a project-relative path from the config into an absolute path."""
    p = Path(relative).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p
