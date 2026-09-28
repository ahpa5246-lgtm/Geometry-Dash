from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    required = ("window", "input", "environment", "learning", "paths")
    missing = [name for name in required if name not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    env = cfg["environment"]
    if env["step_hz"] <= 0:
        raise ValueError("environment.step_hz must be > 0")
    if env["freeze_frames"] < 1:
        raise ValueError("environment.freeze_frames must be >= 1")

    win = cfg["window"]
    if win["observation_width"] < 36 or win["observation_height"] < 36:
        raise ValueError("Observation size is too small for the CNN.")

    return cfg
