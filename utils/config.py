from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from utils.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


@lru_cache(maxsize=None)
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """the settings of config/config.yaml, read once per path."""
    if not path.exists():
        raise ConfigError(f"config file {path} not found")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} must hold a mapping of settings")
    return cfg


def setting(cfg: dict[str, Any], *keys: str) -> Any:
    """nested value cfg[k1][k2]...; a missing key raises ConfigError naming the full path."""
    value: Any = cfg
    for depth, key in enumerate(keys):
        if not isinstance(value, dict) or key not in value:
            raise ConfigError(f"setting '{'.'.join(keys[:depth + 1])}' missing from config/config.yaml")
        value = value[key]
    return value
