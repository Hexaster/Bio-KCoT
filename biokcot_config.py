"""Shared configuration loader for BioKCoT scripts."""

import json
import os
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _load_env_file(path):
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def load_config():
    configured_path = os.getenv("BIOKCOT_CONFIG", "config.json")
    config_path = Path(configured_path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def get(key, default=None, env=None):
    """Read a dotted config key, with an optional environment override."""
    if env and os.getenv(env):
        return os.environ[env]
    value = load_config()
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def env(key, default=None, required=False):
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(f"Set {key} in .env or in the process environment.")
    return value


def path(key, default=None, env=None):
    """Read a path and resolve repository-relative values from the project root."""
    value = get(key, default=default, env=env)
    if value is None:
        return None
    resolved = Path(value).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved

