"""Harness configuration: nano-mle.toml at the repository root (gitignored).

nano-mle.example.toml holds the defaults and documents each setting. A missing
file or key falls back to the defaults below.
"""

import os
import tomllib
from functools import cache
from pathlib import Path

DEFAULTS = {
    "execution": {"cpu_threads": 32, "grid_n_jobs": 1},
    "plans": {"restrict_primitives": False},
    "prompts": {"data_volume_study": True},
    # Parameters per memory, e.g. [memory.window] leaderboard = 8; empty means the
    # memory's own defaults (see memory.py).
    "memory": {"window": {}, "full": {}},
}


def config_path() -> Path:
    if os.environ.get("NANO_MLE_CONFIG"):
        return Path(os.environ["NANO_MLE_CONFIG"])
    return Path(__file__).resolve().parents[2] / "nano-mle.toml"


@cache
def load_config() -> dict:
    path = config_path()
    loaded = tomllib.loads(path.read_text()) if path.exists() else {}
    unknown = set(loaded) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown sections in {path}: {sorted(unknown)}")
    config = {}
    for section, values in DEFAULTS.items():
        given = loaded.get(section, {})
        extra = set(given) - set(values)
        if extra:
            raise ValueError(f"Unknown keys in [{section}] of {path}: {sorted(extra)}")
        config[section] = {**values, **given}
    return config


def cpu_threads() -> int:
    threads = int(load_config()["execution"]["cpu_threads"])
    return threads if threads > 0 else (os.cpu_count() or 1)


def grid_n_jobs() -> int:
    return max(1, int(load_config()["execution"]["grid_n_jobs"]))
