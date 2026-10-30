"""Configuration loading and global seeding for the self-healing cloud platform.

Central entry point used by every module (see SPEC.md). The config file is
``config.yaml`` at the project root and mirrors the parameters described in the
paper (IJCET 16(5) 2025, Article ID IJCET_16_05_012).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load the YAML configuration and inject the resolved project root.

    ``cfg["project"]["root"]`` is set to the directory containing the config
    file so that every relative path in the config can be resolved with
    :func:`resolve_path` regardless of the current working directory.
    """
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        candidate = Path.cwd() / cfg_path
        cfg_path = candidate if candidate.exists() else PROJECT_ROOT / path
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("project", {})
    cfg["project"]["root"] = str(cfg_path.resolve().parent)
    return cfg


def resolve_path(cfg: dict, rel: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    rel = Path(rel)
    if rel.is_absolute():
        return rel
    root = Path(cfg.get("project", {}).get("root", PROJECT_ROOT))
    return root / rel


def set_global_seeds(cfg: dict) -> None:
    """Seed python, numpy and tensorflow RNGs from ``cfg["project"]["seed"]``."""
    seed = int(cfg.get("project", {}).get("seed", 42))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:  # tensorflow is heavyweight; only seed it if already importable
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            import keras

            keras.utils.set_random_seed(seed)
        except ImportError:  # pragma: no cover
            pass
    except ImportError:  # pragma: no cover
        pass
