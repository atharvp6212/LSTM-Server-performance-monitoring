"""Tests for selfheal.config (SPEC: "load_config resolves root; seeds don't crash").

Covers the configuration/seeding entry points every module of the paper
implementation depends on (IJCET 16(5) 2025 — environment/reproducibility
requirements in SPEC.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from selfheal.config import load_config, resolve_path, set_global_seeds


def test_load_config_injects_project_root(project_root: Path) -> None:
    """load_config on the real config.yaml sets cfg["project"]["root"]."""
    cfg = load_config(project_root / "config.yaml")
    assert isinstance(cfg, dict)
    assert cfg["project"]["root"] == str(project_root)
    # The paper's parameter groups must all be present in the shipped config.
    for section in ("project", "paths", "dataset", "preprocessing", "windowing",
                    "features", "model", "decision", "healing", "monitoring"):
        assert section in cfg, f"config.yaml is missing section {section!r}"


def test_load_config_root_follows_config_location(tmp_path: Path) -> None:
    """The injected root is the directory containing the loaded config file."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("project:\n  seed: 7\npaths:\n  reports: reports\n",
                        encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg["project"]["root"] == str(tmp_path)
    assert cfg["project"]["seed"] == 7


def test_resolve_path_relative_and_absolute(cfg: dict, tmp_path: Path) -> None:
    """Relative paths resolve against the root; absolute paths pass through."""
    resolved = resolve_path(cfg, "reports/metrics.json")
    assert resolved == Path(cfg["project"]["root"]) / "reports" / "metrics.json"

    absolute = tmp_path / "elsewhere" / "file.json"
    assert resolve_path(cfg, absolute) == absolute


def test_set_global_seeds_is_deterministic(cfg: dict) -> None:
    """Seeding does not crash and makes numpy's global RNG reproducible."""
    set_global_seeds(cfg)
    first = np.random.rand(4)
    set_global_seeds(cfg)
    second = np.random.rand(4)
    np.testing.assert_allclose(first, second)
