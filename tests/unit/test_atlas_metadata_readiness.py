from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "scripts/reports/build_atlas_metadata_readiness.py"
    spec = importlib.util.spec_from_file_location("atlas_metadata_readiness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missingness_and_identifier_parsing_are_explicit():
    module = load_module()
    assert module.present("0")
    assert not module.present("NA")
    assert module.split_ids("A;B, C") == ["A", "B", "C"]
    assert not module.truthy("missing")
