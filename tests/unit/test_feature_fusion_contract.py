"""Tests for fused feature layout contracts."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from methanet.schema import FUNCTIONAL_FEATURE_COLUMNS


def _load_fuse_features_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "workflow" / "scripts" / "fuse_features.py"
    spec = spec_from_file_location("fuse_features", module_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workflow_read_functional_uses_all_schema_columns(tmp_path: Path) -> None:
    module = _load_fuse_features_module()
    values = {
        column: float(index)
        for index, column in enumerate(FUNCTIONAL_FEATURE_COLUMNS)
    }
    path = tmp_path / "sample.tsv"
    pd.DataFrame([{"sample_id": "S1", **values}]).to_csv(path, sep="\t", index=False)

    observed = module.read_functional(path)

    assert observed.shape == (13,)
    assert np.allclose(
        observed,
        np.array([values[column] for column in FUNCTIONAL_FEATURE_COLUMNS]),
    )


def test_workflow_read_functional_rejects_partial_legacy_panel(
    tmp_path: Path,
) -> None:
    module = _load_fuse_features_module()
    path = tmp_path / "legacy.tsv"
    pd.DataFrame(
        [
            {
                "sample_id": "S1",
                "mcrA": 1.0,
                "pmoA": 2.0,
                "dsrA": 3.0,
                "nifH": 4.0,
                "cbbL": 5.0,
                "mcrA_pmoA_ratio": 6.0,
            }
        ]
    ).to_csv(path, sep="\t", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        module.read_functional(path)
