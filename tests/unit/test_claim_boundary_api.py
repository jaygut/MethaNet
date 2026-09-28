"""The shipped package must not expose methane-risk tiers (claim contract)."""

import tomllib
from pathlib import Path

import methanet

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_top_level_package_does_not_export_risk_tiers() -> None:
    for name in ("RiskTier", "MethaNetEnsemble", "ClassificationResult"):
        assert name not in methanet.__all__
        assert not hasattr(methanet, name)


def test_wheel_does_not_ship_legacy_tier_inference() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    packages = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/api_bridge" not in packages
