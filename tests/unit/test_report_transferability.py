from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PARQUET_AVAILABLE = importlib.util.find_spec("pyarrow") is not None

pytestmark = pytest.mark.skipif(
    not PARQUET_AVAILABLE,
    reason="pyarrow not available",
)


def test_report_transferability_generates_poc_report(tmp_path: Path) -> None:
    functional_dim = 2
    esm2_dim = 2
    genome_dim = 2
    total_dim = functional_dim + esm2_dim + genome_dim

    rng = np.random.default_rng(42)

    rows = []
    for i in range(3):
        flux = float(i + 1)
        features = np.array(
            [
                flux,
                0.5 * flux,
                0.1 * flux,
                0.2 * flux,
                0.3 * flux,
                0.4 * flux,
            ],
            dtype=float,
        )
        rows.append(
            {
                "sample_id": f"S{i}",
                "domain": "rumen",
                "flux_value": flux,
                **{f"f_{j}": float(features[j]) for j in range(total_dim)},
            }
        )

    for i in range(3):
        flux = float(i + 1)
        features = np.array(
            [
                flux,
                0.5 * flux,
                0.1 * flux + 1.0,
                0.2 * flux + 1.0,
                0.3 * flux + 1.0,
                0.4 * flux + 1.0,
            ],
            dtype=float,
        )
        features += rng.normal(0, 0.01, size=features.shape)
        rows.append(
            {
                "sample_id": f"T{i}",
                "domain": "coastal",
                "flux_value": flux,
                **{f"f_{j}": float(features[j]) for j in range(total_dim)},
            }
        )

    df = pd.DataFrame(rows)

    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path, index=False)

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "T0_dnabert2_metrics.json").write_text(
        json.dumps(
            {
                "n_contigs": 1,
                "total_bp": 1000,
                "kmer_size": 6,
                "max_length": 512,
                "n_truncated": 1,
                "frac_truncated": 1.0,
                "mean_truncated_tokens": 10.0,
                "pooling": "mean",
            }
        )
    )

    out_dir = tmp_path / "out"
    report_path = out_dir / "poc_report.json"

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "workflow" / "scripts" / "report_transferability.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--features",
            str(features_path),
            "--output-dir",
            str(out_dir),
            "--source-domain",
            "rumen",
            "--target-domain",
            "coastal",
            "--target-col",
            "flux_value",
            "--functional-dim",
            str(functional_dim),
            "--esm2-dim",
            str(esm2_dim),
            "--genome-dim",
            str(genome_dim),
            "--min-labeled-per-domain",
            "3",
            "--bootstrap-samples",
            "10",
            "--dnabert2-metrics-dir",
            str(metrics_dir),
            "--report-out",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise AssertionError(
            "report_transferability.py failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    assert report_path.exists()

    payload = json.loads(report_path.read_text())
    assert payload["source_domain"] == "rumen"
    assert payload["target_domain"] == "coastal"
    assert "missingness_by_group" in payload
    assert "domain_shift_by_group" in payload
    assert "cross_domain_transfer" in payload
    assert "gating" in payload
    assert "dnabert2_truncation" in payload

    assert payload["dnabert2_truncation"]["n_with_metrics"] == 1
    assert (out_dir / "domain_shift_by_group.csv").exists()
    assert (out_dir / "feature_finiteness_by_group.csv").exists()
    assert (out_dir / "cross_domain_transfer_by_group.csv").exists()
    assert (out_dir / "transferability_summary.json").exists()


def test_real_fused_layout_group_slices_do_not_overlap() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "workflow" / "scripts" / "report_transferability.py"
    spec = spec_from_file_location("report_transferability", module_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    groups = module._build_groups(
        total_dim=13 + 1280 + 768,
        functional_dim=13,
        esm2_dim=1280,
        genome_dim=768,
    )
    group_map = {group.name: group for group in groups}

    assert group_map["functional"].slices == (slice(0, 13),)
    assert group_map["esm2"].slices == (slice(13, 1293),)
    assert group_map["genome"].slices == (slice(1293, 2061),)
