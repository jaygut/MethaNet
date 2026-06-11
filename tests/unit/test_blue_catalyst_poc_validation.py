"""Tests for Blue Catalyst POC denominator and QC validation."""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


def _load_validation_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "validate_blue_catalyst_poc.py"
    spec = spec_from_file_location("validate_blue_catalyst_poc", module_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_report_keeps_final_and_pre_final_denominators(
    tmp_path: Path,
) -> None:
    module = _load_validation_module()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    np.savez(
        artifacts / "genome_embeddings.npz",
        embeddings=np.ones((3, 4), dtype=np.float32),
        sample=np.array(["R1", "R2", "W1"], dtype=object),
    )
    pd.DataFrame(
        [
            {
                "sample": "R1",
                "source": "rumen",
                "ecosystem": "rumen",
                "domain": "Archaea",
            },
            {
                "sample": "R2",
                "source": "rumen",
                "ecosystem": "rumen",
                "domain": "Bacteria",
            },
            {
                "sample": "W1",
                "source": "mucc",
                "ecosystem": "wetland",
                "domain": "Unknown",
            },
        ]
    ).to_csv(artifacts / "embedding_metadata.tsv", sep="\t", index=False)
    (artifacts / "embedding_stats.json").write_text(
        json.dumps(
            {
                "candidate_total": 3,
                "embedded": 3,
                "no_valid": 0,
                "empty_emb": 0,
                "non_finite": 0,
                "excluded_coassembly": 1,
                "pending_remaining": 0,
            }
        )
    )
    pd.DataFrame(
        [
            {"source": "mucc", "ecosystem": "wetland", "n_samples": 2},
            {"source": "rumen", "ecosystem": "rumen", "n_samples": 2},
        ]
    ).to_csv(artifacts / "sample_source_counts.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "sample": "R1",
                "source": "rumen",
                "ecosystem": "rumen",
                "domain": "Archaea",
                "bridging_score": 0.9,
                "mixing_coeff": 0.4,
            }
        ]
    ).to_csv(artifacts / "bridging_genomes_top.tsv", sep="\t", index=False)

    analytics = tmp_path / "analytics_summary.json"
    analytics.write_text(
        json.dumps({"confounding_assessment": {"source_confounded": True}})
    )

    report = module.build_report(
        artifacts_dir=artifacts,
        analytics_summary=analytics,
        top_n=1,
        expected_total=3,
        expected_rumen=2,
        expected_wetland=1,
        expected_dim=4,
    )

    assert report["status"] == "ok"
    assert report["final_embedding_matrix_shape"] == [3, 4]
    assert report["finite_vector_count"] == 3
    assert report["final_counts"]["ecosystem"] == {"rumen": 2, "wetland": 1}
    assert report["pre_final_source_counts"][0]["n_samples"] == 2
    assert report["embedding_attrition"]["excluded_coassembly"] == 1
    assert report["source_confounded"] is True
    assert report["top_bridge_candidates"][0]["sample"] == "R1"
