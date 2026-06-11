"""Tests for Blue Catalyst FG batch pipeline helper script."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


def _load_pipeline_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "blue_catalyst_fg_batch_pipeline.py"
    spec = spec_from_file_location("blue_catalyst_fg_batch_pipeline", script_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_builds_batch_artifacts(tmp_path):
    module = _load_pipeline_module()

    emb_meta = tmp_path / "embedding_metadata.tsv"
    emb_npz = tmp_path / "genome_embeddings.npz"
    out_dir = tmp_path / "fg_artifacts"

    df = pd.DataFrame(
        {
            "sample": ["mag_001", "mag_002", "mag_003"],
            "source": ["rumen", "rumen", "mucc"],
            "ecosystem": ["rumen", "rumen", "wetland"],
            "domain": ["Bacteria", "Bacteria", "Archaea"],
            "proteome_faa": ["/tmp/a.faa", "/tmp/b.faa", "/tmp/c.faa"],
        }
    )
    df.to_csv(emb_meta, sep="\t", index=False)
    np.savez(emb_npz, embeddings=np.array([[1.0, 2.0], [0.1, 0.2], [3.0, 4.0]]))

    args = module.build_parser().parse_args(
        [
            "plan",
            "--embedding-metadata",
            str(emb_meta),
            "--embedding-npz",
            str(emb_npz),
            "--embedding-run-id",
            "embed_run_1",
            "--output-dir",
            str(out_dir),
            "--batch-size",
            "2",
        ]
    )
    rc = args.func(args)
    assert rc == 0

    batch_plan = pd.read_csv(out_dir / "fg_batch_plan.tsv", sep="\t")
    assert batch_plan.shape[0] == 2
    assert set(batch_plan["n_mags"].tolist()) == {1, 2}

    summary = (out_dir / "fg_plan_summary.json").read_text(encoding="utf-8")
    assert "\"n_batches\": 2" in summary


def test_merge_reconciles_embeddings_and_features(tmp_path):
    module = _load_pipeline_module()

    plan_dir = tmp_path / "fg_artifacts"
    results_dir = plan_dir / "batch_results"
    out_dir = plan_dir
    plan_dir.mkdir(parents=True)

    embedding_index = pd.DataFrame(
        {
            "canonical_genome_id": ["mag_001", "mag_002", "mag_003"],
            "sample": ["mag_001", "mag_002", "mag_003"],
            "source": ["rumen", "rumen", "mucc"],
            "ecosystem": ["rumen", "rumen", "wetland"],
            "domain": ["Bacteria", "Bacteria", "Archaea"],
            "proteome_faa": ["/tmp/a.faa", "/tmp/b.faa", "/tmp/c.faa"],
            "proteome_sha256": ["", "", ""],
            "embedding_index": [0, 1, 2],
            "embedding_run_id": ["embed_run_1"] * 3,
            "embedding_npz_path": ["emb.npz"] * 3,
            "embedding_metadata_path": ["emb.tsv"] * 3,
        }
    )
    embedding_index.to_csv(
        plan_dir / "embedding_index_frozen.tsv",
        sep="\t",
        index=False,
    )

    batch_plan = pd.DataFrame(
        {
            "batch_id": [0, 1],
            "batch_name": ["batch_0000", "batch_0001"],
            "n_mags": [2, 1],
            "batch_manifest": ["x", "y"],
        }
    )
    batch_plan.to_csv(plan_dir / "fg_batch_plan.tsv", sep="\t", index=False)

    (results_dir / "batch_0000").mkdir(parents=True)
    (results_dir / "batch_0001").mkdir(parents=True)

    feat_cols = {
        "canonical_genome_id": ["mag_001", "mag_002"],
        "sample": ["mag_001", "mag_002"],
        "source": ["rumen", "rumen"],
        "ecosystem": ["rumen", "rumen"],
        "domain": ["Bacteria", "Bacteria"],
        "proteome_faa": ["/tmp/a.faa", "/tmp/b.faa"],
        "proteome_sha256": ["", ""],
        "batch_id": [0, 0],
        "batch_name": ["batch_0000", "batch_0000"],
        "methanogenic_potential": ["low", "high"],
        "mcrA": [1.0, 2.0],
        "mcrB": [0.0, 0.0],
        "mcrG": [0.0, 0.0],
        "pmoA": [0.5, 0.1],
        "mmoX": [0.0, 0.0],
        "dsrA": [0.0, 0.0],
        "dsrB": [0.0, 0.0],
        "nifH": [0.0, 0.0],
        "cbbL": [0.0, 0.0],
        "mtaB": [0.0, 0.0],
        "mttB": [0.0, 0.0],
        "mtbA": [0.0, 0.0],
        "mcrA_pmoA_ratio": [1.0, 2.0],
    }
    pd.DataFrame(feat_cols).to_csv(
        results_dir / "batch_0000" / "fg_features.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        columns=[
            "canonical_genome_id",
            "sample",
            "proteome_faa",
            "error_type",
            "error_message",
        ]
    ).to_csv(
        results_dir / "batch_0000" / "fg_failures.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(columns=list(feat_cols.keys())).to_csv(
        results_dir / "batch_0001" / "fg_features.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        [{
            "canonical_genome_id": "mag_003",
            "sample": "mag_003",
            "proteome_faa": "/tmp/c.faa",
            "error_type": "missing_proteome",
            "error_message": "proteome file not found",
        }]
    ).to_csv(results_dir / "batch_0001" / "fg_failures.tsv", sep="\t", index=False)

    args = module.build_parser().parse_args(
        [
            "merge",
            "--fg-plan-dir",
            str(plan_dir),
            "--batch-results-dir",
            str(results_dir),
            "--output-dir",
            str(out_dir),
            "--min-join-coverage",
            "0.60",
        ]
    )
    rc = args.func(args)
    assert rc == 0

    report = pd.read_csv(out_dir / "id_reconciliation_report.tsv", sep="\t")
    assert report.loc[0, "n_embeddings"] == 3
    assert report.loc[0, "n_joined"] == 2
    assert report.loc[0, "n_embedding_only"] == 1

    matrix = pd.read_csv(out_dir / "modeling_feature_matrix.tsv", sep="\t")
    assert matrix.shape[0] == 2
