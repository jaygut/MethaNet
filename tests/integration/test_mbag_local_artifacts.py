from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from methanet.mbag import (
    MBAGPaths,
    build_functional_dataset,
    build_knn_graph,
    discover_completed_runs,
    load_crosswalk,
    load_embedding_artifacts,
)


PARQUET_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


@pytest.mark.skipif(not PARQUET_AVAILABLE, reason="pyarrow not available")
def test_mbag_loads_current_apolo_artifacts_when_present() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    paths = MBAGPaths(repo_root=repo_root)
    required = [
        paths.crosswalk,
        paths.functional_manifest,
        paths.embedding_artifact_dir / "genome_embeddings.npz",
        paths.per_mag_dir,
    ]
    if not all(path.exists() for path in required):
        pytest.skip("Apolo MethaNet MBAG artifacts are not present")

    crosswalk = load_crosswalk(paths.crosswalk, paths.functional_manifest)
    embeddings = load_embedding_artifacts(paths.embedding_artifact_dir)
    run_status, selected = discover_completed_runs(paths.per_mag_dir)

    assert len(crosswalk) == 662
    assert embeddings.embeddings.shape[0] == 662
    assert not run_status.empty
    assert len(selected) > 0

    subset = dict(list(selected.items())[:2])
    functional = build_functional_dataset(subset)
    assert set(functional.profiles["proteome_id"]) == set(subset)
    assert {
        "methane_feature_count",
        "sulfur_feature_count",
        "substrate_feature_count",
        "reliability_weight",
    }.issubset(functional.profiles.columns)

    graph = build_knn_graph(
        embeddings.metadata["proteome_id"].head(20),
        embeddings.embeddings[:20],
        embeddings.metadata["ecosystem"].head(20),
        k=3,
        metric="cosine",
    )
    assert graph.edges["source_id"].nunique() == 20
    assert graph.node_metrics["cross_domain_neighbor_fraction"].between(0, 1).all()
