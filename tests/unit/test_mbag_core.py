from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from methanet.mbag.core import (
    build_knn_graph,
    compute_reliability_weight,
    provisional_bridge_scores,
    sinkhorn_transport,
    source_leakage_audit,
)


def test_reliability_weight_is_bounded_and_coverage_limited() -> None:
    high = compute_reliability_weight(95.0, 1.0, True, 1.0, "resolved_genus")
    low_cov = compute_reliability_weight(95.0, 1.0, True, 0.1, "resolved_genus")
    contaminated = compute_reliability_weight(95.0, 15.0, True, 1.0, "resolved_genus")

    assert 0.0 <= high <= 1.0
    assert 0.0 <= low_cov < high
    assert 0.0 <= contaminated < high


def test_knn_graph_reports_cross_domain_neighbors() -> None:
    ids = ["a1", "a2", "b1", "b2"]
    domains = ["A", "A", "B", "B"]
    matrix = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
        ]
    )

    graph = build_knn_graph(ids, matrix, domains, k=2, metric="euclidean")

    assert set(graph.edges.columns) >= {"source_id", "target_id", "cross_domain", "weight"}
    assert set(graph.node_metrics["proteome_id"]) == set(ids)
    assert graph.node_metrics["cross_domain_neighbor_fraction"].between(0.0, 1.0).all()
    assert graph.edges.groupby("source_id").size().eq(2).all()


def test_sinkhorn_transport_returns_top_couplings_and_node_metrics() -> None:
    source_ids = ["s1", "s2"]
    target_ids = ["t1", "t2"]
    source = np.array([[0.0, 0.0], [1.0, 1.0]])
    target = np.array([[0.1, 0.0], [1.1, 1.0]])

    transport = sinkhorn_transport(
        source_ids,
        target_ids,
        source,
        target,
        metric="euclidean",
        epsilon=0.2,
        top_per_source=1,
    )

    assert len(transport.couplings) == 2
    assert set(transport.node_metrics["proteome_id"]) == set(source_ids + target_ids)
    assert transport.node_metrics["ot_best_coupling"].gt(0).all()
    assert transport.cost_summary["cost_median"] >= 0


def test_source_leakage_audit_detects_separable_labels() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(loc=-2.0, scale=0.1, size=(8, 3))
    right = rng.normal(loc=2.0, scale=0.1, size=(8, 3))
    matrix = np.vstack([left, right])
    labels = ["left"] * 8 + ["right"] * 8

    audit = source_leakage_audit(matrix, labels, label_name="toy")

    assert audit.status == "warn_high_leakage"
    assert audit.balanced_accuracy is not None
    assert audit.balanced_accuracy >= 0.8


def test_provisional_bridge_scores_adds_status_and_score() -> None:
    df = pd.DataFrame(
        {
            "proteome_id": ["a", "b", "c"],
            "cross_domain_neighbor_fraction": [0.1, 0.5, 0.9],
            "ot_best_coupling": [0.2, 0.3, 0.4],
            "mixing_coeff": [0.0, 0.2, 0.8],
            "functional_concordance": [0.0, 0.5, 1.0],
            "mechanism_support": [0.0, 0.2, 1.0],
            "candidate_specificity": [0.0, 0.1, 0.9],
            "qc_penalty": [0.0, 0.2, 0.1],
            "annotation_missingness": [1.0, 0.3, 0.1],
            "source_leakage_penalty": [0.2, 0.2, 0.2],
        }
    )

    observed = provisional_bridge_scores(df)

    assert "mbag_score_provisional" in observed.columns
    assert observed["mbag_score_status"].eq("provisional_internal").all()
    assert observed.loc[2, "mbag_score_provisional"] > observed.loc[0, "mbag_score_provisional"]


def test_knn_graph_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError, match="row count"):
        build_knn_graph(["a"], np.zeros((2, 2)), ["A"], k=1)
