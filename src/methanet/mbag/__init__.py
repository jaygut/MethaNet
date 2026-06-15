"""MethaNet Bridge Attestation Graph utilities.

The MBAG package turns ESM2 bridge hypotheses and functional MAG evidence into
auditable, provisional candidate-prioritization artifacts. It is deliberately
MAG-level and screening-oriented; it does not produce calibrated MRV risk.
"""

from methanet.mbag.core import (
    GraphResult,
    LeakageAudit,
    TransportResult,
    build_knn_graph,
    compute_reliability_weight,
    provisional_bridge_scores,
    sinkhorn_transport,
    source_leakage_audit,
)
from methanet.mbag.data import (
    EmbeddingArtifacts,
    FunctionalDataset,
    MBAGPaths,
    build_functional_dataset,
    discover_completed_runs,
    load_crosswalk,
    load_embedding_artifacts,
)

__all__ = [
    "EmbeddingArtifacts",
    "FunctionalDataset",
    "GraphResult",
    "LeakageAudit",
    "MBAGPaths",
    "TransportResult",
    "build_functional_dataset",
    "build_knn_graph",
    "compute_reliability_weight",
    "discover_completed_runs",
    "load_crosswalk",
    "load_embedding_artifacts",
    "provisional_bridge_scores",
    "sinkhorn_transport",
    "source_leakage_audit",
]
