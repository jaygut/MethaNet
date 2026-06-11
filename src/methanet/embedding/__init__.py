"""Foundation model embedding module.

This module provides embeddings from protein and genomic foundation models:
- ESM-2: Protein language model embeddings (1280-dim)
- GenomeOcean: Genome-level embeddings (3072-dim, optional)
- DNABERT-2: Genome embeddings via workflow scripts (768-dim)
- Feature fusion utilities for combining multiple embedding types
"""

from methanet.embedding.fusion import FeatureFusion, FusedFeatures, FusionConfig


def __getattr__(name: str):
    """Lazy-load optional foundation model backends."""
    if name in {"EmbeddingConfig", "ESM2Embedder"}:
        from methanet.embedding.esm2 import EmbeddingConfig, ESM2Embedder

        return {
            "EmbeddingConfig": EmbeddingConfig,
            "ESM2Embedder": ESM2Embedder,
        }[name]

    if name in {"GenomeOceanConfig", "GenomeOceanEmbedder"}:
        from methanet.embedding.genomeocean import (
            GenomeOceanConfig,
            GenomeOceanEmbedder,
        )

        return {
            "GenomeOceanConfig": GenomeOceanConfig,
            "GenomeOceanEmbedder": GenomeOceanEmbedder,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ESM2Embedder",
    "EmbeddingConfig",
    "GenomeOceanEmbedder",
    "GenomeOceanConfig",
    "FeatureFusion",
    "FusionConfig",
    "FusedFeatures",
]
