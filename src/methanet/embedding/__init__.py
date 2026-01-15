"""Foundation model embedding module.

This module provides embeddings from protein and genomic foundation models:
- ESM-2: Protein language model embeddings (1280-dim)
- GenomeOcean: Genome-level embeddings (3072-dim, optional)
- DNABERT-2: Genome embeddings via workflow scripts (768-dim)
- Feature fusion utilities for combining multiple embedding types
"""

from methanet.embedding.esm2 import ESM2Embedder, EmbeddingConfig
from methanet.embedding.genomeocean import GenomeOceanEmbedder, GenomeOceanConfig
from methanet.embedding.fusion import FeatureFusion, FusedFeatures

__all__ = [
    "ESM2Embedder",
    "EmbeddingConfig",
    "GenomeOceanEmbedder",
    "GenomeOceanConfig",
    "FeatureFusion",
    "FusedFeatures",
]
