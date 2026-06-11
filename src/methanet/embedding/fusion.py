"""Feature fusion for combining multiple embedding types.

This module provides utilities for combining:
- Functional gene features (13-dim: 12 markers + ratio)
- ESM-2 protein embeddings (1280-dim)
- Genome embeddings (DNABERT-2 768-dim or GenomeOcean 3072-dim)
- Environmental metadata (~100-dim)

Total fused dimension depends on genome embedding backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

from methanet.functional.quantify import FunctionalProfile
from methanet.schema import (
    DEFAULT_FUNCTIONAL_PATHWAYS,
    DEFAULT_GENOME_DIM,
    DNABERT2_DIM,
    FUNCTIONAL_VECTOR_DIM,
)
from methanet.schema import (
    ESM2_DIM as SCHEMA_ESM2_DIM,
)
from methanet.schema import (
    GENOMEOCEAN_DIM as SCHEMA_GENOMEOCEAN_DIM,
)

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class FusedFeatures:
    """Container for fused feature vectors.

    Attributes:
        sample_id: Sample identifier.
        features: Full fused feature vector.
        functional_dim: Dimension of functional features.
        esm2_dim: Dimension of ESM-2 embeddings.
        genomeocean_dim: Dimension of genome embeddings.
        metadata_dim: Dimension of environmental metadata.
    """

    sample_id: str
    features: np.ndarray
    functional_dim: int = 13
    esm2_dim: int = 1280
    genomeocean_dim: int = DEFAULT_GENOME_DIM
    metadata_dim: int = 0

    @property
    def total_dim(self) -> int:
        """Total feature dimension."""
        return len(self.features)

    def get_functional(self) -> np.ndarray:
        """Extract functional gene features."""
        return self.features[: self.functional_dim]

    def get_esm2(self) -> np.ndarray:
        """Extract ESM-2 embeddings."""
        start = self.functional_dim
        end = start + self.esm2_dim
        return self.features[start:end]

    def get_genomeocean(self) -> np.ndarray:
        """Extract genome embeddings."""
        start = self.functional_dim + self.esm2_dim
        end = start + self.genomeocean_dim
        return self.features[start:end]

    def get_metadata(self) -> np.ndarray:
        """Extract environmental metadata features."""
        if self.metadata_dim == 0:
            return np.array([])
        start = self.functional_dim + self.esm2_dim + self.genomeocean_dim
        return self.features[start:]


@dataclass
class FusionConfig:
    """Configuration for feature fusion.

    Attributes:
        include_functional: Include functional gene features.
        include_esm2: Include ESM-2 embeddings.
        include_genomeocean: Include genome embeddings (GenomeOcean or DNABERT-2).
        include_metadata: Include environmental metadata.
        normalize: Apply L2 normalization to embeddings.
        functional_pathways: Number of pathway completeness features.
        genome_dim: Expected genome embedding dimension.
    """

    include_functional: bool = True
    include_esm2: bool = True
    include_genomeocean: bool = True
    include_metadata: bool = True
    normalize: bool = True
    # KEGG or broader pathway completeness scores appended after the 13-vector.
    functional_pathways: int = DEFAULT_FUNCTIONAL_PATHWAYS
    genome_dim: int = DEFAULT_GENOME_DIM


class FeatureFusion:
    """Combine multiple feature types into unified vectors.

    This class handles the fusion of:
    1. Functional gene abundances (mcrA, pmoA, dsrA, nifH, cbbL + pathway scores)
    2. ESM-2 protein embeddings (mean-pooled)
    3. Genome embeddings (GenomeOcean or DNABERT-2)
    4. Environmental covariates (optional)
    """

    # Feature dimensions
    BASE_FUNCTIONAL_DIM = FUNCTIONAL_VECTOR_DIM
    ESM2_DIM = SCHEMA_ESM2_DIM
    GENOMEOCEAN_DIM = SCHEMA_GENOMEOCEAN_DIM
    DNABERT2_DIM = DNABERT2_DIM

    def __init__(self, config: Optional[FusionConfig] = None):
        """Initialize feature fusion.

        Args:
            config: Fusion configuration.
        """
        self.config = config or FusionConfig()

    def fuse(
        self,
        sample_id: str,
        functional: Optional[Union[FunctionalProfile, np.ndarray]] = None,
        esm2_embedding: Optional[np.ndarray] = None,
        genomeocean_embedding: Optional[np.ndarray] = None,
        metadata: Optional[np.ndarray] = None,
        pathway_scores: Optional[np.ndarray] = None,
    ) -> FusedFeatures:
        """Fuse multiple feature types into single vector.

        Args:
            sample_id: Sample identifier.
            functional: Functional profile or vector.
            esm2_embedding: ESM-2 embedding array (1280-dim).
            genomeocean_embedding: Genome embedding array.
            metadata: Environmental metadata features.
            pathway_scores: KEGG pathway completeness scores.

        Returns:
            FusedFeatures container with concatenated features.
        """
        components = []

        # Functional features
        func_dim = 0
        if self.config.include_functional:
            functional_missing = functional is None
            if functional is not None:
                if isinstance(functional, FunctionalProfile):
                    func_vec = functional.to_vector()
                else:
                    func_vec = functional
            else:
                func_vec = np.full(self.BASE_FUNCTIONAL_DIM, np.nan)

            # Add pathway scores if provided
            if pathway_scores is not None:
                func_vec = np.concatenate([func_vec, pathway_scores])
            elif self.config.functional_pathways > 0:
                fill_value = np.nan if functional_missing else 0.0
                func_vec = np.concatenate(
                    [func_vec, np.full(self.config.functional_pathways, fill_value)]
                )

            func_dim = len(func_vec)
            components.append(func_vec)

        # ESM-2 embeddings
        esm2_dim = 0
        if self.config.include_esm2:
            if esm2_embedding is not None:
                esm2_vec = esm2_embedding
                if self.config.normalize:
                    norm = np.linalg.norm(esm2_vec)
                    if norm > 0:
                        esm2_vec = esm2_vec / norm
            else:
                esm2_vec = np.full(self.ESM2_DIM, np.nan)

            esm2_dim = len(esm2_vec)
            components.append(esm2_vec)

        # Genome embeddings
        go_dim = 0
        if self.config.include_genomeocean:
            if genomeocean_embedding is not None:
                go_vec = genomeocean_embedding
                if self.config.normalize:
                    norm = np.linalg.norm(go_vec)
                    if norm > 0:
                        go_vec = go_vec / norm
            else:
                go_vec = np.full(self.config.genome_dim, np.nan)

            go_dim = len(go_vec)
            components.append(go_vec)

        # Environmental metadata
        meta_dim = 0
        if self.config.include_metadata and metadata is not None:
            meta_dim = len(metadata)
            components.append(metadata)

        # Concatenate all components
        if components:
            fused = np.concatenate(components)
        else:
            fused = np.array([])

        return FusedFeatures(
            sample_id=sample_id,
            features=fused,
            functional_dim=func_dim,
            esm2_dim=esm2_dim,
            genomeocean_dim=go_dim,
            metadata_dim=meta_dim,
        )

    def fuse_batch(
        self,
        sample_ids: List[str],
        functional_profiles: Optional[List[FunctionalProfile]] = None,
        esm2_embeddings: Optional[Dict[str, np.ndarray]] = None,
        genomeocean_embeddings: Optional[Dict[str, np.ndarray]] = None,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> List[FusedFeatures]:
        """Fuse features for multiple samples.

        Args:
            sample_ids: List of sample identifiers.
            functional_profiles: Optional list of FunctionalProfile objects.
            esm2_embeddings: Optional dict mapping sample_id to ESM-2 embedding.
            genomeocean_embeddings: Optional dict mapping sample_id to genome embedding.
            metadata_df: Optional DataFrame with sample_id as index.

        Returns:
            List of FusedFeatures objects.
        """
        results = []

        for i, sample_id in enumerate(sample_ids):
            functional = (
                functional_profiles[i] if functional_profiles else None
            )
            esm2 = (
                esm2_embeddings.get(sample_id)
                if esm2_embeddings
                else None
            )
            genomeocean = (
                genomeocean_embeddings.get(sample_id)
                if genomeocean_embeddings
                else None
            )
            metadata = None
            if metadata_df is not None and sample_id in metadata_df.index:
                metadata = metadata_df.loc[sample_id].values

            results.append(
                self.fuse(
                    sample_id=sample_id,
                    functional=functional,
                    esm2_embedding=esm2,
                    genomeocean_embedding=genomeocean,
                    metadata=metadata,
                )
            )

        return results

    def to_matrix(self, fused_features: List[FusedFeatures]) -> np.ndarray:
        """Convert list of fused features to feature matrix.

        Args:
            fused_features: List of FusedFeatures objects.

        Returns:
            2D array of shape (n_samples, n_features).
        """
        if not fused_features:
            return np.array([])

        return np.stack([f.features for f in fused_features])

    @property
    def expected_dim(self) -> int:
        """Expected total feature dimension based on config."""
        dim = 0
        if self.config.include_functional:
            dim += self.BASE_FUNCTIONAL_DIM + self.config.functional_pathways
        if self.config.include_esm2:
            dim += self.ESM2_DIM
        if self.config.include_genomeocean:
            dim += self.config.genome_dim
        return dim
