"""
MethaNet: Microbial Methane Risk Assessment for Blue Carbon Verification

Transfer learning framework for predicting methane emission risk in coastal
ecosystems using functional gene signatures and foundation model embeddings.

The system leverages transfer learning from data-rich rumen microbiome studies
to predict methane dynamics in data-sparse coastal ecosystems. The key insight
is that methanogenic pathway conservation across environments enables
cross-domain prediction.

Example usage:
    >>> from methanet import MethaNetEnsemble, RiskTier
    >>> ensemble = MethaNetEnsemble()
    >>> ensemble.fit(X_train, y_train)
    >>> results = ensemble.classify_risk(X_test)
    >>> print(f"Risk Tier: {results[0].risk_tier}")
"""

__version__ = "1.0.0"
__author__ = "Philosof, Alon and Gutierrez, Jay"

# Core classification
from methanet.classification import (
    ClassificationResult,
    EnsembleConfig,
    MethaNetEnsemble,
    RiskTier,
)

# Functional gene quantification
from methanet.functional import FunctionalProfile, FunctionalQuantifier


# Embeddings (lazy import for optional dependencies)
def get_embedder(model_type: str = "esm2"):
    """Get embedding model (lazy import for optional dependencies).

    Args:
        model_type: One of 'esm2' or 'genomeocean'.

    Returns:
        Embedder class.
    """
    if model_type == "esm2":
        from methanet.embedding import ESM2Embedder
        return ESM2Embedder
    elif model_type == "genomeocean":
        from methanet.embedding import GenomeOceanEmbedder
        return GenomeOceanEmbedder
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    # Core classes
    "MethaNetEnsemble",
    "EnsembleConfig",
    "RiskTier",
    "ClassificationResult",
    # Functional
    "FunctionalQuantifier",
    "FunctionalProfile",
    # Utilities
    "get_embedder",
    # Metadata
    "__version__",
    "__author__",
]
