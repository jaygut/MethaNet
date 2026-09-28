"""
MethaNet: molecular evidence for wetland methane research.

Genome- and proteome-level screening with foundation-model embeddings and
functional gene profiles, feeding the EmergentBiome evidence graph. The package
supports molecular review and study design only.

The legacy A-E classification scaffold in ``methanet.classification`` is an
unvalidated research prototype. It is deliberately not exported here: no output
may be reported as a methane-risk tier until sample mapping, abundance,
environmental covariates, uncertainty propagation and flux validation exist
(see docs/methanet_positioning_and_claims.md).

Example usage:
    >>> from methanet import FunctionalQuantifier, get_embedder
    >>> Embedder = get_embedder("esm2")
"""

__version__ = "1.0.0"
__author__ = "Philosof, Alon and Gutierrez, Jay"

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
    # Functional
    "FunctionalQuantifier",
    "FunctionalProfile",
    # Utilities
    "get_embedder",
    # Metadata
    "__version__",
    "__author__",
]
