"""Domain adaptation module for transfer learning.

This module provides domain adaptation techniques for transferring
knowledge from data-rich source domains (rumen) to data-sparse
target domains (coastal ecosystems).

Methods implemented:
- MMD (Maximum Mean Discrepancy): Aligns mean embeddings
- CORAL (Correlation Alignment): Aligns covariance structures
- MCW (MMD-CORAL Weighted): Combined approach with neural adaptation
"""

from methanet.domain_adapt.losses import MMDLoss, CORALLoss
from methanet.domain_adapt.mcw import (
    DomainAdapter,
    DomainAdaptConfig,
    FeatureEncoder,
    DomainAdaptiveClassifier,
)

__all__ = [
    "MMDLoss",
    "CORALLoss",
    "DomainAdapter",
    "DomainAdaptConfig",
    "FeatureEncoder",
    "DomainAdaptiveClassifier",
]
