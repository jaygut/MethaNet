"""Ensemble classification module.

This module provides the core MethaNet ensemble classifier combining:
- XGBoost (weight: 0.35)
- Neural Network MLP (weight: 0.30)
- Random Forest (weight: 0.20)
- FAISS k-NN similarity (weight: 0.15)

Output: Risk scores (0-100) with 95% confidence intervals,
mapped to risk tiers A-E.
"""

from methanet.classification.risk_tiers import RiskTier, ClassificationResult
from methanet.classification.ensemble import MethaNetEnsemble, EnsembleConfig

__all__ = [
    "MethaNetEnsemble",
    "EnsembleConfig",
    "RiskTier",
    "ClassificationResult",
]
