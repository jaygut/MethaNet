"""Legacy ensemble classification scaffold (unvalidated research prototype).

Nothing in this module is calibrated against measured methane. Its A-E outputs
are not methane-risk tiers and must not be reported as such until the gates in
docs/methanet_positioning_and_claims.md are met. It is kept for research and is
not exported from the top-level ``methanet`` package.

Original design notes follow.

This module provides the core MethaNet ensemble classifier combining:
- XGBoost (weight: 0.35)
- Neural Network MLP (weight: 0.30)
- Random Forest (weight: 0.20)
- FAISS k-NN similarity (weight: 0.15)

Output: Risk scores (0-100) with 95% confidence intervals,
mapped to risk tiers A-E.
"""

from methanet.classification.ensemble import EnsembleConfig, MethaNetEnsemble
from methanet.classification.risk_tiers import ClassificationResult, RiskTier

__all__ = [
    "MethaNetEnsemble",
    "EnsembleConfig",
    "RiskTier",
    "ClassificationResult",
]
