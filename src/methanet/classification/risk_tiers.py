"""Risk tier definitions and classification results.

This module defines the MethaNet risk classification system:
- Tier A (0-10%): Very low risk, annual monitoring
- Tier B (10-25%): Low risk, semi-annual monitoring
- Tier C (25-45%): Moderate risk, quarterly monitoring
- Tier D (45-65%): Elevated risk, monthly monitoring
- Tier E (65-100%): High risk, continuous monitoring
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class RiskTier(Enum):
    """Risk tier classification for methane emission potential.

    Each tier corresponds to a range of risk scores and
    recommended monitoring intervals.
    """

    A = "very_low"  # 0-10%
    B = "low"  # 10-25%
    C = "moderate"  # 25-45%
    D = "elevated"  # 45-65%
    E = "high"  # 65-100%

    @classmethod
    def from_score(cls, score: float) -> "RiskTier":
        """Convert risk score (0-100) to tier.

        Args:
            score: Risk score percentage.

        Returns:
            Corresponding RiskTier.
        """
        if score < 10:
            return cls.A
        elif score < 25:
            return cls.B
        elif score < 45:
            return cls.C
        elif score < 65:
            return cls.D
        else:
            return cls.E

    @property
    def midpoint(self) -> float:
        """Midpoint of the tier's score range.

        Returns:
            Midpoint value for expected value calculations.
        """
        midpoints = {
            RiskTier.A: 5.0,
            RiskTier.B: 17.5,
            RiskTier.C: 35.0,
            RiskTier.D: 55.0,
            RiskTier.E: 82.5,
        }
        return midpoints[self]

    @property
    def monitoring_interval(self) -> str:
        """Recommended monitoring frequency.

        Returns:
            Human-readable interval description.
        """
        intervals = {
            RiskTier.A: "annual",
            RiskTier.B: "semi-annual",
            RiskTier.C: "quarterly",
            RiskTier.D: "monthly",
            RiskTier.E: "continuous",
        }
        return intervals[self]

    @property
    def score_range(self) -> tuple:
        """Score range boundaries for this tier.

        Returns:
            Tuple of (lower, upper) bounds.
        """
        ranges = {
            RiskTier.A: (0, 10),
            RiskTier.B: (10, 25),
            RiskTier.C: (25, 45),
            RiskTier.D: (45, 65),
            RiskTier.E: (65, 100),
        }
        return ranges[self]

    @property
    def description(self) -> str:
        """Human-readable tier description.

        Returns:
            Description for reports and UI.
        """
        descriptions = {
            RiskTier.A: "Very low methane emission risk",
            RiskTier.B: "Low methane emission risk",
            RiskTier.C: "Moderate methane emission risk",
            RiskTier.D: "Elevated methane emission risk",
            RiskTier.E: "High methane emission risk",
        }
        return descriptions[self]


@dataclass
class ClassificationResult:
    """Complete risk classification result.

    Attributes:
        risk_score: Overall risk score (0-100).
        risk_tier: Assigned risk tier (A-E).
        confidence_lower: Lower bound of 95% CI.
        confidence_upper: Upper bound of 95% CI.
        component_scores: Per-model risk scores.
        methane_component: Methane pathway contribution.
        stability_component: Ecosystem stability contribution.
        resilience_component: Resilience/recovery contribution.
        sample_id: Optional sample identifier.
    """

    risk_score: float
    risk_tier: RiskTier
    confidence_lower: float
    confidence_upper: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    methane_component: float = 0.0
    stability_component: float = 0.0
    resilience_component: float = 0.0
    sample_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "sample_id": self.sample_id,
            "risk_score": round(self.risk_score, 2),
            "risk_tier": self.risk_tier.name,
            "risk_tier_label": self.risk_tier.value,
            "confidence_interval": {
                "lower": round(self.confidence_lower, 2),
                "upper": round(self.confidence_upper, 2),
                "level": 0.95,
            },
            "component_scores": {
                k: round(v, 2) for k, v in self.component_scores.items()
            },
            "components": {
                "methane": round(self.methane_component, 2),
                "stability": round(self.stability_component, 2),
                "resilience": round(self.resilience_component, 2),
            },
            "monitoring": {
                "recommended_interval": self.risk_tier.monitoring_interval,
                "tier_description": self.risk_tier.description,
            },
        }

    @property
    def confidence_width(self) -> float:
        """Width of confidence interval."""
        return self.confidence_upper - self.confidence_lower

    @property
    def is_high_risk(self) -> bool:
        """Check if classified as high risk (D or E)."""
        return self.risk_tier in (RiskTier.D, RiskTier.E)

    @property
    def is_low_risk(self) -> bool:
        """Check if classified as low risk (A or B)."""
        return self.risk_tier in (RiskTier.A, RiskTier.B)


# Tier midpoints for expected value calculation
TIER_MIDPOINTS = {
    0: 5.0,  # A
    1: 17.5,  # B
    2: 35.0,  # C
    3: 55.0,  # D
    4: 82.5,  # E
}


# Convenience functions for common operations


def score_to_tier(score: float) -> RiskTier:
    """Convert risk score (0.0-1.0) to risk tier.

    Args:
        score: Risk score as probability (0.0-1.0).

    Returns:
        Corresponding RiskTier.

    Raises:
        ValueError: If score is outside [0, 1] range.
    """
    if score < 0 or score > 1:
        raise ValueError(f"Score must be between 0 and 1, got {score}")

    # Convert probability to percentage for from_score
    return RiskTier.from_score(score * 100)


def get_monitoring_interval(tier: RiskTier) -> str:
    """Get recommended monitoring interval for a tier.

    Args:
        tier: Risk tier.

    Returns:
        Human-readable monitoring interval.
    """
    intervals = {
        RiskTier.A: "24 months",
        RiskTier.B: "12 months",
        RiskTier.C: "6 months",
        RiskTier.D: "3 months",
        RiskTier.E: "Monthly",
    }
    return intervals[tier]


def get_tier_thresholds() -> Dict[str, tuple]:
    """Get threshold boundaries for all tiers.

    Returns:
        Dictionary mapping tier names to (lower, upper) bounds.
    """
    return {
        "A": (0.0, 0.10),
        "B": (0.10, 0.25),
        "C": (0.25, 0.45),
        "D": (0.45, 0.65),
        "E": (0.65, 1.0),
    }
