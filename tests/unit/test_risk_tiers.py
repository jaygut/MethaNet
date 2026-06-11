"""Unit tests for risk tier classification."""

import pytest

from methanet.classification.risk_tiers import (
    ClassificationResult,
    RiskTier,
    get_monitoring_interval,
    get_tier_thresholds,
    score_to_tier,
)


class TestRiskTier:
    """Tests for RiskTier enum."""

    def test_tier_values(self):
        """Test that all tier values are defined."""
        assert RiskTier.A.value == "very_low"
        assert RiskTier.B.value == "low"
        assert RiskTier.C.value == "moderate"
        assert RiskTier.D.value == "elevated"
        assert RiskTier.E.value == "high"

    def test_tier_count(self):
        """Test that there are exactly 5 tiers."""
        assert len(RiskTier) == 5


class TestScoreToTier:
    """Tests for score_to_tier function."""

    def test_tier_a_boundary(self):
        """Test Tier A boundaries (0-10%)."""
        assert score_to_tier(0.0) == RiskTier.A
        assert score_to_tier(0.05) == RiskTier.A
        assert score_to_tier(0.099) == RiskTier.A

    def test_tier_b_boundary(self):
        """Test Tier B boundaries (10-25%)."""
        assert score_to_tier(0.10) == RiskTier.B
        assert score_to_tier(0.15) == RiskTier.B
        assert score_to_tier(0.249) == RiskTier.B

    def test_tier_c_boundary(self):
        """Test Tier C boundaries (25-45%)."""
        assert score_to_tier(0.25) == RiskTier.C
        assert score_to_tier(0.35) == RiskTier.C
        assert score_to_tier(0.449) == RiskTier.C

    def test_tier_d_boundary(self):
        """Test Tier D boundaries (45-65%)."""
        assert score_to_tier(0.45) == RiskTier.D
        assert score_to_tier(0.55) == RiskTier.D
        assert score_to_tier(0.649) == RiskTier.D

    def test_tier_e_boundary(self):
        """Test Tier E boundaries (65-100%)."""
        assert score_to_tier(0.65) == RiskTier.E
        assert score_to_tier(0.80) == RiskTier.E
        assert score_to_tier(1.0) == RiskTier.E

    def test_invalid_scores(self):
        """Test that invalid scores raise ValueError."""
        with pytest.raises(ValueError):
            score_to_tier(-0.1)
        with pytest.raises(ValueError):
            score_to_tier(1.1)


class TestMonitoringInterval:
    """Tests for monitoring interval function."""

    def test_tier_a_monitoring(self):
        """Test Tier A monitoring interval (24 months)."""
        assert get_monitoring_interval(RiskTier.A) == "24 months"

    def test_tier_b_monitoring(self):
        """Test Tier B monitoring interval (12 months)."""
        assert get_monitoring_interval(RiskTier.B) == "12 months"

    def test_tier_c_monitoring(self):
        """Test Tier C monitoring interval (6 months)."""
        assert get_monitoring_interval(RiskTier.C) == "6 months"

    def test_tier_d_monitoring(self):
        """Test Tier D monitoring interval (3 months)."""
        assert get_monitoring_interval(RiskTier.D) == "3 months"

    def test_tier_e_monitoring(self):
        """Test Tier E monitoring interval (monthly)."""
        assert get_monitoring_interval(RiskTier.E) == "1 month"


class TestTierThresholds:
    """Tests for tier threshold retrieval."""

    def test_thresholds_structure(self):
        """Test that thresholds have correct structure."""
        thresholds = get_tier_thresholds()

        assert "A" in thresholds
        assert "B" in thresholds
        assert "C" in thresholds
        assert "D" in thresholds
        assert "E" in thresholds

    def test_thresholds_values(self):
        """Test threshold values match PRD specifications."""
        thresholds = get_tier_thresholds()

        assert thresholds["A"] == (0.0, 0.10)
        assert thresholds["B"] == (0.10, 0.25)
        assert thresholds["C"] == (0.25, 0.45)
        assert thresholds["D"] == (0.45, 0.65)
        assert thresholds["E"] == (0.65, 1.0)


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_result_creation(self):
        """Test creating a classification result."""
        result = ClassificationResult(
            sample_id="test_sample",
            risk_tier=RiskTier.C,
            risk_score=35.0,
            confidence_lower=28.0,
            confidence_upper=42.0,
            component_scores={"xgboost": 0.35, "neural_net": 0.30},
        )

        assert result.sample_id == "test_sample"
        assert result.risk_tier == RiskTier.C
        assert result.risk_score == 35.0
        assert result.confidence_lower == 28.0
        assert result.confidence_upper == 42.0
        assert len(result.component_scores) == 2

    def test_result_monitoring_interval(self):
        """Test that result provides monitoring interval via tier."""
        result = ClassificationResult(
            sample_id="test",
            risk_tier=RiskTier.D,
            risk_score=55.0,
            confidence_lower=50.0,
            confidence_upper=60.0,
        )

        # Access monitoring interval through the tier
        assert result.risk_tier.monitoring_interval == "3 months"

    def test_result_confidence_width(self):
        """Test confidence interval width calculation."""
        result = ClassificationResult(
            sample_id="test",
            risk_tier=RiskTier.B,
            risk_score=20.0,
            confidence_lower=15.0,
            confidence_upper=25.0,
        )

        assert result.confidence_width == pytest.approx(10.0, rel=1e-3)

    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        result = ClassificationResult(
            sample_id="test",
            risk_tier=RiskTier.A,
            risk_score=5.0,
            confidence_lower=2.0,
            confidence_upper=8.0,
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["sample_id"] == "test"
        assert d["risk_tier"] == "A"
        assert d["risk_score"] == 5.0

    def test_result_is_high_risk(self):
        """Test high risk classification."""
        high_risk = ClassificationResult(
            risk_tier=RiskTier.E,
            risk_score=75.0,
            confidence_lower=70.0,
            confidence_upper=80.0,
        )
        low_risk = ClassificationResult(
            risk_tier=RiskTier.A,
            risk_score=5.0,
            confidence_lower=2.0,
            confidence_upper=8.0,
        )

        assert high_risk.is_high_risk
        assert not low_risk.is_high_risk

    def test_result_is_low_risk(self):
        """Test low risk classification."""
        low_risk = ClassificationResult(
            risk_tier=RiskTier.B,
            risk_score=15.0,
            confidence_lower=10.0,
            confidence_upper=20.0,
        )
        moderate = ClassificationResult(
            risk_tier=RiskTier.C,
            risk_score=35.0,
            confidence_lower=30.0,
            confidence_upper=40.0,
        )

        assert low_risk.is_low_risk
        assert not moderate.is_low_risk
