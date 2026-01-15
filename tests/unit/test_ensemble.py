"""Unit tests for ensemble classification."""

import pytest
import numpy as np

try:
    from methanet.classification.ensemble import (
        MethaNetEnsemble,
        EnsembleConfig,
    )
    from methanet.classification.risk_tiers import RiskTier
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


# Skip tests if sklearn not available
pytestmark = pytest.mark.skipif(
    not IMPORTS_AVAILABLE,
    reason="scikit-learn not available"
)


class TestEnsembleConfig:
    """Tests for EnsembleConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EnsembleConfig()

        assert config.model_weights["xgboost"] == 0.35
        assert config.model_weights["neural_net"] == 0.30
        assert config.model_weights["random_forest"] == 0.20
        assert config.model_weights["faiss_knn"] == 0.15
        assert config.bootstrap_iterations == 1000
        assert config.confidence_level == 0.95
        assert config.random_state == 42

    def test_custom_config(self):
        """Test custom configuration."""
        config = EnsembleConfig(
            model_weights={"xgboost": 0.5, "random_forest": 0.5},
            bootstrap_iterations=500,
            random_state=123,
        )

        assert config.model_weights["xgboost"] == 0.5
        assert config.bootstrap_iterations == 500
        assert config.random_state == 123

    def test_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        config = EnsembleConfig()
        total_weight = sum(config.model_weights.values())

        assert total_weight == pytest.approx(1.0, rel=1e-6)


class TestMethaNetEnsemble:
    """Tests for MethaNetEnsemble classifier."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 5, n_samples)  # 5 risk tiers

        return X, y

    @pytest.fixture
    def simple_config(self):
        """Create a simple config for testing."""
        return EnsembleConfig(
            model_weights={"random_forest": 1.0},  # Single model for speed
            bootstrap_iterations=10,  # Reduced for testing
            random_state=42,
        )

    def test_ensemble_init(self, simple_config):
        """Test ensemble initialization."""
        ensemble = MethaNetEnsemble(simple_config)

        assert ensemble.config == simple_config
        assert ensemble.models == {}
        assert not ensemble.is_fitted

    def test_ensemble_fit(self, sample_data, simple_config):
        """Test ensemble fitting."""
        X, y = sample_data
        ensemble = MethaNetEnsemble(simple_config)

        ensemble.fit(X, y)

        assert ensemble.is_fitted
        assert "random_forest" in ensemble.models

    def test_ensemble_predict(self, sample_data, simple_config):
        """Test ensemble prediction."""
        X, y = sample_data
        ensemble = MethaNetEnsemble(simple_config)
        ensemble.fit(X, y)

        predictions = ensemble.predict(X[:10])

        assert len(predictions) == 10
        assert all(isinstance(p, int) for p in predictions)
        assert all(0 <= p <= 4 for p in predictions)

    def test_ensemble_predict_proba(self, sample_data, simple_config):
        """Test probability prediction."""
        X, y = sample_data
        ensemble = MethaNetEnsemble(simple_config)
        ensemble.fit(X, y)

        proba = ensemble.predict_proba(X[:10])

        assert proba.shape == (10, 5)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_ensemble_classify_risk(self, sample_data, simple_config):
        """Test risk classification."""
        X, y = sample_data
        ensemble = MethaNetEnsemble(simple_config)
        ensemble.fit(X, y)

        results = ensemble.classify_risk(X[:5])

        assert len(results) == 5
        for result in results:
            assert isinstance(result.risk_tier, RiskTier)
            assert 0 <= result.score <= 1
            assert result.ci_lower <= result.score <= result.ci_upper

    def test_ensemble_not_fitted_error(self, simple_config):
        """Test error when predicting without fitting."""
        ensemble = MethaNetEnsemble(simple_config)
        X = np.random.randn(10, 50)

        with pytest.raises(RuntimeError, match="not fitted"):
            ensemble.predict(X)

    def test_ensemble_feature_importance(self, sample_data, simple_config):
        """Test feature importance extraction."""
        X, y = sample_data
        ensemble = MethaNetEnsemble(simple_config)
        ensemble.fit(X, y)

        importances = ensemble.get_feature_importance()

        assert len(importances) == X.shape[1]
        assert all(v >= 0 for v in importances.values())

    def test_ensemble_save_load(self, sample_data, simple_config, tmp_path):
        """Test model saving and loading."""
        X, y = sample_data
        ensemble = MethaNetEnsemble(simple_config)
        ensemble.fit(X, y)

        # Save
        save_path = tmp_path / "ensemble.joblib"
        ensemble.save(save_path)

        # Load
        loaded = MethaNetEnsemble.load(save_path)

        # Compare predictions
        original_pred = ensemble.predict(X[:5])
        loaded_pred = loaded.predict(X[:5])

        assert np.array_equal(original_pred, loaded_pred)


class TestEnsembleWeighting:
    """Tests for ensemble weighting behavior."""

    @pytest.fixture
    def mock_predictions(self):
        """Create mock predictions from multiple models."""
        # 5 samples, 5 classes
        return {
            "xgboost": np.array([
                [0.8, 0.1, 0.05, 0.03, 0.02],
                [0.1, 0.7, 0.1, 0.05, 0.05],
                [0.05, 0.1, 0.7, 0.1, 0.05],
                [0.02, 0.03, 0.1, 0.75, 0.1],
                [0.01, 0.02, 0.02, 0.1, 0.85],
            ]),
            "neural_net": np.array([
                [0.9, 0.05, 0.02, 0.02, 0.01],
                [0.15, 0.6, 0.15, 0.05, 0.05],
                [0.1, 0.15, 0.6, 0.1, 0.05],
                [0.05, 0.05, 0.15, 0.65, 0.1],
                [0.02, 0.03, 0.05, 0.15, 0.75],
            ]),
        }

    def test_weighted_average(self, mock_predictions):
        """Test that weighted average is computed correctly."""
        weights = {"xgboost": 0.6, "neural_net": 0.4}

        expected = (
            weights["xgboost"] * mock_predictions["xgboost"] +
            weights["neural_net"] * mock_predictions["neural_net"]
        )

        # Verify probabilities sum to 1
        assert np.allclose(expected.sum(axis=1), 1.0)

        # First sample should have highest probability for class 0
        assert expected[0].argmax() == 0

    def test_equal_weights_equivalent_to_mean(self, mock_predictions):
        """Test that equal weights give simple average."""
        weights = {"xgboost": 0.5, "neural_net": 0.5}

        weighted = (
            weights["xgboost"] * mock_predictions["xgboost"] +
            weights["neural_net"] * mock_predictions["neural_net"]
        )

        simple_mean = np.mean(
            [mock_predictions["xgboost"], mock_predictions["neural_net"]],
            axis=0
        )

        assert np.allclose(weighted, simple_mean)
