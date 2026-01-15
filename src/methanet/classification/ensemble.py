"""Ensemble classifier for methane risk prediction.

This module implements the MethaNet ensemble combining four models
with optimized weights for coastal ecosystem prediction:
- XGBoost (35%): Gradient boosting for structured features
- Neural Network (30%): Deep learning for complex patterns
- Random Forest (20%): Robust baseline with OOB estimates
- FAISS k-NN (15%): Similarity-based for transfer learning
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from methanet.classification.risk_tiers import (
    RiskTier,
    ClassificationResult,
    TIER_MIDPOINTS,
)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble classifier.

    Attributes:
        model_weights: Weights for each model component.
        xgb_params: XGBoost hyperparameters.
        nn_params: Neural network hyperparameters.
        rf_params: Random forest hyperparameters.
        knn_k: Number of neighbors for FAISS k-NN.
        bootstrap_iterations: Number of bootstrap samples for CI.
        confidence_level: Confidence level for intervals.
        random_state: Random seed for reproducibility.
    """

    model_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "xgboost": 0.35,
            "neural_net": 0.30,
            "random_forest": 0.20,
            "faiss_knn": 0.15,
        }
    )

    xgb_params: Dict = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
        }
    )

    nn_params: Dict = field(
        default_factory=lambda: {
            "hidden_layer_sizes": (512, 256, 128),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.001,
            "batch_size": 64,
            "learning_rate_init": 0.001,
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
        }
    )

    rf_params: Dict = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "oob_score": True,
            "n_jobs": -1,
        }
    )

    knn_k: int = 10
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    random_state: int = 42
    ci_method: str = "bootstrap"  # bootstrap or approx
    calibration_method: Optional[str] = None  # isotonic or sigmoid


class MethaNetEnsemble:
    """Ensemble classifier for methane risk prediction.

    Combines XGBoost, neural network, random forest, and FAISS k-NN
    with configurable weights for robust prediction.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """Initialize ensemble.

        Args:
            config: Ensemble configuration.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn required. Install: pip install scikit-learn"
            )

        self.config = config or EnsembleConfig()
        self.scaler = StandardScaler()
        self.models: Dict = {}
        self.faiss_index = None
        self.train_labels = None
        self._train_X = None
        self._train_y = None
        self._is_fitted = False
        self._n_classes = 5  # Risk tiers A-E
        self._active_models: List[str] = []
        self._calibrator = None

    @property
    def is_fitted(self) -> bool:
        """Return whether the ensemble has been fitted."""
        return self._is_fitted

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "MethaNetEnsemble":
        """Train all ensemble components.

        Args:
            X: Training features.
            y: Training labels (0-4 for tiers A-E).
            sample_weight: Optional sample weights.

        Returns:
            Self for method chaining.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self._n_classes = len(np.unique(y))
        self.classes_ = np.arange(self._n_classes)
        self._train_X = np.asarray(X)
        self._train_y = np.asarray(y)
        self._active_models = [
            name for name, weight in self.config.model_weights.items() if weight > 0
        ]
        self._calibrator = None

        # Train XGBoost
        if "xgboost" in self._active_models and XGBOOST_AVAILABLE:
            self.models["xgboost"] = xgb.XGBClassifier(
                **self.config.xgb_params,
                num_class=self._n_classes,
                random_state=self.config.random_state,
            )
            self.models["xgboost"].fit(X_scaled, y, sample_weight=sample_weight)
        elif "xgboost" in self._active_models:
            # Fallback to sklearn GradientBoosting
            from sklearn.ensemble import GradientBoostingClassifier

            self.models["xgboost"] = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=self.config.random_state,
            )
            self.models["xgboost"].fit(X_scaled, y, sample_weight=sample_weight)

        # Train Neural Network
        if "neural_net" in self._active_models:
            self.models["neural_net"] = MLPClassifier(
                **self.config.nn_params,
                random_state=self.config.random_state,
            )
            self.models["neural_net"].fit(X_scaled, y)

        # Train Random Forest
        if "random_forest" in self._active_models:
            self.models["random_forest"] = RandomForestClassifier(
                **self.config.rf_params,
                random_state=self.config.random_state,
            )
            self.models["random_forest"].fit(X_scaled, y, sample_weight=sample_weight)

        # Build FAISS index
        if "faiss_knn" in self._active_models:
            self._build_faiss_index(X_scaled, y)

        self._is_fitted = True
        return self

    def _build_faiss_index(self, X: np.ndarray, y: np.ndarray) -> None:
        """Build FAISS index for k-NN component.

        Args:
            X: Scaled training features.
            y: Training labels.
        """
        if not FAISS_AVAILABLE:
            self.faiss_index = None
            self.train_labels = y.copy()
            self._train_X = X.copy()  # Fallback for sklearn k-NN
            return

        X_float32 = X.astype(np.float32)
        d = X_float32.shape[1]

        # Use IndexFlatL2 for exact search (suitable for <10k samples)
        self.faiss_index = faiss.IndexFlatL2(d)
        self.faiss_index.add(X_float32)
        self.train_labels = y.copy()

    def _predict_proba_uncalibrated(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability predictions.

        Args:
            X: Input features.

        Returns:
            Class probability matrix [n_samples, n_classes].
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        n_samples = X_scaled.shape[0]

        # Get predictions from each model
        probs = {}

        # XGBoost
        if "xgboost" in self.models:
            probs["xgboost"] = self.models["xgboost"].predict_proba(X_scaled)

        # Neural Network
        if "neural_net" in self.models:
            probs["neural_net"] = self.models["neural_net"].predict_proba(X_scaled)

        # Random Forest
        if "random_forest" in self.models:
            probs["random_forest"] = self.models["random_forest"].predict_proba(X_scaled)

        # FAISS k-NN (or sklearn fallback)
        if "faiss_knn" in self._active_models:
            probs["faiss_knn"] = self._faiss_predict_proba(X_scaled)

        # Weighted ensemble
        ensemble_probs = np.zeros((n_samples, self._n_classes))
        total_weight = 0.0

        for model_name, weight in self.config.model_weights.items():
            if model_name in probs:
                ensemble_probs += weight * probs[model_name]
                total_weight += weight

        if total_weight == 0:
            raise RuntimeError("No active models available for prediction.")
        ensemble_probs /= total_weight

        return ensemble_probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get calibrated or raw ensemble probabilities."""
        if self._calibrator is not None:
            return self._calibrator.predict_proba(X)
        return self._predict_proba_uncalibrated(X)

    def _faiss_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """k-NN probability prediction.

        Args:
            X: Scaled input features.

        Returns:
            Class probability matrix.
        """
        k = self.config.knn_k
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, self._n_classes))

        if FAISS_AVAILABLE and self.faiss_index is not None:
            X_float32 = X.astype(np.float32)
            distances, indices = self.faiss_index.search(X_float32, k)

            # Convert distances to weights (inverse distance)
            weights = 1.0 / (distances + 1e-6)
            weights /= weights.sum(axis=1, keepdims=True)

            for i in range(n_samples):
                for j, idx in enumerate(indices[i]):
                    label = int(self.train_labels[idx])
                    if 0 <= label < self._n_classes:
                        probs[i, label] += weights[i, j]
        else:
            # Fallback: sklearn NearestNeighbors
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
            nn.fit(self._train_X)
            distances, indices = nn.kneighbors(X)

            weights = 1.0 / (distances + 1e-6)
            weights /= weights.sum(axis=1, keepdims=True)

            for i in range(n_samples):
                for j, idx in enumerate(indices[i]):
                    label = int(self.train_labels[idx])
                    if 0 <= label < self._n_classes:
                        probs[i, label] += weights[i, j]

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Input features.

        Returns:
            Predicted class indices.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def classify_risk(
        self,
        X: np.ndarray,
        sample_ids: Optional[List[str]] = None,
        compute_ci: bool = True,
    ) -> List[ClassificationResult]:
        """Full risk classification with confidence intervals.

        Args:
            X: Input features.
            sample_ids: Optional sample identifiers.
            compute_ci: Whether to compute confidence intervals.

        Returns:
            List of ClassificationResult objects.
        """
        probs = self.predict_proba(X)
        n_samples = X.shape[0]
        results = []

        # Tier midpoints for expected value calculation
        tier_weights = np.array([TIER_MIDPOINTS[i] for i in range(self._n_classes)])

        ci_lower = None
        ci_upper = None
        if compute_ci:
            if self.config.ci_method == "bootstrap":
                ci_lower, ci_upper = self._bootstrap_ci_batch(X, tier_weights)
            else:
                ci_lower, ci_upper = self._approx_ci_batch(X, tier_weights)

        for i in range(n_samples):
            # Risk score = expected value over tier probabilities
            risk_score = float(np.dot(probs[i], tier_weights))

            # Compute confidence interval via bootstrap
            if compute_ci:
                ci_lower_i = float(ci_lower[i])
                ci_upper_i = float(ci_upper[i])
            else:
                ci_lower_i = max(0, risk_score - 5)
                ci_upper_i = min(100, risk_score + 5)

            # Per-model component scores
            X_scaled = self.scaler.transform(X[i : i + 1])
            component_scores = {}
            for name in ["xgboost", "neural_net", "random_forest"]:
                if name in self.models:
                    model_probs = self.models[name].predict_proba(X_scaled)[0]
                    component_scores[name] = float(np.dot(model_probs, tier_weights))

            # Feature-based components
            methane_component = self._compute_methane_component(X[i])
            stability_component = self._compute_stability_component(X[i])
            resilience_component = self._compute_resilience_component(X[i])

            sample_id = sample_ids[i] if sample_ids else None

            results.append(
                ClassificationResult(
                    risk_score=risk_score,
                    risk_tier=RiskTier.from_score(risk_score),
                    confidence_lower=ci_lower_i,
                    confidence_upper=ci_upper_i,
                    component_scores=component_scores,
                    methane_component=methane_component,
                    stability_component=stability_component,
                    resilience_component=resilience_component,
                    sample_id=sample_id,
                )
            )

        return results

    def _approx_ci_batch(
        self,
        X: np.ndarray,
        tier_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate CI by perturbing probabilities."""
        n_iter = self.config.bootstrap_iterations
        alpha = 1 - self.config.confidence_level
        base_probs = self.predict_proba(X)
        rng = np.random.default_rng(self.config.random_state)

        samples = []
        for _ in range(max(n_iter, 1)):
            noise = rng.normal(0, 0.02, size=base_probs.shape)
            noisy_probs = np.clip(base_probs + noise, 0, 1)
            noisy_probs /= noisy_probs.sum(axis=1, keepdims=True)
            scores = noisy_probs @ tier_weights
            samples.append(scores)

        scores = np.stack(samples, axis=0)
        lower = np.percentile(scores, 100 * alpha / 2, axis=0)
        upper = np.percentile(scores, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

    def _bootstrap_ci_batch(
        self,
        X: np.ndarray,
        tier_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute bootstrap CI by refitting on resampled training data."""
        if self._train_X is None or self._train_y is None:
            raise RuntimeError("Bootstrap CI requires stored training data.")

        n_iter = max(self.config.bootstrap_iterations, 1)
        alpha = 1 - self.config.confidence_level
        rng = np.random.default_rng(self.config.random_state)

        scores = np.zeros((n_iter, X.shape[0]))
        base_config = replace(
            self.config,
            ci_method="approx",
            bootstrap_iterations=0,
            calibration_method=None,
        )

        for i in range(n_iter):
            idx = rng.choice(len(self._train_X), size=len(self._train_X), replace=True)
            bootstrap_model = MethaNetEnsemble(base_config)
            bootstrap_model.fit(self._train_X[idx], self._train_y[idx])
            probs = bootstrap_model.predict_proba(X)
            scores[i] = probs @ tier_weights

        lower = np.percentile(scores, 100 * alpha / 2, axis=0)
        upper = np.percentile(scores, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

    def _compute_methane_component(self, x: np.ndarray) -> float:
        """Extract methane-specific component.

        Based on functional gene features (mcrA/pmoA ratio).
        """
        # Features 0-5 typically contain functional markers
        if len(x) > 5:
            ratio = x[5]  # mcrA/pmoA ratio
            return float(np.clip(ratio * 10 + 50, 0, 100))
        return 50.0

    def _compute_stability_component(self, x: np.ndarray) -> float:
        """Extract ecosystem stability component.

        Based on embedding variance and diversity metrics.
        """
        # Placeholder: would use embedding statistics
        return 50.0

    def _compute_resilience_component(self, x: np.ndarray) -> float:
        """Extract resilience component.

        Based on functional diversity and pathway completeness.
        """
        # Placeholder: would use pathway scores
        return 50.0

    def calibrate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: str = "isotonic",
    ) -> "MethaNetEnsemble":
        """Fit a probability calibrator using validation data."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for calibration.")
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        class _PrefitWrapper:
            def __init__(self, parent: "MethaNetEnsemble"):
                self.parent = parent
                self.classes_ = parent.classes_

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                return self.parent._predict_proba_uncalibrated(X)

            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.argmax(self.predict_proba(X), axis=1)

        wrapper = _PrefitWrapper(self)
        calibrator = CalibratedClassifierCV(wrapper, method=method, cv="prefit")
        calibrator.fit(X_val, y_val)
        self._calibrator = calibrator
        return self

    def get_feature_importance(self) -> Dict[int, float]:
        """Return aggregated feature importances where available."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_features = self.scaler.mean_.shape[0]
        importances = np.zeros(n_features, dtype=float)
        weight_sum = 0.0

        for name, model in self.models.items():
            weight = self.config.model_weights.get(name, 0.0)
            if weight <= 0:
                continue

            values = None
            if hasattr(model, "feature_importances_"):
                values = model.feature_importances_
            elif hasattr(model, "coef_"):
                values = np.mean(np.abs(model.coef_), axis=0)
            elif hasattr(model, "coefs_") and model.coefs_:
                values = np.mean(np.abs(model.coefs_[0]), axis=1)

            if values is None or len(values) != n_features:
                continue

            importances += weight * values
            weight_sum += weight

        if weight_sum == 0:
            raise RuntimeError("No feature importances available from active models.")

        importances /= weight_sum
        return {i: float(val) for i, val in enumerate(importances)}

    def save(self, path: Path) -> None:
        """Save ensemble to disk.

        Args:
            path: Directory path for saving.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save sklearn models
        joblib.dump(self.scaler, path / "scaler.joblib")
        joblib.dump(self.models.get("neural_net"), path / "neural_net.joblib")
        joblib.dump(self.models.get("random_forest"), path / "random_forest.joblib")

        # Save XGBoost
        if "xgboost" in self.models and XGBOOST_AVAILABLE:
            self.models["xgboost"].save_model(str(path / "xgboost.ubj"))
        elif "xgboost" in self.models:
            joblib.dump(self.models["xgboost"], path / "xgboost.joblib")

        # Save FAISS index
        if FAISS_AVAILABLE and self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(path / "faiss.index"))
        np.save(path / "train_labels.npy", self.train_labels)

        # Save config and metadata
        joblib.dump(self.config, path / "config.joblib")
        joblib.dump(
            {"n_classes": self._n_classes, "is_fitted": self._is_fitted},
            path / "metadata.joblib",
        )

    @classmethod
    def load(cls, path: Path) -> "MethaNetEnsemble":
        """Load ensemble from disk.

        Args:
            path: Directory path containing saved model.

        Returns:
            Loaded MethaNetEnsemble instance.
        """
        path = Path(path)

        config = joblib.load(path / "config.joblib")
        ensemble = cls(config)

        # Load sklearn models
        ensemble.scaler = joblib.load(path / "scaler.joblib")
        ensemble.models["neural_net"] = joblib.load(path / "neural_net.joblib")
        ensemble.models["random_forest"] = joblib.load(path / "random_forest.joblib")

        # Load XGBoost
        if (path / "xgboost.ubj").exists() and XGBOOST_AVAILABLE:
            ensemble.models["xgboost"] = xgb.XGBClassifier()
            ensemble.models["xgboost"].load_model(str(path / "xgboost.ubj"))
        elif (path / "xgboost.joblib").exists():
            ensemble.models["xgboost"] = joblib.load(path / "xgboost.joblib")

        # Load FAISS index
        if FAISS_AVAILABLE and (path / "faiss.index").exists():
            ensemble.faiss_index = faiss.read_index(str(path / "faiss.index"))
        ensemble.train_labels = np.load(path / "train_labels.npy")

        # Load metadata
        metadata = joblib.load(path / "metadata.joblib")
        ensemble._n_classes = metadata["n_classes"]
        ensemble._is_fitted = metadata["is_fitted"]

        return ensemble
