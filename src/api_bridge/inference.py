"""ONNX inference utilities for MethaNet production deployment.

This module provides efficient inference using ONNX Runtime
for deployed MethaNet models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

try:
    import onnxruntime as ort

    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

from methanet.classification.risk_tiers import RiskTier, TIER_MIDPOINTS


@dataclass
class InferenceConfig:
    """Configuration for ONNX inference.

    Attributes:
        use_gpu: Use GPU acceleration if available.
        num_threads: Number of CPU threads for inference.
        optimization_level: Graph optimization level (0-3).
    """

    use_gpu: bool = True
    num_threads: int = 4
    optimization_level: int = 3


class ONNXInference:
    """ONNX Runtime inference engine for MethaNet.

    Provides efficient batch inference with optional GPU acceleration.
    """

    def __init__(
        self,
        model_path: Path,
        config: Optional[InferenceConfig] = None,
    ):
        """Initialize ONNX inference engine.

        Args:
            model_path: Path to ONNX model file.
            config: Inference configuration.

        Raises:
            ImportError: If ONNX Runtime not available.
            FileNotFoundError: If model file not found.
        """
        if not ORT_AVAILABLE:
            raise ImportError(
                "ONNX Runtime required. Install: pip install onnxruntime-gpu"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.config = config or InferenceConfig()
        self.session = self._create_session()

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _create_session(self) -> "ort.InferenceSession":
        """Create ONNX Runtime session with configured options.

        Returns:
            Configured InferenceSession.
        """
        # Session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.config.num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel(
            self.config.optimization_level
        )

        # Execution providers
        providers = []
        if self.config.use_gpu:
            # Try GPU providers
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            elif "CoreMLExecutionProvider" in available_providers:
                providers.append("CoreMLExecutionProvider")

        providers.append("CPUExecutionProvider")

        return ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Run inference and get class probabilities.

        Args:
            X: Input features [n_samples, n_features].

        Returns:
            Probability matrix [n_samples, n_classes].
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Ensure float32 for ONNX
        X = X.astype(np.float32)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: X})

        return outputs[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference and get predicted class labels.

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
    ) -> List[dict]:
        """Classify risk with full results.

        Args:
            X: Input features.
            sample_ids: Optional sample identifiers.

        Returns:
            List of result dictionaries.
        """
        probs = self.predict_proba(X)
        n_samples = X.shape[0]
        n_classes = probs.shape[1]

        # Tier midpoints
        tier_weights = np.array([TIER_MIDPOINTS.get(i, 50) for i in range(n_classes)])

        results = []
        for i in range(n_samples):
            risk_score = float(np.dot(probs[i], tier_weights))
            risk_tier = RiskTier.from_score(risk_score)

            result = {
                "sample_id": sample_ids[i] if sample_ids else f"sample_{i}",
                "risk_score": round(risk_score, 2),
                "risk_tier": risk_tier.name,
                "risk_tier_label": risk_tier.value,
                "monitoring_interval": risk_tier.monitoring_interval,
                "class_probabilities": {
                    f"tier_{chr(65+j)}": round(float(probs[i, j]), 4)
                    for j in range(n_classes)
                },
            }
            results.append(result)

        return results

    @property
    def input_shape(self) -> tuple:
        """Get expected input shape."""
        return tuple(self.session.get_inputs()[0].shape)

    @property
    def output_shape(self) -> tuple:
        """Get output shape."""
        return tuple(self.session.get_outputs()[0].shape)

    @property
    def providers(self) -> List[str]:
        """Get active execution providers."""
        return self.session.get_providers()


def batch_inference(
    model_path: Path,
    X: np.ndarray,
    batch_size: int = 64,
    config: Optional[InferenceConfig] = None,
) -> np.ndarray:
    """Run batched inference for large datasets.

    Args:
        model_path: Path to ONNX model.
        X: Full input array.
        batch_size: Batch size for inference.
        config: Optional inference config.

    Returns:
        Full probability matrix.
    """
    engine = ONNXInference(model_path, config)

    n_samples = X.shape[0]
    all_probs = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = X[start:end]
        probs = engine.predict_proba(batch)
        all_probs.append(probs)

    return np.vstack(all_probs)
