"""MMD-CORAL Weighted (MCW) domain adaptation.

This module implements the combined MMD-CORAL domain adaptation
approach with neural network feature encoding for transfer learning
from rumen to coastal ecosystems.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from methanet.domain_adapt.losses import CORALLoss, MMDLoss


@dataclass
class DomainAdaptConfig:
    """Configuration for domain adaptation.

    Attributes:
        hidden_dims: Hidden layer dimensions for encoder.
        mmd_weight: Weight for MMD loss term.
        coral_weight: Weight for CORAL loss term.
        dropout: Dropout probability.
        learning_rate: Optimizer learning rate.
        batch_size: Training batch size.
        epochs: Maximum training epochs.
        patience: Early stopping patience.
        device: Compute device ('cuda', 'cpu', 'auto').
    """

    hidden_dims: Tuple[int, ...] = (1024, 512, 256)
    mmd_weight: float = 0.1
    coral_weight: float = 0.1
    dropout: float = 0.3
    learning_rate: float = 1e-4
    batch_size: int = 64
    epochs: int = 100
    patience: int = 10
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = (
                "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
            )


if TORCH_AVAILABLE:

    class FeatureEncoder(nn.Module):
        """Shared feature encoder for domain adaptation.

        Multi-layer MLP with batch normalization and dropout
        for learning domain-invariant representations.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: Tuple[int, ...],
            dropout: float = 0.3,
        ):
            """Initialize encoder.

            Args:
                input_dim: Input feature dimension.
                hidden_dims: Tuple of hidden layer sizes.
                dropout: Dropout probability.
            """
            super().__init__()

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                prev_dim = hidden_dim

            self.encoder = nn.Sequential(*layers)
            self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through encoder."""
            return self.encoder(x)

    class DomainAdaptiveClassifier(nn.Module):
        """Domain-adaptive classifier with MMD-CORAL regularization.

        Combines a shared feature encoder with domain adaptation
        losses for transfer learning.
        """

        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            config: DomainAdaptConfig,
        ):
            """Initialize classifier.

            Args:
                input_dim: Input feature dimension.
                num_classes: Number of output classes.
                config: Domain adaptation configuration.
            """
            super().__init__()
            self.config = config

            # Shared encoder
            self.encoder = FeatureEncoder(
                input_dim, config.hidden_dims, config.dropout
            )

            # Classifier head
            self.classifier = nn.Linear(self.encoder.output_dim, num_classes)

            # Domain adaptation losses
            self.mmd_loss = MMDLoss()
            self.coral_loss = CORALLoss()

        def forward(
            self,
            x: torch.Tensor,
            return_features: bool = False,
        ) -> torch.Tensor:
            """Forward pass through encoder and classifier.

            Args:
                x: Input features.
                return_features: If True, also return encoded features.

            Returns:
                Class logits, and optionally encoded features.
            """
            features = self.encoder(x)
            logits = self.classifier(features)

            if return_features:
                return logits, features
            return logits

        def compute_adaptation_loss(
            self,
            source_features: torch.Tensor,
            target_features: torch.Tensor,
        ) -> torch.Tensor:
            """Compute combined MMD + CORAL loss.

            Args:
                source_features: Encoded source domain features.
                target_features: Encoded target domain features.

            Returns:
                Weighted sum of MMD and CORAL losses.
            """
            mmd = self.mmd_loss(source_features, target_features)
            coral = self.coral_loss(source_features, target_features)

            return self.config.mmd_weight * mmd + self.config.coral_weight * coral

    class DomainAdapter:
        """Train and apply domain-adaptive classifier.

        This class handles the training loop with domain adaptation
        and provides prediction methods for new samples.
        """

        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            config: Optional[DomainAdaptConfig] = None,
        ):
            """Initialize domain adapter.

            Args:
                input_dim: Input feature dimension.
                num_classes: Number of output classes (risk tiers).
                config: Domain adaptation configuration.
            """
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch required. Install: pip install torch"
                )

            self.config = config or DomainAdaptConfig()
            self.device = torch.device(self.config.device)

            self.model = DomainAdaptiveClassifier(
                input_dim, num_classes, self.config
            ).to(self.device)

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-5,
            )

            self.criterion = nn.CrossEntropyLoss()
            self._is_fitted = False

        def fit(
            self,
            source_X: np.ndarray,
            source_y: np.ndarray,
            target_X: np.ndarray,
            target_y: Optional[np.ndarray] = None,
            verbose: bool = False,
        ) -> "DomainAdapter":
            """Train with domain adaptation.

            Args:
                source_X: Source domain features.
                source_y: Source domain labels.
                target_X: Target domain features (unlabeled OK).
                target_y: Optional target domain labels.
                verbose: Print training progress.

            Returns:
                Self for method chaining.
            """
            # Convert to tensors
            source_X = torch.FloatTensor(source_X).to(self.device)
            source_y = torch.LongTensor(source_y).to(self.device)
            target_X = torch.FloatTensor(target_X).to(self.device)

            source_dataset = TensorDataset(source_X, source_y)
            source_loader = DataLoader(
                source_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
            )

            best_loss = float("inf")
            patience_counter = 0

            for epoch in range(self.config.epochs):
                self.model.train()
                epoch_loss = 0.0

                for batch_X, batch_y in source_loader:
                    # Sample target batch (random subset)
                    idx = torch.randperm(target_X.size(0))[: batch_X.size(0)]
                    target_batch = target_X[idx]

                    # Forward pass
                    source_logits, source_features = self.model(
                        batch_X, return_features=True
                    )
                    _, target_features = self.model(
                        target_batch, return_features=True
                    )

                    # Task loss (classification on source)
                    task_loss = self.criterion(source_logits, batch_y)

                    # Domain adaptation loss
                    adapt_loss = self.model.compute_adaptation_loss(
                        source_features, target_features
                    )

                    # Combined loss
                    loss = task_loss + adapt_loss

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                # Early stopping
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.config.epochs}, "
                        f"Loss: {epoch_loss:.4f}"
                    )

            self._is_fitted = True
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict class labels.

            Args:
                X: Input features.

            Returns:
                Predicted class indices.
            """
            if not self._is_fitted:
                raise RuntimeError("Model not fitted. Call fit() first.")

            self.model.eval()
            X = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                logits = self.model(X)
                predictions = logits.argmax(dim=1)

            return predictions.cpu().numpy()

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict class probabilities.

            Args:
                X: Input features.

            Returns:
                Class probability matrix [n_samples, n_classes].
            """
            if not self._is_fitted:
                raise RuntimeError("Model not fitted. Call fit() first.")

            self.model.eval()
            X = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                logits = self.model(X)
                probs = torch.softmax(logits, dim=1)

            return probs.cpu().numpy()

        def get_adapted_features(self, X: np.ndarray) -> np.ndarray:
            """Get domain-adapted feature representations.

            Args:
                X: Input features.

            Returns:
                Encoded feature representations.
            """
            if not self._is_fitted:
                raise RuntimeError("Model not fitted. Call fit() first.")

            self.model.eval()
            X = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                _, features = self.model(X, return_features=True)

            return features.cpu().numpy()

        def save(self, path: str) -> None:
            """Save model to disk.

            Args:
                path: Path for saving model checkpoint.
            """
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config,
                },
                path,
            )

        @classmethod
        def load(cls, path: str, input_dim: int, num_classes: int) -> "DomainAdapter":
            """Load model from disk.

            Args:
                path: Path to model checkpoint.
                input_dim: Input feature dimension.
                num_classes: Number of output classes.

            Returns:
                Loaded DomainAdapter instance.
            """
            checkpoint = torch.load(path, map_location="cpu")
            config = checkpoint["config"]

            adapter = cls(input_dim, num_classes, config)
            adapter.model.load_state_dict(checkpoint["model_state_dict"])
            adapter.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            adapter._is_fitted = True

            return adapter

else:
    # Placeholder classes when PyTorch not available
    class FeatureEncoder:
        """Placeholder (requires PyTorch)."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Install: pip install torch")

    class DomainAdaptiveClassifier:
        """Placeholder (requires PyTorch)."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Install: pip install torch")

    class DomainAdapter:
        """Placeholder (requires PyTorch)."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Install: pip install torch")
