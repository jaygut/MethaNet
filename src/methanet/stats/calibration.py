"""Calibration utilities for classification models."""

from typing import Dict, Tuple
import numpy as np


def calibration_curve_multiclass(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """Compute calibration curve data for multiclass probabilities."""
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array of shape (n_samples, n_classes)")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins, right=True)

    bin_accuracy = np.zeros(n_bins)
    bin_confidence = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for b in range(1, n_bins + 1):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_count[b - 1] = mask.sum()
        bin_accuracy[b - 1] = correct[mask].mean()
        bin_confidence[b - 1] = confidences[mask].mean()

    return {
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_count": bin_count,
        "bins": bins,
    }


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float]:
    """Compute ECE and MCE for multiclass predictions."""
    curve = calibration_curve_multiclass(probs, y_true, n_bins)
    counts = curve["bin_count"]
    total = counts.sum()
    if total == 0:
        return 0.0, 0.0

    gap = np.abs(curve["bin_accuracy"] - curve["bin_confidence"])
    ece = float((gap * counts).sum() / total)
    mce = float(gap.max()) if gap.size else 0.0
    return ece, mce
