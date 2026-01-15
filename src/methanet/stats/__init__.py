"""Statistical analysis module.

This module provides statistical methods for model evaluation:
- Bootstrap confidence intervals
- Classification metrics computation
- Cross-validation utilities
"""

from methanet.stats.bootstrap import bootstrap_ci, bootstrap_regression_ci
from methanet.stats.calibration import (
    calibration_curve_multiclass,
    expected_calibration_error,
)
from methanet.stats.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)

__all__ = [
    "bootstrap_ci",
    "bootstrap_regression_ci",
    "calibration_curve_multiclass",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "expected_calibration_error",
]
