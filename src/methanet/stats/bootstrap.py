"""Bootstrap confidence interval computation.

This module implements bootstrap methods for uncertainty quantification
in both classification and regression settings.
"""

from typing import Callable, Optional, Tuple
import numpy as np


def bootstrap_ci(
    predictions: np.ndarray,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for scalar predictions.

    Args:
        predictions: Array of prediction values or bootstrap samples.
        n_iterations: Number of bootstrap resampling iterations.
        confidence_level: Desired confidence level (e.g., 0.95).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound) for the CI.
    """
    rng = np.random.default_rng(random_state)
    alpha = 1 - confidence_level

    n = len(predictions)
    bootstrap_means = np.zeros(n_iterations)

    for i in range(n_iterations):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_means[i] = np.mean(predictions[indices])

    # Percentile method
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper


def bootstrap_regression_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compute bootstrap CI for regression metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted values.
        metric_fn: Function computing metric from (y_true, y_pred).
        n_iterations: Number of bootstrap iterations.
        confidence_level: Desired confidence level.
        random_state: Random seed.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(random_state)
    alpha = 1 - confidence_level

    n = len(y_true)
    bootstrap_metrics = np.zeros(n_iterations)

    # Point estimate
    point_estimate = metric_fn(y_true, y_pred)

    for i in range(n_iterations):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_metrics[i] = metric_fn(y_true[indices], y_pred[indices])

    ci_lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))

    return point_estimate, ci_lower, ci_upper


def stratified_bootstrap_ci(
    predictions: np.ndarray,
    groups: np.ndarray,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute stratified bootstrap CI preserving group structure.

    Useful when samples come from different strata (e.g., ecosystems).

    Args:
        predictions: Array of prediction values.
        groups: Array of group labels for each prediction.
        n_iterations: Number of bootstrap iterations.
        confidence_level: Desired confidence level.
        random_state: Random seed.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    rng = np.random.default_rng(random_state)
    alpha = 1 - confidence_level

    unique_groups = np.unique(groups)
    bootstrap_means = np.zeros(n_iterations)

    for i in range(n_iterations):
        resampled = []
        for group in unique_groups:
            group_mask = groups == group
            group_preds = predictions[group_mask]
            n_group = len(group_preds)
            indices = rng.choice(n_group, size=n_group, replace=True)
            resampled.extend(group_preds[indices])

        bootstrap_means[i] = np.mean(resampled)

    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper
