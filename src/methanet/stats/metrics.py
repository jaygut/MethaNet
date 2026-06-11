"""Classification and regression metrics computation.

This module provides metrics for evaluating MethaNet models
including both standard ML metrics and domain-specific ones.
"""

from typing import Dict, Optional

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[list] = None,
) -> Dict:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        y_proba: Optional probability predictions for AUC.
        class_names: Optional names for classes.

    Returns:
        Dictionary of metric names to values.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn required. Install: pip install scikit-learn"
        )

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_precision": precision_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
    }

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    metrics["per_class_f1"] = {
        class_names[i]: float(per_class_f1[i]) for i in range(len(per_class_f1))
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # AUC-ROC if probabilities provided
    if y_proba is not None:
        try:
            if y_proba.ndim == 2:
                metrics["auc_roc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
            else:
                metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            # AUC not defined for some cases
            metrics["auc_roc"] = None

    # Classification report as dict
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    metrics["classification_report"] = report

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """Compute regression metrics for flux prediction.

    Args:
        y_true: True flux values.
        y_pred: Predicted flux values.

    Returns:
        Dictionary of metric names to values.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn required. Install: pip install scikit-learn"
        )

    metrics = {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mse": mean_squared_error(y_true, y_pred),
    }

    # Normalized metrics
    y_range = y_true.max() - y_true.min()
    if y_range > 0:
        metrics["nrmse"] = metrics["rmse"] / y_range
        metrics["nmae"] = metrics["mae"] / y_range

    # Mean absolute percentage error
    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = np.mean(
            np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
        )
        metrics["mape"] = mape * 100

    # Correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    metrics["pearson_r"] = correlation
    metrics["r2_correlation"] = correlation**2

    return metrics


def compute_transfer_metrics(
    source_metrics: Dict,
    target_metrics: Dict,
) -> Dict:
    """Compute metrics comparing source and target domain performance.

    Args:
        source_metrics: Metrics on source domain.
        target_metrics: Metrics on target domain.

    Returns:
        Dictionary with transfer-specific metrics.
    """
    transfer_metrics = {
        "source_accuracy": source_metrics.get("balanced_accuracy"),
        "target_accuracy": target_metrics.get("balanced_accuracy"),
    }

    # Performance drop
    if (
        source_metrics.get("balanced_accuracy")
        and target_metrics.get("balanced_accuracy")
    ):
        transfer_metrics["accuracy_drop"] = (
            source_metrics["balanced_accuracy"] - target_metrics["balanced_accuracy"]
        )
        transfer_metrics["transfer_ratio"] = (
            target_metrics["balanced_accuracy"] / source_metrics["balanced_accuracy"]
        )

    # F1 comparison
    if source_metrics.get("macro_f1") and target_metrics.get("macro_f1"):
        transfer_metrics["f1_drop"] = (
            source_metrics["macro_f1"] - target_metrics["macro_f1"]
        )

    return transfer_metrics
