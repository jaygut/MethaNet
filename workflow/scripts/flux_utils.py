"""Utilities for flux prediction scripts.

These helpers make the flux predictor stage robust to pilots where only a
subset of samples have measured flux labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class FluxData:
    """Container for labeled flux prediction data."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    sample_ids: list[str]
    stats: dict[str, int]


@dataclass
class FluxFeatures:
    """Container for unlabeled feature matrices aligned to training columns."""

    X: np.ndarray
    feature_names: list[str]
    sample_ids: list[str]
    stats: dict[str, int]


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing required feature columns: {preview}")


def _default_sample_ids(df: pd.DataFrame, sample_id_col: str) -> list[str]:
    if sample_id_col in df.columns:
        return df[sample_id_col].astype(str).tolist()
    return df.index.astype(str).tolist()


def _finite_row_mask(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    return np.isfinite(values).all(axis=1)


def load_labeled_flux_data(
    path: Path,
    target: str,
    feature_names: Sequence[str] | None = None,
    sample_id_col: str = "sample_id",
) -> FluxData:
    """Load and clean labeled flux data from a fused feature table.

    Rows are retained only if:
    - the target label is present and finite
    - all selected feature columns are finite
    """
    df = pd.read_parquet(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in features file.")

    sample_ids_all = _default_sample_ids(df, sample_id_col)

    numeric = df.select_dtypes(include=["number"]).copy()
    if target not in numeric.columns:
        raise ValueError(f"Target column '{target}' is not numeric.")

    if feature_names is None:
        feature_names = [col for col in numeric.columns if col != target]
    else:
        feature_names = list(feature_names)

    _require_columns(numeric, [target, *feature_names])

    y_all = numeric[target].to_numpy(dtype=float)
    X_all = numeric[feature_names].to_numpy(dtype=float)

    target_mask = np.isfinite(y_all)
    feature_mask = _finite_row_mask(X_all)
    mask = target_mask & feature_mask

    X = X_all[mask]
    y = y_all[mask]
    sample_ids = [sid for sid, keep in zip(sample_ids_all, mask) if keep]

    stats = {
        "n_total": int(len(df)),
        "n_target_finite": int(target_mask.sum()),
        "n_features_finite": int(feature_mask.sum()),
        "n_labeled_clean": int(mask.sum()),
    }

    if stats["n_labeled_clean"] == 0:
        raise ValueError(
            "No labeled rows remain after filtering for finite targets and features."
        )

    return FluxData(
        X=X,
        y=y,
        feature_names=list(feature_names),
        sample_ids=sample_ids,
        stats=stats,
    )


def load_aligned_features(
    path: Path,
    feature_names: Sequence[str],
    sample_id_col: str = "sample_id",
) -> FluxFeatures:
    """Load features aligned to training feature names.

    Rows are retained only if all selected feature columns are finite.
    """
    df = pd.read_parquet(path)
    sample_ids_all = _default_sample_ids(df, sample_id_col)

    numeric = df.select_dtypes(include=["number"]).copy()
    feature_names = list(feature_names)
    _require_columns(numeric, feature_names)

    X_all = numeric[feature_names].to_numpy(dtype=float)
    mask = _finite_row_mask(X_all)

    X = X_all[mask]
    sample_ids = [sid for sid, keep in zip(sample_ids_all, mask) if keep]

    stats = {
        "n_total": int(len(df)),
        "n_features_finite": int(mask.sum()),
    }

    if stats["n_features_finite"] == 0:
        raise ValueError("No rows have fully finite feature values.")

    return FluxFeatures(
        X=X,
        feature_names=feature_names,
        sample_ids=sample_ids,
        stats=stats,
    )
