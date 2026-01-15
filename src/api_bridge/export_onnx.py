"""ONNX export utilities for MethaNet models.

This module provides functions to export trained PyTorch models
to ONNX format for production deployment.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnx
    import onnx.checker

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def export_neural_net_to_onnx(
    model: "nn.Module",
    input_dim: int,
    output_path: Path,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export PyTorch neural network to ONNX format.

    Args:
        model: Trained PyTorch model.
        input_dim: Input feature dimension.
        output_path: Path for saving ONNX model.
        opset_version: ONNX opset version.
        dynamic_batch: Enable dynamic batch size.

    Returns:
        Path to exported ONNX model.

    Raises:
        ImportError: If PyTorch or ONNX not available.
        RuntimeError: If export or validation fails.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install: pip install torch")
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX required. Install: pip install onnx")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set model to eval mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, input_dim)

    # Configure dynamic axes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "features": {0: "batch_size"},
            "probabilities": {0: "batch_size"},
        }

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["probabilities"],
        dynamic_axes=dynamic_axes,
    )

    # Validate
    validate_onnx_model(output_path)

    return output_path


def validate_onnx_model(model_path: Path) -> bool:
    """Validate ONNX model structure.

    Args:
        model_path: Path to ONNX model file.

    Returns:
        True if validation passes.

    Raises:
        RuntimeError: If validation fails.
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX required. Install: pip install onnx")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model)
        return True
    except Exception as e:
        raise RuntimeError(f"ONNX validation failed: {e}") from e


def export_sklearn_to_onnx(
    model,
    input_dim: int,
    output_path: Path,
    model_type: str = "classifier",
) -> Path:
    """Export scikit-learn model to ONNX format.

    Args:
        model: Trained sklearn model.
        input_dim: Input feature dimension.
        output_path: Path for saving ONNX model.
        model_type: 'classifier' or 'regressor'.

    Returns:
        Path to exported ONNX model.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        raise ImportError(
            "skl2onnx required. Install: pip install skl2onnx"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define input type
    initial_type = [("features", FloatTensorType([None, input_dim]))]

    # Convert
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Validate
    validate_onnx_model(output_path)

    return output_path


def export_sklearn_pipeline_to_onnx(
    scaler,
    model,
    input_dim: int,
    output_path: Path,
) -> Path:
    """Export a scaler + sklearn model pipeline to ONNX."""
    try:
        from sklearn.pipeline import Pipeline
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        raise ImportError(
            "skl2onnx and scikit-learn required. Install: pip install skl2onnx scikit-learn"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline([("scaler", scaler), ("model", model)])
    initial_type = [("features", FloatTensorType([None, input_dim]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    validate_onnx_model(output_path)
    return output_path


def export_ensemble_to_onnx(
    models: Dict[str, object],
    scaler,
    input_dim: int,
    output_dir: Path,
    model_names: Optional[list] = None,
    require_exportable: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Export supported ensemble components to ONNX pipelines."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    names = model_names or list(models.keys())

    for name in names:
        if name == "faiss_knn":
            skipped[name] = "faiss_knn is not ONNX-exportable"
            continue

        model = models.get(name)
        if model is None:
            skipped[name] = "model not found"
            continue

        try:
            out_path = output_dir / f"{name}.onnx"
            export_sklearn_pipeline_to_onnx(scaler, model, input_dim, out_path)
            exported[name] = str(out_path)
        except Exception as exc:
            skipped[name] = str(exc)
            if require_exportable:
                raise

    return {"exported": exported, "skipped": skipped}


def get_onnx_metadata(model_path: Path) -> dict:
    """Extract metadata from ONNX model.

    Args:
        model_path: Path to ONNX model.

    Returns:
        Dictionary with model metadata.
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX required. Install: pip install onnx")

    onnx_model = onnx.load(str(model_path))

    metadata = {
        "ir_version": onnx_model.ir_version,
        "opset_version": onnx_model.opset_import[0].version,
        "producer_name": onnx_model.producer_name,
        "producer_version": onnx_model.producer_version,
        "domain": onnx_model.domain,
        "model_version": onnx_model.model_version,
    }

    # Input/output info
    metadata["inputs"] = [
        {"name": inp.name, "shape": [d.dim_value for d in inp.type.tensor_type.shape.dim]}
        for inp in onnx_model.graph.input
    ]
    metadata["outputs"] = [
        {"name": out.name, "shape": [d.dim_value for d in out.type.tensor_type.shape.dim]}
        for out in onnx_model.graph.output
    ]

    return metadata
