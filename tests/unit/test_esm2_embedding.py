"""Tests for ESM2 embedding layer resolution behavior."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_esm2_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "src" / "methanet" / "embedding" / "esm2.py"
    spec = spec_from_file_location("methanet.embedding.esm2", module_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_esm2_module = _load_esm2_module()
EmbeddingConfig = _esm2_module.EmbeddingConfig
resolve_pooling_layers = _esm2_module.resolve_pooling_layers


def test_embedding_config_defaults_to_final_layer_mean_pooling() -> None:
    config = EmbeddingConfig()
    assert config.model_name == "facebook/esm2_t33_650M_UR50D"
    assert config.pooling_layers == (33,)
    assert config.pooling_strategy == "mean"


def test_resolve_pooling_layers_keeps_valid_indices() -> None:
    resolved = resolve_pooling_layers((20, 21, 22), total_hidden_states=34)
    assert resolved == (20, 21, 22)


def test_resolve_pooling_layers_supports_negative_indices() -> None:
    resolved = resolve_pooling_layers((-1, -2), total_hidden_states=13)
    assert resolved == (12, 11)


def test_resolve_pooling_layers_falls_back_when_out_of_range() -> None:
    resolved = resolve_pooling_layers((20, 21, 22), total_hidden_states=13)
    assert resolved == tuple(range(1, 13))
