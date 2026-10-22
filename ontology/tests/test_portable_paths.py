"""Portable manifests name the repository root as $METHANET_ROOT."""
from mvo.ingest import resolve
from mvo.molecular_audit import local_path


def test_root_token_resolves_inside_the_repository(tmp_path):
    repo = tmp_path.resolve()
    target = repo / "results" / "dim_mag.parquet"
    target.parent.mkdir()
    target.write_bytes(b"")
    for value in ("$METHANET_ROOT/results/dim_mag.parquet", "${METHANET_ROOT}/results/dim_mag.parquet"):
        assert resolve(repo, value) == target
        assert local_path(repo, value) == target
    assert resolve(repo, "results/dim_mag.parquet") == target
