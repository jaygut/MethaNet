from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "scripts/reports/build_atlas_metadata_readiness.py"
    spec = importlib.util.spec_from_file_location("atlas_metadata_readiness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missingness_and_identifier_parsing_are_explicit():
    module = load_module()
    assert module.present("0")
    assert not module.present("NA")
    assert module.split_ids("A;B, C") == ["A", "B", "C"]
    assert not module.truthy("missing")


def test_process_projection_maps_typed_tower_value_time_and_unit():
    module = load_module()
    tower = {
        "flux_observation_id": "tower_1",
        "methane_flux_nmol_m2_s": "264.9941912",
        "source_datetime_start_local_or_timezone_unknown": "2015-06-01T00:00:00",
        "source_datetime_end_local_or_timezone_unknown": "2015-06-01T00:30:00",
        "source_datetime_timezone_status": "source_timestamp_unzoned_do_not_convert_or_infer_utc",
        "source_value_status": "reported_valid",
        "measurement_approach": "gap_filled_eddy_covariance",
        "temporal_resolution": "half_hourly",
        "sample_join_status": "unlinked_no_authoritative_sequence_sample_crosswalk",
        "site_code": "US-OWC",
    }
    rows = module.build_process_observation_rows([], [], [tower])
    assert len(rows) == 1
    assert rows[0]["value"] == "264.9941912"
    assert rows[0]["value_unit"] == "nmol m-2 s-1"
    assert rows[0]["source_date"] == "2015-06-01"
    assert rows[0]["source_datetime_start_local_or_timezone_unknown"] == "2015-06-01T00:00:00"
    assert rows[0]["source_datetime_timezone_status"] == "source_timestamp_unzoned_do_not_convert_or_infer_utc"

    broken = dict(tower, methane_flux_nmol_m2_s="")
    with pytest.raises(ValueError, match="valid flux value"):
        module.build_process_observation_rows([], [], [broken])
