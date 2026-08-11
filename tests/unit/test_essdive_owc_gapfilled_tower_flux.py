from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


def _load_script() -> object:
    repo_root = Path(__file__).resolve().parents[2]
    spec = spec_from_file_location(
        "stage_essdive_owc_gapfilled_tower_flux",
        repo_root / "scripts/external/stage_essdive_owc_gapfilled_tower_flux.py",
    )
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gapfilled_tower_flux_keeps_unlinked_site_time_grain(tmp_path: Path) -> None:
    module = _load_script()
    flux_path = tmp_path / "US_OWC_CH4_CO2_LE.csv"
    pd.DataFrame(
        [
            {
                "Location": "US-OWC",
                "TIMESTAMP_START": "201506010000",
                "TIMESTAMP_END": "201506010030",
                "NEE_F": "-2.5",
                "RE": "3.0",
                "GPP": "5.5",
                "FCH4_F": "264.994",
                "LE_F": "12.3",
            },
            {
                "Location": "US-OWC",
                "TIMESTAMP_START": "201506010030",
                "TIMESTAMP_END": "201506010100",
                "NEE_F": "",
                "RE": "",
                "GPP": "",
                "FCH4_F": "-9999",
                "LE_F": "",
            },
        ]
    ).to_csv(flux_path, index=False)

    table = module.gapfilled_tower_flux_table(
        flux_path,
        {
            "site_code": "US-OWC",
            "latitude": 41.37927778,
            "longitude": -82.51233333,
            "elevation_m": 174.0,
            "site_description": "Freshwater estuarine wetland",
        },
    )

    assert len(table) == 2
    assert table.loc[0, "methane_flux_nmol_m2_s"] == 264.994
    assert table.loc[0, "measurement_approach"] == "gap_filled_eddy_covariance"
    assert table.loc[0, "source_datetime_timezone_status"] == (
        "source_timestamp_unzoned_do_not_convert_or_infer_utc"
    )
    assert table.loc[1, "source_value_status"] == "source_missing_sentinel"
    assert set(table["sample_join_status"]) == {
        "unlinked_no_authoritative_sequence_sample_crosswalk"
    }
    assert set(table["ecological_resolution_tier"]) == {
        "site_tower_datetime_flux_observation"
    }


def test_parse_location_table_skips_essdive_metadata_rows(tmp_path: Path) -> None:
    module = _load_script()
    location_path = tmp_path / "US_OWC_locations.csv"
    location_path.write_text(
        "Submission_Contact_Name,Submission_Contact_Email,Location_ID,Description,Latitude,Longitude,Elevation\n"
        "Free text,Free text,Free text,Free text,numeric,numeric,numeric\n"
        "N/A,N/A,N/A,N/A,decimal degrees,decimal degrees,meters\n"
        "A. Tang,a@example.org,US-OWC,Freshwater wetland,41.37927778,-82.51233333,174\n"
    )

    location = module.parse_location_table(location_path)

    assert location["site_code"] == "US-OWC"
    assert location["latitude"] == 41.37927778
    assert location["longitude"] == -82.51233333
