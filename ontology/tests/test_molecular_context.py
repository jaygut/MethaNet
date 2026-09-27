from mvo.molecular_context import finite_number, candidate_chambers


def test_missing_not_zero_and_negative_flux_not_clipped():
    assert finite_number("0") == "0"
    assert finite_number("-1.25") == "-1.25"
    for missing in (None,"","NA","nan","-9999","Inf"):
        assert finite_number(missing) is None


def test_same_day_patch_is_only_candidate_with_known_day():
    flux=[{"source_date":"2018-08-07","location_code":"M1","flux_observation_id":"f"}]
    assert candidate_chambers({"collection_date":"2018-08-07","site_id":"M1"},flux)==["f"]
    assert candidate_chambers({"collection_date":"2018","site_id":"M1"},flux)==[]
    assert candidate_chambers({"collection_date":"2018-08-07","site_id":"Mud"},flux)==[]
