from apps.search.seed_data import build_canonical_payload_fixture


def test_canonical_payload_fixture_maps_project_dates_correctly():
    payload = build_canonical_payload_fixture()

    assert payload.doc_id == "11008395"
    assert payload.pjt
    assert payload.pjt[0].start_dt == "2019-10-07"
    assert payload.pjt[0].end_dt == "2020-04-06"
    assert payload.pjt[0].stan_yr == 2019
    assert payload.blng_org_nm_exact == "주식회사미소테크".upper()
