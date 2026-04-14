from apps.search.seed_data import build_canonical_payload_fixture


def test_canonical_payload_fixture_maps_project_dates_correctly():
    payload = build_canonical_payload_fixture()

    assert payload.basic_info.researcher_id == "11008395"
    assert payload.research_projects
    assert payload.research_projects[0].project_start_date == "2019-10-07"
    assert payload.research_projects[0].project_end_date == "2020-04-06"
    assert payload.research_projects[0].reference_year == 2019
    assert payload.basic_info.affiliated_organization_exact == "미소테크".upper()
