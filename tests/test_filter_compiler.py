from qdrant_client import models

from apps.search.filters import QdrantFilterCompiler


def test_filter_compiler_builds_nested_filters_and_org_exclusion():
    compiler = QdrantFilterCompiler()

    compiled = compiler.compile(
        hard_filters={
            "degree_slct_nm": "박사",
            "scie_cnt_min": 1,
            "art_recent_years": 5,
            "art_sci_slct_nm": "SCIE",
            "pjt_recent_years": 5,
        },
        exclude_orgs=["A기관"],
    )

    assert compiled is not None
    assert compiled.must is not None
    assert compiled.must_not is not None
    assert any(isinstance(item, models.NestedCondition) for item in compiled.must)
    assert any(getattr(item, "key", None) == "researcher_profile.highest_degree" for item in compiled.must if hasattr(item, "key"))
    assert any(getattr(item, "key", None) == "basic_info.affiliated_organization_exact" for item in compiled.must_not if hasattr(item, "key"))
