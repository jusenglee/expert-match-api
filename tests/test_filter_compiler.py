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


def test_activity_continuity_query_uses_or_logic():
    """
    회귀 방지: '활동 지속성' 의도로 art/pat/pjt recent_years 세 조건이 동시 생성될 때
    AND(must) 교집합이 아니라 OR(min_should, min_count=1) 합집합으로 묶여야 한다.
    기존 AND 로직은 결과 0건을 초래했음 (postmortem-activity-filter-empty-result 참조).
    """
    compiler = QdrantFilterCompiler()

    compiled = compiler.compile(
        hard_filters={
            "art_recent_years": 5,
            "pat_recent_years": 5,
            "pjt_recent_years": 5,
        },
        exclude_orgs=[],
    )

    assert compiled is not None
    assert compiled.must is not None

    # must 배열에 단 하나의 원소(OR 묶음)만 있어야 함
    # (기존 버그: 3개의 개별 NestedCondition이 must에 삽입되어 AND 교집합 적용)
    assert len(compiled.must) == 1, (
        "활동 지속성 조건 3개는 must 배열에 개별 삽입(AND)이 아니라 "
        "OR(min_should) 하나로 묶여야 합니다."
    )

    or_filter = compiled.must[0]
    assert isinstance(or_filter, models.Filter), "OR 묶음은 models.Filter 타입이어야 합니다."
    assert or_filter.min_should is not None, "min_should 필드가 설정되어야 합니다."
    assert or_filter.min_should.min_count == 1, "min_count는 1이어야 합니다 (하나 이상 만족)."
    assert len(or_filter.min_should.conditions) == 3, "논문·특허·과제 3개 조건이 모두 포함되어야 합니다."

    # 각 조건이 올바른 Nested 키를 참조하는지 확인
    nested_keys = {
        c.nested.key
        for c in or_filter.min_should.conditions
        if isinstance(c, models.NestedCondition)
    }
    assert nested_keys == {"publications", "intellectual_properties", "research_projects"}, (
        f"예상 nested keys와 다릅니다: {nested_keys}"
    )


def test_activity_continuity_with_scie_qualifier_keeps_must():
    """
    SCIE 한정자가 포함된 논문 조건은 OR 풀이 아닌 독립 must 조건으로 유지되어야 한다.
    """
    compiler = QdrantFilterCompiler()

    compiled = compiler.compile(
        hard_filters={
            "art_recent_years": 3,
            "art_sci_slct_nm": "SCIE",  # 한정자 포함 → must 유지
            "pjt_recent_years": 3,      # 한정자 없음 → OR 풀 합류
        },
        exclude_orgs=[],
    )

    assert compiled is not None
    assert compiled.must is not None

    # art 조건(SCIE 포함)은 독립 must로, pjt 조건은 단독 recent_activity이므로 직접 must에 삽입
    # → 총 2개의 must 원소 (NestedCondition×2, OR 묶음 없음)
    nested_conditions = [c for c in compiled.must if isinstance(c, models.NestedCondition)]
    or_filters = [c for c in compiled.must if isinstance(c, models.Filter)]

    assert len(nested_conditions) == 2, "art(SCIE) + pjt 각각 독립 NestedCondition이어야 합니다."
    assert len(or_filters) == 0, "2개 미만이므로 OR 묶음(Filter)이 생성되지 않아야 합니다."

    nested_keys = {c.nested.key for c in nested_conditions}
    assert nested_keys == {"publications", "research_projects"}


def test_single_recent_years_not_wrapped_in_or():
    """
    단일 recent_years 조건은 OR 묶음 없이 NestedCondition 직접 must 삽입이어야 한다.
    """
    compiler = QdrantFilterCompiler()

    compiled = compiler.compile(
        hard_filters={"pjt_recent_years": 3},
        exclude_orgs=[],
    )

    assert compiled is not None
    assert len(compiled.must) == 1
    assert isinstance(compiled.must[0], models.NestedCondition)
    assert compiled.must[0].nested.key == "research_projects"
