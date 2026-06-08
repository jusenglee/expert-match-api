"""flat 계약(v2.1) hard_filters → Qdrant 필터 컴파일 검증.

DATA_CONTRACT §1.1 허용 키(highest_degree / recent_years+recent_doc_types /
*_count_min)를 flat root 경로로 매핑한다:
- highest_degree → root highest_degree
- *_count_min → root count (assessor 별칭 전부 → researcher_assessor_activity_count)
- recency → root doc_date(DatetimeRange) 기준, doc_type별 OR(min_should, min_count=1)

doc_attrs.*는 유동 필드이므로 hard filter로 컴파일하지 않는다.
기관 include/exclude는 정규화된 root 필드가 없으므로 retriever 앱단 post-filter 소관이다.
다중 doc_type recency는 AND가 아니라 OR(min_should, min_count=1) — 0건 회귀 방지(필수).
"""
from datetime import UTC, datetime

from qdrant_client import models

from apps.search.filters import QdrantFilterCompiler


def _field_keys(conditions):
    return {getattr(c, "key", None) for c in (conditions or []) if hasattr(c, "key")}


def _expected_cutoff(recent_years):
    # 소스는 RFC3339 문자열 "<year>-01-01T00:00:00Z"를 만들지만 qdrant pydantic 모델이
    # DatetimeRange.gte를 tz-aware datetime으로 정규화하므로 동일 datetime으로 비교한다.
    year = datetime.now(UTC).year - recent_years
    return datetime(year, 1, 1, tzinfo=UTC)


def test_compiles_root_meta_count_degree_and_ignores_doc_attrs_and_orgs():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={
            "highest_degree": "박사",
            "scie_publication_count_min": 1,
            "research_project_count_min": 3,
            "journal_class": "SCIE",
        },
        exclude_orgs=["A기관"],
        include_orgs=["B기관"],
    )

    assert compiled is not None
    must_keys = _field_keys(compiled.must)
    # flat root 키
    assert "highest_degree" in must_keys
    assert "scie_publication_count" in must_keys
    assert "research_project_count" in must_keys
    # doc_attrs.*와 기관 조건은 Qdrant hard filter로 컴파일하지 않는다.
    assert "doc_attrs.indexing_database" not in must_keys
    assert "affiliated_organization" not in must_keys
    assert compiled.must_not is None
    # v1.x nested 경로 잔재 없음
    assert not any(isinstance(c, models.NestedCondition) for c in compiled.must)
    assert not any("researcher_meta" in (k or "") for k in must_keys)
    assert not any("domain_attrs" in (k or "") for k in must_keys)


def test_highest_degree_uses_match_any():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"highest_degree": ["박사", "석사"]},
        exclude_orgs=[],
    )
    assert compiled is not None
    degree_conds = [
        c for c in compiled.must if getattr(c, "key", None) == "highest_degree"
    ]
    assert len(degree_conds) == 1
    assert isinstance(degree_conds[0].match, models.MatchAny)
    assert degree_conds[0].match.any == ["박사", "석사"]


def test_count_min_uses_range_gte():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"publication_count_min": 5},
        exclude_orgs=[],
    )
    assert compiled is not None
    pub_conds = [
        c for c in compiled.must if getattr(c, "key", None) == "publication_count"
    ]
    assert len(pub_conds) == 1
    assert isinstance(pub_conds[0].range, models.Range)
    assert pub_conds[0].range.gte == 5


def test_assessor_count_aliases_all_map_to_single_field():
    """구 split 별칭(researcher_assessor_count_min / expert_assessor_count_min /
    assessor_activity_count_min)은 모두 단일 researcher_assessor_activity_count로 흡수된다."""
    for alias in (
        "researcher_assessor_activity_count_min",
        "assessor_activity_count_min",
        "researcher_assessor_count_min",
        "expert_assessor_count_min",
    ):
        compiled = QdrantFilterCompiler().compile(
            hard_filters={alias: 2},
            exclude_orgs=[],
        )
        assert compiled is not None, alias
        keys = _field_keys(compiled.must)
        assert keys == {"researcher_assessor_activity_count"}, alias


def test_multi_doc_type_recency_uses_or_min_should():
    """회귀 방지: 여러 doc_type recency는 AND가 아니라 OR(min_should, min_count=1)."""
    compiled = QdrantFilterCompiler().compile(
        hard_filters={
            "recent_years": 5,
            "recent_doc_types": ["paper", "project", "patent"],
        },
        exclude_orgs=[],
    )

    assert compiled is not None and compiled.must is not None
    # 단일 OR 묶음만 must에 존재
    assert len(compiled.must) == 1
    or_filter = compiled.must[0]
    assert isinstance(or_filter, models.Filter)
    assert or_filter.min_should is not None
    assert or_filter.min_should.min_count == 1
    assert len(or_filter.min_should.conditions) == 3
    cutoff = _expected_cutoff(5)
    for cond in or_filter.min_should.conditions:
        assert isinstance(cond, models.Filter)
        keys = _field_keys(cond.must)
        # 각 조건은 doc_type + doc_date must
        assert keys == {"doc_type", "doc_date"}
        for fc in cond.must:
            if fc.key == "doc_date":
                assert isinstance(fc.range, models.DatetimeRange)
                assert fc.range.gte == cutoff


def test_family_recent_doc_type_expands_to_member_doc_types():
    """recent_doc_types에 family명을 주면 그 family의 doc_type 전체로 확장된다."""
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"recent_years": 3, "recent_doc_types": ["achievement"]},
        exclude_orgs=[],
    )
    assert compiled is not None and len(compiled.must) == 1
    or_filter = compiled.must[0]
    assert isinstance(or_filter, models.Filter)
    # achievement = paper / patent / project (3종)
    assert len(or_filter.min_should.conditions) == 3
    doc_type_values = set()
    for cond in or_filter.min_should.conditions:
        for fc in cond.must:
            if getattr(fc, "key", None) == "doc_type":
                doc_type_values.add(fc.match.value)
    assert doc_type_values == {"paper", "patent", "project"}


def test_assessment_family_recent_doc_type_expands_to_assessor_activity():
    """assessment family는 단일 멤버(assessor_activity)로 확장 → OR로 감싸지 않는다."""
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"recent_years": 4, "recent_doc_types": ["assessment"]},
        exclude_orgs=[],
    )
    assert compiled is not None and len(compiled.must) == 1
    cond = compiled.must[0]
    assert isinstance(cond, models.Filter)
    assert cond.min_should is None  # 단일 멤버 → OR로 감싸지 않음
    assert _field_keys(cond.must) == {"doc_type", "doc_date"}
    doc_type_conds = [c for c in cond.must if getattr(c, "key", None) == "doc_type"]
    assert doc_type_conds[0].match.value == "assessor_activity"


def test_single_recent_doc_type_not_wrapped_in_or():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"recent_years": 3, "recent_doc_types": ["project"]},
        exclude_orgs=[],
    )
    assert compiled is not None and len(compiled.must) == 1
    cond = compiled.must[0]
    assert isinstance(cond, models.Filter)
    assert cond.min_should is None  # OR로 감싸지 않음
    assert _field_keys(cond.must) == {"doc_type", "doc_date"}
    for fc in cond.must:
        if fc.key == "doc_date":
            assert isinstance(fc.range, models.DatetimeRange)
            assert fc.range.gte == _expected_cutoff(3)


def test_recent_years_without_doc_types_is_single_doc_date_condition():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"recent_years": 2},
        exclude_orgs=[],
    )
    assert compiled is not None and len(compiled.must) == 1
    cond = compiled.must[0]
    assert isinstance(cond, models.FieldCondition)
    assert cond.key == "doc_date"
    assert isinstance(cond.range, models.DatetimeRange)
    assert cond.range.gte == _expected_cutoff(2)


def test_unknown_recent_doc_type_is_ignored_falls_back_to_bare_doc_date():
    """알 수 없는 doc_type 항목은 무시되어 확장 대상이 0개 → bare doc_date 조건."""
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"recent_years": 1, "recent_doc_types": ["nonexistent"]},
        exclude_orgs=[],
    )
    assert compiled is not None and len(compiled.must) == 1
    cond = compiled.must[0]
    assert isinstance(cond, models.FieldCondition)
    assert cond.key == "doc_date"


def test_org_only_filters_are_left_to_retriever_post_filter():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={},
        exclude_orgs=["제외기관"],
        include_orgs=["포함기관"],
    )
    assert compiled is None


def test_empty_filters_returns_none():
    assert QdrantFilterCompiler().compile(hard_filters={}, exclude_orgs=[]) is None
