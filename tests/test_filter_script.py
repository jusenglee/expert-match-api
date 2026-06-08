"""flat 계약(v2.1) QdrantFilterCompiler 검증.

원본 test_filter_script.py는 라이브 Qdrant에 붙어 org 배제를 수동 확인하던 스크립트였다
(SearchSchemaRegistry / payload.basic_info 등 폐기 API 사용). 현재 계약에서는 org include/exclude가
정규화된 root 필드 부재로 Qdrant exact pre-filter가 아니라 retriever 앱단 post-filter 소관이다.

test_filter_compiler.py와 중복을 피하기 위해 여기서는 org include/exclude Qdrant 필터 제외,
*_count_min 하위호환 별칭, doc_attrs 필터 무시, doc_date(datetime)
recency 컷오프 값 등 보완 커버리지에 집중한다.
"""
from __future__ import annotations

from datetime import UTC, datetime

from qdrant_client import models

from apps.search.doc_types import DOC_TYPE_TO_FAMILY, DOC_TYPES, Family
from apps.search.filters import QdrantFilterCompiler


def _field_conditions(conditions):
    return [c for c in (conditions or []) if isinstance(c, models.FieldCondition)]


def _field_keys(conditions):
    return {c.key for c in _field_conditions(conditions)}


def _condition_for(conditions, key):
    for c in _field_conditions(conditions):
        if c.key == key:
            return c
    return None


# ---------------------------------------------------------------------------
# org include / exclude - retriever 앱단 post-filter 소관
# ---------------------------------------------------------------------------


def test_exclude_org_is_not_compiled_to_qdrant_filter():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={}, exclude_orgs=["주식회사 미소테크"], include_orgs=[]
    )

    assert compiled is None


def test_include_org_is_not_compiled_to_qdrant_filter():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={}, exclude_orgs=[], include_orgs=["한국전자통신연구원"]
    )

    assert compiled is None


def test_include_and_exclude_with_root_filter_keeps_orgs_out_of_qdrant_filter():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"highest_degree": "박사"},
        exclude_orgs=["배제기관"],
        include_orgs=["포함기관"],
    )
    assert compiled is not None
    assert _condition_for(compiled.must, "highest_degree") is not None
    assert _condition_for(compiled.must, "affiliated_organization") is None
    assert compiled.must_not is None


def test_blank_or_unnormalizable_org_is_dropped():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={}, exclude_orgs=["주식회사", "()"], include_orgs=["   "]
    )
    assert compiled is None


# ---------------------------------------------------------------------------
# *_count_min — flat root count >= N (하위호환 별칭 포함)
# ---------------------------------------------------------------------------


def test_count_min_keys_map_to_flat_root_count_fields():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={
            "publication_count_min": 5,
            "scie_publication_count_min": 1,
            "intellectual_property_count_min": 2,
            "research_project_count_min": 3,
            "researcher_assessor_activity_count_min": 4,
        },
        exclude_orgs=[],
    )
    assert compiled is not None
    keys = _field_keys(compiled.must)
    assert {
        "publication_count",
        "scie_publication_count",
        "intellectual_property_count",
        "research_project_count",
        "researcher_assessor_activity_count",
    } <= keys
    # 값이 Range(gte=...)로 들어가는지
    cond = _condition_for(compiled.must, "publication_count")
    assert isinstance(cond.range, models.Range)
    assert cond.range.gte == 5


def test_legacy_assessor_count_aliases_collapse_to_single_field():
    """구 split 키들은 단일 researcher_assessor_activity_count로 흡수된다(하위호환)."""
    for alias in (
        "assessor_activity_count_min",
        "researcher_assessor_count_min",
        "expert_assessor_count_min",
    ):
        compiled = QdrantFilterCompiler().compile(
            hard_filters={alias: 7}, exclude_orgs=[]
        )
        assert compiled is not None, alias
        cond = _condition_for(compiled.must, "researcher_assessor_activity_count")
        assert cond is not None, f"{alias} 은 단일 assessor count로 매핑되어야 한다"
        assert cond.range.gte == 7


def test_count_min_zero_is_emitted_but_none_is_dropped():
    # 0은 유효한 하한(>=0) → 조건 생성; 키 미존재(None)는 무시
    compiled_zero = QdrantFilterCompiler().compile(
        hard_filters={"publication_count_min": 0}, exclude_orgs=[]
    )
    assert compiled_zero is not None
    assert _condition_for(compiled_zero.must, "publication_count").range.gte == 0

    compiled_none = QdrantFilterCompiler().compile(
        hard_filters={"publication_count_min": None}, exclude_orgs=[]
    )
    assert compiled_none is None


# ---------------------------------------------------------------------------
# journal_class/doc_attrs - 유동 필드이므로 hard filter에서 무시
# ---------------------------------------------------------------------------


def test_journal_class_is_ignored_because_doc_attrs_are_dynamic():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"journal_class": "SCIE"}, exclude_orgs=[]
    )
    assert compiled is None


def test_journal_class_list_is_ignored_because_doc_attrs_are_dynamic():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"journal_class": ["SCIE", "SCOPUS"]}, exclude_orgs=[]
    )
    assert compiled is None


# ---------------------------------------------------------------------------
# highest_degree — root, MatchAny
# ---------------------------------------------------------------------------


def test_highest_degree_scalar_and_list_use_root_match_any():
    scalar = QdrantFilterCompiler().compile(
        hard_filters={"highest_degree": "박사"}, exclude_orgs=[]
    )
    cond = _condition_for(scalar.must, "highest_degree")
    assert cond is not None and cond.match.any == ["박사"]

    listed = QdrantFilterCompiler().compile(
        hard_filters={"highest_degree": ["박사", "석사"]}, exclude_orgs=[]
    )
    cond2 = _condition_for(listed.must, "highest_degree")
    assert cond2.match.any == ["박사", "석사"]
    # 중첩/메타 경로 잔재 없음
    assert "researcher_meta.highest_degree" not in _field_keys(scalar.must)


# ---------------------------------------------------------------------------
# recency — root doc_date(datetime), 다중 doc_type은 OR(min_should)
# ---------------------------------------------------------------------------


def test_recency_without_doc_types_is_single_doc_date_datetime_range():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"recent_years": 3}, exclude_orgs=[]
    )
    assert compiled is not None and len(compiled.must) == 1
    cond = compiled.must[0]
    assert isinstance(cond, models.FieldCondition)
    assert cond.key == "doc_date"
    assert isinstance(cond.range, models.DatetimeRange)
    # qdrant DatetimeRange는 RFC3339 문자열을 tz-aware datetime으로 강제 변환한다.
    expected_year = datetime.now(UTC).year - 3
    gte = cond.range.gte
    parsed = datetime.fromisoformat(gte) if isinstance(gte, str) else gte
    assert parsed == datetime(expected_year, 1, 1, tzinfo=UTC)


def test_single_recent_doc_type_not_wrapped_in_or():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"recent_years": 5, "recent_doc_types": ["paper"]},
        exclude_orgs=[],
    )
    assert compiled is not None and len(compiled.must) == 1
    cond = compiled.must[0]
    assert isinstance(cond, models.Filter)
    assert cond.min_should is None  # 단일이면 OR로 감싸지 않음
    assert _field_keys(cond.must) == {"doc_type", "doc_date"}
    doc_type_cond = _condition_for(cond.must, "doc_type")
    assert doc_type_cond.match.value == "paper"
    date_cond = _condition_for(cond.must, "doc_date")
    assert isinstance(date_cond.range, models.DatetimeRange)


def test_multi_doc_type_recency_uses_or_min_should():
    """회귀 방지(필수): 다중 doc_type recency는 AND가 아니라 OR(min_should, min_count=1)."""
    compiled = QdrantFilterCompiler().compile(
        hard_filters={
            "recent_years": 5,
            "recent_doc_types": ["paper", "patent", "project"],
        },
        exclude_orgs=[],
    )
    assert compiled is not None and compiled.must is not None
    assert len(compiled.must) == 1  # 단일 OR 묶음만
    or_filter = compiled.must[0]
    assert isinstance(or_filter, models.Filter)
    assert or_filter.min_should is not None
    assert or_filter.min_should.min_count == 1
    assert len(or_filter.min_should.conditions) == 3
    seen_doc_types = set()
    for cond in or_filter.min_should.conditions:
        assert isinstance(cond, models.Filter)
        assert _field_keys(cond.must) == {"doc_type", "doc_date"}
        seen_doc_types.add(_condition_for(cond.must, "doc_type").match.value)
    assert seen_doc_types == {"paper", "patent", "project"}


def test_family_recent_doc_type_expands_to_member_doc_types():
    """recent_doc_types에 family명(achievement)을 주면 그 family의 doc_type 전체로 확장."""
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"recent_years": 3, "recent_doc_types": ["achievement"]},
        exclude_orgs=[],
    )
    or_filter = compiled.must[0]
    assert isinstance(or_filter, models.Filter)
    members = {
        dt for dt, fam in DOC_TYPE_TO_FAMILY.items() if fam == Family.ACHIEVEMENT
    }
    assert members == {"paper", "patent", "project"}
    assert len(or_filter.min_should.conditions) == len(members)
    seen = set()
    for cond in or_filter.min_should.conditions:
        seen.add(_condition_for(cond.must, "doc_type").match.value)
    assert seen == members


def test_assessment_family_single_member_not_wrapped_in_or():
    """assessment family는 assessor_activity 단일 멤버 → OR로 감싸지 않는다."""
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"recent_years": 2, "recent_doc_types": ["assessment"]},
        exclude_orgs=[],
    )
    assert compiled is not None and len(compiled.must) == 1
    cond = compiled.must[0]
    assert isinstance(cond, models.Filter)
    assert cond.min_should is None
    assert _condition_for(cond.must, "doc_type").match.value == "assessor_activity"


def test_unknown_recent_doc_type_is_ignored_falls_back_to_plain_date():
    """알 수 없는 recent_doc_types 항목은 무시 → 확장 결과가 비면 doc_date 단일 조건."""
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"recent_years": 4, "recent_doc_types": ["bogus_type"]},
        exclude_orgs=[],
    )
    assert compiled is not None and len(compiled.must) == 1
    cond = compiled.must[0]
    assert isinstance(cond, models.FieldCondition)
    assert cond.key == "doc_date"


# ---------------------------------------------------------------------------
# 통합 + 빈 입력
# ---------------------------------------------------------------------------


def test_combined_filters_all_use_flat_keys_no_nested_residue():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={
            "highest_degree": "박사",
            "scie_publication_count_min": 1,
            "research_project_count_min": 3,
            "journal_class": "SCIE",
            "recent_years": 5,
            "recent_doc_types": ["paper", "patent"],
        },
        exclude_orgs=["배제기관"],
        include_orgs=["포함기관"],
    )
    assert compiled is not None
    must_keys = _field_keys(compiled.must)
    assert {
        "highest_degree",
        "scie_publication_count",
        "research_project_count",
    } <= must_keys
    assert "doc_attrs.indexing_database" not in must_keys
    assert "affiliated_organization" not in must_keys
    assert compiled.must_not is None
    # 어떤 must/must_not 조건도 v1.x 중첩 경로를 쓰지 않는다
    all_field_keys = _field_keys(compiled.must) | _field_keys(compiled.must_not)
    assert all(not k.startswith("researcher_meta.") for k in all_field_keys)
    assert all(not k.startswith("basic_info.") for k in all_field_keys)
    assert not any(isinstance(c, models.NestedCondition) for c in compiled.must)
    # recency OR 묶음이 must에 정확히 하나 존재
    or_filters = [
        c for c in compiled.must if isinstance(c, models.Filter) and c.min_should
    ]
    assert len(or_filters) == 1


def test_empty_filters_returns_none():
    assert QdrantFilterCompiler().compile(hard_filters={}, exclude_orgs=[]) is None


def test_doc_types_constant_is_exactly_five_flat_types():
    assert set(DOC_TYPES) == {
        "paper",
        "patent",
        "project",
        "assessor_activity",
        "specialty",
    }
