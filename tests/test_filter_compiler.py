"""WO-C C4: v2.0 hard_filters → Qdrant 필터 컴파일 검증.

DATA_CONTRACT §1.1 허용 키(highest_degree / recent_years+recent_doc_types /
*_count_min / journal_class)를 researcher_meta·event_year·domain_attrs 경로로 매핑.
다중 doc_type recency는 OR(min_should, min_count=1) — 0건 회귀 방지(필수).
"""
from qdrant_client import models

from apps.search.filters import QdrantFilterCompiler


def _field_keys(conditions):
    return {getattr(c, "key", None) for c in (conditions or []) if hasattr(c, "key")}


def test_compiles_meta_count_degree_journal_and_org_exclusion():
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
    assert "researcher_meta.highest_degree" in must_keys
    assert "researcher_meta.scie_publication_count" in must_keys
    assert "researcher_meta.research_project_count" in must_keys
    assert "domain_attrs.journal_class" in must_keys
    assert "researcher_meta.affiliated_organization" in must_keys  # include
    # exclude → must_not, researcher_meta.affiliated_organization
    assert "researcher_meta.affiliated_organization" in _field_keys(compiled.must_not)
    # v1.x nested / basic_info 경로 잔재 없음
    assert not any(isinstance(c, models.NestedCondition) for c in compiled.must)


def test_multi_doc_type_recency_uses_or_min_should():
    """회귀 방지: 여러 doc_type recency는 AND가 아니라 OR(min_should, min_count=1)."""
    compiled = QdrantFilterCompiler().compile(
        hard_filters={
            "recent_years": 5,
            "recent_doc_types": ["publication", "research_project", "intellectual_property"],
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
    # 각 조건은 doc_type + event_year must
    for cond in or_filter.min_should.conditions:
        assert isinstance(cond, models.Filter)
        keys = _field_keys(cond.must)
        assert keys == {"doc_type", "event_year"}


def test_family_recent_doc_type_expands_to_member_doc_types():
    """recent_doc_types에 family명을 주면 그 family의 doc_type 전체로 확장된다."""
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"recent_years": 3, "recent_doc_types": ["achievement"]},
        exclude_orgs=[],
    )
    or_filter = compiled.must[0]
    assert isinstance(or_filter, models.Filter)
    # achievement = publication / intellectual_property / research_project (3종)
    assert len(or_filter.min_should.conditions) == 3
    doc_type_values = set()
    for cond in or_filter.min_should.conditions:
        for fc in cond.must:
            if getattr(fc, "key", None) == "doc_type":
                doc_type_values.add(fc.match.value)
    assert doc_type_values == {"publication", "intellectual_property", "research_project"}


def test_single_recent_doc_type_not_wrapped_in_or():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"recent_years": 3, "recent_doc_types": ["research_project"]},
        exclude_orgs=[],
    )
    assert compiled is not None and len(compiled.must) == 1
    cond = compiled.must[0]
    assert isinstance(cond, models.Filter)
    assert cond.min_should is None  # OR로 감싸지 않음
    assert _field_keys(cond.must) == {"doc_type", "event_year"}


def test_recent_years_without_doc_types_is_single_event_year_condition():
    compiled = QdrantFilterCompiler().compile(
        hard_filters={"recent_years": 2},
        exclude_orgs=[],
    )
    assert compiled is not None and len(compiled.must) == 1
    cond = compiled.must[0]
    assert isinstance(cond, models.FieldCondition)
    assert cond.key == "event_year"


def test_empty_filters_returns_none():
    assert QdrantFilterCompiler().compile(hard_filters={}, exclude_orgs=[]) is None
