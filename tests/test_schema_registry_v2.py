"""WO-C C2: v2.0 schema_registry 상수(단일 벡터명·인덱스 세트·family) 검증 (additive)."""
from __future__ import annotations

from apps.search import schema_registry as sr
from apps.search.doc_types import DOC_TYPES


def test_single_named_vectors():
    assert sr.DENSE_VECTOR_NAME == "dense_e5i"
    assert sr.SPARSE_VECTOR_NAME == "sparse_splade"


def test_doc_types_and_families_reexported():
    assert sr.DOC_TYPES_V2 == DOC_TYPES
    assert set(sr.FAMILIES) == {"identity", "achievement", "assessment", "expertise"}
    assert set(sr.DOC_TYPE_TO_FAMILY_V2) == set(DOC_TYPES)


def test_payload_index_fields_v2_shape():
    paths = [path for path, _ in sr.PAYLOAD_INDEX_FIELDS_V2]
    # 핵심 인덱스 존재
    assert ("researcher_id", "keyword") in sr.PAYLOAD_INDEX_FIELDS_V2
    assert ("doc_type", "keyword") in sr.PAYLOAD_INDEX_FIELDS_V2
    assert ("event_year", "integer") in sr.PAYLOAD_INDEX_FIELDS_V2
    assert ("event_date", "datetime") in sr.PAYLOAD_INDEX_FIELDS_V2
    # 8 researcher_meta count 전부 integer 인덱스
    for count in (
        "publication_count",
        "scie_publication_count",
        "intellectual_property_count",
        "research_project_count",
        "researcher_assessor_count",
        "expert_assessor_count",
    ):
        assert (f"researcher_meta.{count}", "integer") in sr.PAYLOAD_INDEX_FIELDS_V2
    # nested(v1.x) 잔재 없음
    assert not any("[]" in path for path in paths)
    # 인덱스 대상은 모두 필터 가능 키 집합에 포함
    assert set(paths) <= sr.FILTERABLE_FIELDS_V2


def test_v1x_constants_still_present_non_destructive():
    # C3에서 제거될 때까지 v1.x 상수 보존
    assert sr.BRANCHES == ("basic", "art", "pat", "pjt")
    assert "basic_vector_e5i" in sr.DENSE_VECTOR_BY_BRANCH.values()
