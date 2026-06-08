"""Qdrant 컬렉션 스키마 단일 출처 (flat chunk 모델, v2.1).

[Architecture]
컬렉션 1개 / 1 chunk = 1 Point / 단일 dense + 단일 sparse named vector / flat payload.
doc_type별 named vector를 두지 않는다 — doc_type은 **payload 필터**로 분기한다(DATA_MODEL §2).
연구자 후보는 검색 시점에 researcher_id로 집계한다.
"""

from __future__ import annotations

# 단일 출처: doc_type 5종 enum + 4 family 매핑(아래에서 재노출).
from apps.search.doc_types import DOC_TYPE_TO_FAMILY, DOC_TYPES, Family

# =========================================================================
# 단일 named vector (실제 라이브 컬렉션 기준)
# =========================================================================
# 입력 텍스트는 둘 다 chunk_text. doc_type은 payload 필터로 분기.
DENSE_VECTOR_NAME = "vector_e5i"      # dense, multilingual-e5-large-instruct (1024, Cosine)
SPARSE_VECTOR_NAME = "vector_splade"  # sparse, PIXIE-Splade-v1.0

DENSE_VECTOR_SIZE = 1024
DENSE_DISTANCE = "Cosine"

# doc_type/family 재노출(검색 코드가 schema_registry 한 곳에서 참조하도록).
DOC_TYPE_TO_FAMILY_MAP: dict[str, str] = dict(DOC_TYPE_TO_FAMILY)
FAMILIES: tuple[str, ...] = tuple(f.value for f in Family)

# =========================================================================
# flat payload 필터/인덱스 (DATA_MODEL §5)
# =========================================================================
# 필터(FieldCondition) 가능한 키. 실데이터 계약상 안정적인 flat root 필드만 포함한다.
# doc_attrs.*는 doc_type별 유동 필드이므로 필터/인덱스 대상으로 삼지 않는다.
FILTERABLE_FIELDS: frozenset[str] = frozenset(
    {
        # 공통 식별/메타 (flat root)
        "researcher_id",
        "doc_type",
        "doc_date",
        "affiliated_organization",
        "highest_degree",
        "publication_count",
        "scie_publication_count",
        "intellectual_property_count",
        "research_project_count",
        "researcher_assessor_activity_count",
    }
)

# Qdrant payload 인덱스 세트. ★부트스트랩이 실제 생성하는 인덱스와 1:1 정합해야 한다.
PAYLOAD_INDEX_FIELDS: tuple[tuple[str, str], ...] = (
    # keyword
    ("researcher_id", "keyword"),
    ("doc_type", "keyword"),
    ("affiliated_organization", "keyword"),
    ("highest_degree", "keyword"),
    # integer (연구자 공통 count — 모든 chunk에 비정규화)
    ("publication_count", "integer"),
    ("scie_publication_count", "integer"),
    ("intellectual_property_count", "integer"),
    ("research_project_count", "integer"),
    ("researcher_assessor_activity_count", "integer"),
    # datetime (recency). 'NONE'/결측은 datetime range에 매칭되지 않음(= recency 제외).
    ("doc_date", "datetime"),
)

__all__ = [
    "DENSE_VECTOR_NAME",
    "SPARSE_VECTOR_NAME",
    "DENSE_VECTOR_SIZE",
    "DENSE_DISTANCE",
    "DOC_TYPES",
    "DOC_TYPE_TO_FAMILY_MAP",
    "FAMILIES",
    "FILTERABLE_FIELDS",
    "PAYLOAD_INDEX_FIELDS",
]
