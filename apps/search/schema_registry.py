"""
Qdrant 컬렉션의 스키마와 벡터 매핑 정보를 통합 관리하는 레지스트리 모듈입니다.

[Architecture Overview]
본 시스템은 한 명의 전문가(Researcher) 데이터를 Qdrant 내에 4개의 서로 다른 관점(Branch)으로 분리하여 저장합니다.
각 브랜치는 동일한 Payload(전문가 메타데이터)를 공유하지만, 벡터(Dense/Sparse) 공간은 별도로 구성되어 있어 
"논문 기반 검색", "특허 기반 검색" 등 도메인에 최적화된 독립적인 점수 산출이 가능합니다.
"""

from __future__ import annotations

from dataclasses import dataclass

# WO-0 단일 출처: doc_type 11종 enum + 4 family 매핑(아래 v2.0 섹션에서 재노출).
from apps.search.doc_types import DOC_TYPE_TO_FAMILY, DOC_TYPES, Family


# =========================================================================
# 시스템에서 사용하는 주요 데이터 브랜치 (Branches)
# =========================================================================
# 검색 시 각 브랜치에 대해 개별적인 서브쿼리가 생성되고 병렬로 실행됩니다.
BRANCHES: tuple[str, str, str, str] = (
    "basic",  # [기본 정보] 연구자의 프로필, 소속, 학력 등 텍스트
    "art",    # [논문 실적] SCI(E) 등 논문 초록, 제목, 키워드
    "pat",    # [특허 실적] 국내외 특허 출원/등록 요약 및 기술 내용
    "pjt",    # [국가 R&D 과제] 수행/참여했던 국책 과제 목표, 내용, 기대효과
)

# =========================================================================
# Dense (밀집/의미론적) 벡터 이름 매핑
# =========================================================================
# Qdrant 내부에서 다중 벡터(Multi-vector) 구조를 사용할 때 각각의 이름입니다.
# E5-Instruct 모델 기반으로 1024차원의 의미론적 임베딩을 저장합니다.
DENSE_VECTOR_BY_BRANCH = {
    "basic": "basic_vector_e5i",
    "art": "art_vector_e5i",
    "pat": "pat_vector_e5i",
    "pjt": "pjt_vector_e5i",
}

# =========================================================================
# Sparse (희소/키워드) 벡터 이름 매핑
# =========================================================================
# 최신 아키텍처는 BM25 대신 SPLADE 기반의 희소 벡터를 사용하여 키워드 매칭 성능을 극대화합니다.
# 주의: Qdrant 컬렉션이 `_splade` 접미사를 가지도록 구성되어 있어야 합니다. (config.py 연동)
SPARSE_VECTOR_BY_BRANCH = {
    "basic": "basic_vector_splade",
    "art": "art_vector_splade",
    "pat": "pat_vector_splade",
    "pjt": "pjt_vector_splade",
}

# 레시피 데이터 등에서 날짜 필드 이름 보정이 필요한 경우 사용
DATE_FIELD_CORRECTIONS = {
    "TOT_RSCH_START_DT": "start_dt",
    "TOT_RSCH_END_DT": "end_dt",
    "STAN_YR": "stan_yr",
}

# 전문가 루트 레벨에서 필터링(FieldCondition)이 가능한 필드 목록
FILTERABLE_ROOT_FIELDS = {
    "basic_info.researcher_id",
    "basic_info.affiliated_organization_exact",
    "researcher_profile.highest_degree",
    "researcher_profile.publication_count",
    "researcher_profile.scie_publication_count",
    "researcher_profile.intellectual_property_count",
    "researcher_profile.research_project_count",
}

# 중첩된 데이터 구조(Nested) 내에서 필터링이 가능한 필드 목록
FILTERABLE_NESTED_FIELDS = {
    "publications": {"journal_index_type", "publication_year_month"},
    "intellectual_properties": {"application_registration_type", "application_country", "application_date", "registration_date"},
    "research_projects": {"project_start_date", "project_end_date", "reference_year", "performing_organization", "managing_agency"},
}

# =========================================================================
# Qdrant 페이로드 인덱스 및 필터링 가능 필드 정의
# =========================================================================
# Qdrant에서 Payload 필터링(Where 조건절)을 고속으로 수행하기 위해 사전에 정의된 인덱스 목록입니다.
PAYLOAD_INDEX_FIELDS: tuple[tuple[str, str], ...] = (
    ("basic_info.researcher_id", "keyword"),
    ("basic_info.affiliated_organization_exact", "keyword"),
    ("researcher_profile.highest_degree", "keyword"),
    ("researcher_profile.publication_count", "integer"),
    ("researcher_profile.scie_publication_count", "integer"),
    ("researcher_profile.intellectual_property_count", "integer"),
    ("researcher_profile.research_project_count", "integer"),
    ("publications[].journal_index_type", "keyword"),
    ("publications[].publication_year_month", "datetime"),
    ("intellectual_properties[].application_registration_type", "keyword"),
    ("intellectual_properties[].application_country", "keyword"),
    ("intellectual_properties[].application_date", "datetime"),
    ("intellectual_properties[].registration_date", "datetime"),
    ("research_projects[].project_start_date", "datetime"),
    ("research_projects[].project_end_date", "datetime"),
    ("research_projects[].reference_year", "integer"),
    ("research_projects[].performing_organization", "keyword"),
    ("research_projects[].managing_agency", "keyword"),
)


# =========================================================================
# v2.0 chunk 모델 스키마 (WO-C C2) — additive
# =========================================================================
# 기존 BRANCHES / DENSE_VECTOR_BY_BRANCH / SPARSE_VECTOR_BY_BRANCH / PAYLOAD_INDEX_FIELDS(nested)는
# 소비 코드(retriever/live_validator/seed_data)가 단일 벡터·chunk payload로 전환되는 C3/C8 슬라이스에서
# 제거한다. 그 전까지는 아래 v2.0 상수를 추가만 해 두고 v1.x 동작을 보존한다.

# 단일 named vector. doc_type별 named vector를 두지 않는다 — doc_type은 payload 필터. (DATA_MODEL §2)
DENSE_VECTOR_NAME = "dense_e5i"
SPARSE_VECTOR_NAME = "sparse_splade"

# WO-0 상수 재노출(검색 코드가 schema_registry 한 곳에서 doc_type/family를 참조하도록).
DOC_TYPES_V2: tuple[str, ...] = DOC_TYPES
DOC_TYPE_TO_FAMILY_V2: dict[str, str] = dict(DOC_TYPE_TO_FAMILY)
FAMILIES: tuple[str, ...] = tuple(f.value for f in Family)

# v2.0 3층 payload에서 필터(FieldCondition) 가능한 키 (DATA_MODEL §5).
FILTERABLE_FIELDS_V2: frozenset[str] = frozenset(
    {
        "researcher_id",
        "doc_type",
        "tags",
        "event_year",
        "event_date",
        "researcher_meta.affiliated_organization",
        "researcher_meta.highest_degree",
        "researcher_meta.publication_count",
        "researcher_meta.scie_publication_count",
        "researcher_meta.intellectual_property_count",
        "researcher_meta.research_project_count",
        "researcher_meta.researcher_assessor_count",
        "researcher_meta.expert_assessor_count",
        "domain_attrs.journal_class",
        "domain_attrs.ip_type",
        "domain_attrs.application_country",
        "domain_attrs.performing_organization",
        "domain_attrs.managing_agency",
        "domain_attrs.appointing_organization",
        "domain_attrs.evaluation_agency_name",
        "domain_attrs.tech_classification_system",
        "domain_attrs.specific_specialty_name",
        "domain_attrs.tech_rank",
        "domain_attrs.specialty_count",
    }
)

# v2.0 Qdrant payload 인덱스 세트 (DATA_MODEL §5).
# ★WO-B 부트스트랩이 실제 생성하는 인덱스와 1:1 정합해야 한다(C8 readiness 기준).
PAYLOAD_INDEX_FIELDS_V2: tuple[tuple[str, str], ...] = (
    ("researcher_id", "keyword"),
    ("doc_type", "keyword"),
    ("tags", "keyword"),
    ("researcher_meta.affiliated_organization", "keyword"),
    ("researcher_meta.highest_degree", "keyword"),
    ("domain_attrs.journal_class", "keyword"),
    ("domain_attrs.ip_type", "keyword"),
    ("domain_attrs.application_country", "keyword"),
    ("domain_attrs.performing_organization", "keyword"),
    ("domain_attrs.managing_agency", "keyword"),
    ("domain_attrs.appointing_organization", "keyword"),
    ("domain_attrs.evaluation_agency_name", "keyword"),
    ("domain_attrs.tech_classification_system", "keyword"),
    ("domain_attrs.specific_specialty_name", "keyword"),
    ("event_year", "integer"),
    ("researcher_meta.publication_count", "integer"),
    ("researcher_meta.scie_publication_count", "integer"),
    ("researcher_meta.intellectual_property_count", "integer"),
    ("researcher_meta.research_project_count", "integer"),
    ("researcher_meta.researcher_assessor_count", "integer"),
    ("researcher_meta.expert_assessor_count", "integer"),
    ("domain_attrs.tech_rank", "integer"),
    ("domain_attrs.specialty_count", "integer"),
    ("event_date", "datetime"),
)


@dataclass(frozen=True, slots=True)
class SearchSchemaRegistry:
    """스키마 정보를 런타임에 제공하는 컨테이너 클래스입니다."""
    dense_vector_by_branch: dict[str, str]
    sparse_vector_by_branch: dict[str, str]

    @classmethod
    def default(cls) -> "SearchSchemaRegistry":
        """기본 설정으로 레지스트리를 생성합니다."""
        return cls(
            dense_vector_by_branch=DENSE_VECTOR_BY_BRANCH,
            sparse_vector_by_branch=SPARSE_VECTOR_BY_BRANCH,
        )

