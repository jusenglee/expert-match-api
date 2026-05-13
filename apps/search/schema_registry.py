"""
Qdrant 컬렉션의 스키마와 벡터 매핑 정보를 통합 관리하는 레지스트리 모듈입니다.

[Architecture Overview]
본 시스템은 한 명의 전문가(Researcher) 데이터를 Qdrant 내에 4개의 서로 다른 관점(Branch)으로 분리하여 저장합니다.
각 브랜치는 동일한 Payload(전문가 메타데이터)를 공유하지만, 벡터(Dense/Sparse) 공간은 별도로 구성되어 있어 
"논문 기반 검색", "특허 기반 검색" 등 도메인에 최적화된 독립적인 점수 산출이 가능합니다.
"""

from __future__ import annotations

from dataclasses import dataclass


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

