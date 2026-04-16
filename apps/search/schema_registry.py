"""
Qdrant 컬렉션의 스키마와 벡터 매핑 정보를 통합 관리하는 레지스트리 모듈입니다.
각 데이터 브랜치(기본, 논문, 특허, 과제)에 해당하는 벡터 이름과
필터링 가능한 페이로드 필드 정의를 포함합니다.
"""

from __future__ import annotations

from dataclasses import dataclass


# 시스템에서 사용하는 주요 데이터 브랜치 정의
BRANCHES: tuple[str, str, str, str] = ("basic", "art", "pat", "pjt")

# 각 브랜치별 Dense(의미론적) 벡터 이름 매핑
DENSE_VECTOR_BY_BRANCH = {
    "basic": "basic_vector_e5i",
    "art": "art_vector_e5i",
    "pat": "pat_vector_e5i",
    "pjt": "pjt_vector_e5i",
}

# 각 브랜치별 Sparse(키워드) 벡터 이름 매핑
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

# Qdrant 인덱스 생성을 위한 페이로드 필드와 데이터 타입 정의
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

