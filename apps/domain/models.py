from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator

BranchName = Literal["basic", "art", "pat", "pjt"]


def _is_blank_string(value: Any) -> bool:
    return isinstance(value, str) and not value.strip()


def _normalize_string_list(value: Any) -> Any:
    if value is None or _is_blank_string(value):
        return []
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, (list, tuple, set)):
        normalized: list[str] = []
        for item in value:
            if item is None or _is_blank_string(item):
                continue
            normalized_value = item.strip() if isinstance(item, str) else str(item).strip()
            if normalized_value:
                normalized.append(normalized_value)
        return normalized
    return value


def _normalize_nested_list(value: Any) -> Any:
    if value is None or _is_blank_string(value):
        return []
    if isinstance(value, tuple):
        return list(value)
    return value


def _normalize_int(value: Any) -> Any:
    if value is None or _is_blank_string(value):
        return 0
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return value
    return value


def _normalize_optional_int(value: Any) -> Any:
    if value is None or _is_blank_string(value):
        return None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return value
    return value


class BasicInfo(BaseModel):
    """
    전문가의 기본 인적사항과 소속 정보(부서, 직위 등)를 담는 데이터 모델입니다.
    """
    researcher_id: str
    researcher_name: str
    gender: str | None = None
    affiliated_organization: str | None = None
    affiliated_organization_exact: str | None = None
    department: str | None = None
    position_title: str | None = None


class ResearcherProfile(BaseModel):
    """
    전문가의 학위, 전공 및 주요 실적(논문, 특허, 과제)의 전체 통계 수치를 담는 데이터 모델입니다.
    """
    highest_degree: str | None = None
    major_field: str | None = None
    publication_count: int = 0
    scie_publication_count: int = 0
    intellectual_property_count: int = 0
    research_project_count: int = 0

    @field_validator(
        "publication_count",
        "scie_publication_count",
        "intellectual_property_count",
        "research_project_count",
        mode="before",
    )
    @classmethod
    def _normalize_counts(cls, value: Any) -> Any:
        return _normalize_int(value)


class PublicationEvidence(BaseModel):
    """
    전문가의 개별 논문 실적 정보를 나타내는 데이터 모델입니다.
    초록(abstract) 및 키워드 정보는 검색과 추천 사유 생성에 핵심적으로 사용됩니다.
    """
    journal_index_type: str | None = None
    publication_title: str
    journal_name: str | None = None
    publication_year_month: str | None = None
    abstract: str | None = None
    korean_keywords: list[str] = Field(default_factory=list)
    english_keywords: list[str] = Field(default_factory=list)

    @field_validator("korean_keywords", "english_keywords", mode="before")
    @classmethod
    def _normalize_keyword_lists(cls, value: Any) -> Any:
        return _normalize_string_list(value)


class IntellectualPropertyEvidence(BaseModel):
    """
    전문가의 특허(지식재산권) 실적 정보를 나타내는 데이터 모델입니다.
    """
    intellectual_property_type: str | None = None
    intellectual_property_title: str
    application_registration_type: str | None = None
    application_country: str | None = None
    application_number: str | None = None
    application_date: str | None = None
    registration_number: str | None = None
    registration_date: str | None = None


class ResearchProjectEvidence(BaseModel):
    """
    전문가의 연구 과제 수행 실적 정보를 나타내는 데이터 모델입니다.
    """
    project_start_date: str | None = None
    project_end_date: str | None = None
    reference_year: int | None = None
    project_title_korean: str | None = None
    project_title_english: str | None = None
    performing_organization: str | None = None
    managing_agency: str | None = None
    research_objective_summary: str | None = None
    research_content_summary: str | None = None

    @field_validator("reference_year", mode="before")
    @classmethod
    def _normalize_reference_year(cls, value: Any) -> Any:
        return _normalize_optional_int(value)

    @computed_field
    @property
    def display_title(self) -> str:
        return self.project_title_korean or self.project_title_english or "Untitled project"


class EvaluationActivity(BaseModel):
    appoint_org_nm: str | None = None
    committee_nm: str | None = None
    appoint_period: str | None = None
    appoint_dt: str | None = None


class ExpertPayload(BaseModel):
    """
    벡터 데이터베이스(Qdrant)에 저장되고 검색 결과로 반환되는 단일 전문가의 원본(Payload) 데이터 모델입니다.
    기본 정보와 프로필, 그리고 모든 주요 실적 리스트를 포괄합니다.
    """
    basic_info: BasicInfo
    researcher_profile: ResearcherProfile
    publications: list[PublicationEvidence] = Field(default_factory=list)
    intellectual_properties: list[IntellectualPropertyEvidence] = Field(default_factory=list)
    research_projects: list[ResearchProjectEvidence] = Field(default_factory=list)
    technical_classifications: list[str] = Field(default_factory=list)
    evaluation_activity_cnt: int = 0
    external_activity_cnt: int = 0
    evaluation_activities: list[EvaluationActivity] = Field(default_factory=list)

    @field_validator(
        "publications",
        "intellectual_properties",
        "research_projects",
        "evaluation_activities",
        mode="before",
    )
    @classmethod
    def _normalize_nested_lists(cls, value: Any) -> Any:
        return _normalize_nested_list(value)

    @field_validator("technical_classifications", mode="before")
    @classmethod
    def _normalize_string_lists(cls, value: Any) -> Any:
        return _normalize_string_list(value)

    @field_validator("evaluation_activity_cnt", "external_activity_cnt", mode="before")
    @classmethod
    def _normalize_root_counts(cls, value: Any) -> Any:
        return _normalize_int(value)

    def to_payload_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SeedEvidencePoint(BaseModel):
    point_id: str
    researcher_id: str
    branch: BranchName
    content_text: str
    payload: ExpertPayload


class TermGroup(BaseModel):
    name: str = ""
    mode: str = "at_least_one"
    terms: list[str] = Field(default_factory=list)

    @field_validator("terms", mode="before")
    @classmethod
    def _normalize_terms(cls, value: Any) -> Any:
        return _normalize_string_list(value)


class PlannerOutput(BaseModel):
    """
    LLM 기반 플래너가 사용자 검색 쿼리를 분석하여 추출한 검색 의도(Intent) 및 각종 키워드/필터 조건들을 담는 데이터 모델입니다.
    이후 검색(Retriever)과 추천 사유 생성(Reasoner) 단계의 핵심 가이드라인으로 사용됩니다.
    """
    intent_summary: str
    hard_filters: dict[str, Any] = Field(default_factory=dict)
    include_orgs: list[str] = Field(default_factory=list)
    exclude_orgs: list[str] = Field(default_factory=list)
    
    # New fields for ADR-001
    domain_term_groups: list[TermGroup] = Field(default_factory=list)
    soft_preference_groups: list[TermGroup] = Field(default_factory=list)
    noise_terms: list[str] = Field(default_factory=list)
    bounded_hyde_document: str = ""
    relaxation_policy: dict[str, Any] = Field(default_factory=dict)

    # Legacy fields
    task_terms: list[str] = Field(default_factory=list)
    core_keywords: list[str] = Field(default_factory=list)
    retrieval_core: list[str] = Field(default_factory=list)
    must_aspects: list[str] = Field(default_factory=list)
    evidence_aspects: list[str] = Field(default_factory=list)
    generic_terms: list[str] = Field(default_factory=list)
    role_terms: list[str] = Field(default_factory=list)
    action_terms: list[str] = Field(default_factory=list)
    bundle_ids: list[str] = Field(default_factory=list)
    intent_flags: dict[str, Any] = Field(default_factory=dict)
    semantic_query: str = ""
    top_k: int = 15

    @field_validator(
        "include_orgs",
        "exclude_orgs",
        "task_terms",
        "core_keywords",
        "retrieval_core",
        "must_aspects",
        "evidence_aspects",
        "generic_terms",
        "role_terms",
        "action_terms",
        "noise_terms",
        "bundle_ids",
        mode="before",
    )
    @classmethod
    def _normalize_string_lists(cls, value: Any) -> Any:
        return _normalize_string_list(value)


class SearchHit(BaseModel):
    """
    벡터 데이터베이스(Qdrant) 등에서 반환된 개별 검색 결과(Hit)를 나타내는 데이터 모델입니다.
    전문가 원본 데이터(payload)와 함께 검색 점수(score), 데이터 존재 여부 플래그를 포함합니다.
    """
    expert_id: str
    score: float
    payload: ExpertPayload
    branch: BranchName | None = None
    data_presence_flags: dict[BranchName, bool] = Field(default_factory=dict)
    
    # 아키텍처 Support Rule 추적 정보
    stable_support_count: int = 0
    expanded_support_count: int = 0
    support_branches: list[BranchName] = Field(default_factory=list)


class GroupedSearchHit(BaseModel):
    expert_id: str
    group_score: float
    hits: list[SearchHit]
    data_presence_flags: dict[BranchName, bool] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    type: Literal["paper", "patent", "project", "profile"]
    title: str
    date: str | None = None
    detail: str | None = None


class CandidateCard(BaseModel):
    """
    검색 시스템에서 반환된 전문가 데이터를 바탕으로 만들어진 후보자 정보 카드 모델입니다.
    추천 파이프라인 전반을 거치며 필터링 상태(risks, data_gaps)와 최종 점수(rank_score)가 채워집니다.
    """
    expert_id: str
    name: str
    organization: str | None = None
    position: str | None = None
    degree: str | None = None
    major: str | None = None
    branch_presence_flags: dict[BranchName, bool] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    technical_classifications: list[str] = Field(default_factory=list)
    evaluation_activity_cnt: int = 0
    evaluation_activities: list[EvaluationActivity] = Field(default_factory=list)
    top_papers: list[PublicationEvidence] = Field(default_factory=list)
    top_patents: list[IntellectualPropertyEvidence] = Field(default_factory=list)
    top_projects: list[ResearchProjectEvidence] = Field(default_factory=list)
    matched_filter_summary: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    shortlist_score: float = 0.0
    rank_score: float = 0.0


class RecommendationDecision(BaseModel):
    """
    최종적으로 사용자에게 반환되는 개별 전문가의 추천 결과(명세서) 데이터 모델입니다.
    최종 순위(rank), 적합도(fit), LLM이 작성한 추천 사유(recommendation_reason) 및 이를 뒷받침하는 증거(evidence) 목록을 포함합니다.
    """
    rank: int
    expert_id: str
    name: str
    organization: str | None = None
    fit: Literal["높음", "중간", "보통"]
    recommendation_reason: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    rank_score: float = 0.0

    @computed_field(return_type=list[str])
    @property
    def reasons(self) -> list[str]:
        normalized_reason = " ".join(self.recommendation_reason.split())
        return [normalized_reason] if normalized_reason else []
