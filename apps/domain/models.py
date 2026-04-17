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
    researcher_id: str
    researcher_name: str
    gender: str | None = None
    affiliated_organization: str | None = None
    affiliated_organization_exact: str | None = None
    department: str | None = None
    position_title: str | None = None


class ResearcherProfile(BaseModel):
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
    intellectual_property_type: str | None = None
    intellectual_property_title: str
    application_registration_type: str | None = None
    application_country: str | None = None
    application_number: str | None = None
    application_date: str | None = None
    registration_number: str | None = None
    registration_date: str | None = None


class ResearchProjectEvidence(BaseModel):
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


class PlannerOutput(BaseModel):
    intent_summary: str
    hard_filters: dict[str, Any] = Field(default_factory=dict)
    include_orgs: list[str] = Field(default_factory=list)
    exclude_orgs: list[str] = Field(default_factory=list)
    task_terms: list[str] = Field(default_factory=list)
    core_keywords: list[str] = Field(default_factory=list)
    retrieval_core: list[str] = Field(default_factory=list)
    must_aspects: list[str] = Field(default_factory=list)
    # evidence_aspects: 한국어/영어 혼합, evidence text(논문 제목·초록·키워드, 과제명 등)와의
    # 실제 매칭에 사용된다. retrieval_core(한국어 BM25용)와 역할이 분리됨.
    # 비어있으면 evidence_selector가 must_aspects로 폴백한다.
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
        "bundle_ids",
        mode="before",
    )
    @classmethod
    def _normalize_string_lists(cls, value: Any) -> Any:
        return _normalize_string_list(value)


class SearchHit(BaseModel):
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
