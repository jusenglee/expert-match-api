from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


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
            normalized.append(item.strip() if isinstance(item, str) else str(item))
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


def _backfill_branch_flags(
    data: Any, *, source_key: str = "branch_coverage", target_key: str
) -> Any:
    if not isinstance(data, dict):
        return data
    if target_key in data or source_key not in data:
        return data
    normalized = dict(data)
    normalized[target_key] = normalized[source_key]
    return normalized


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
    
    # Optional legacy fields
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


class SeedExpertRecord(BaseModel):
    # Deprecated: SeedEvidencePoint로 대체 예정
    point_id: str
    payload: ExpertPayload
    basic_text: str
    art_text: str
    pat_text: str
    pjt_text: str


class PlannerOutput(BaseModel):
    intent_summary: str
    hard_filters: dict[str, Any] = Field(default_factory=dict)
    include_orgs: list[str] = Field(
        default_factory=list,
        description="추천 대상을 특정 기관 소속으로 제한할 때 사용. "
                    "'X 소속 연구자 중', 'X에서 추천' 패턴 → must 조건. "
                    "exclude_orgs(제외)와 반대 개념.",
    )
    exclude_orgs: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)
    core_keywords: list[str] = Field(default_factory=list)
    branch_weights: dict[BranchName, float] = Field(default_factory=dict)
    branch_query_hints: dict[BranchName, str] = Field(
        default_factory=dict,
        description="Deprecated. Kept for backward compatibility and not used by retrieval.",
    )
    top_k: int = 15


class SearchHit(BaseModel):
    expert_id: str
    score: float
    payload: ExpertPayload
    branch: BranchName | None = None # 매칭된 실적의 종류
    data_presence_flags: dict[BranchName, bool] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _backfill_legacy_branch_coverage(cls, data: Any) -> Any:
        return _backfill_branch_flags(data, target_key="data_presence_flags")

    @computed_field
    @property
    def branch_coverage(self) -> dict[BranchName, bool]:
        return self.data_presence_flags


class GroupedSearchHit(BaseModel):
    expert_id: str
    group_score: float # 그룹 내 최고 점수 또는 집계 점수
    hits: list[SearchHit] # 해당 전문가의 매칭된 실적 리스트
    data_presence_flags: dict[BranchName, bool] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _backfill_legacy_branch_coverage(cls, data: Any) -> Any:
        return _backfill_branch_flags(data, target_key="data_presence_flags")

    @computed_field
    @property
    def branch_coverage(self) -> dict[BranchName, bool]:
        return self.data_presence_flags


class EvidenceItem(BaseModel):
    type: Literal["paper", "patent", "project", "profile"]
    title: str
    date: str | None = None
    detail: str | None = None

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_evidence_type(cls, v: Any) -> Any:
        """LLM이 플래너 브랜치 축약어(pjt, art, pat)를 그대로 출력하는 경우를 방어합니다.
        내부 브랜치명과 Pydantic 허용값 사이의 스키마 파편화로 발생하는 Validation 오류를
        자동으로 복구합니다 (브랜치명 → 정규 type 값 매핑).
        """
        _BRANCH_ALIAS: dict[str, str] = {
            "pjt": "project",
            "art": "paper",
            "pat": "patent",
        }
        if isinstance(v, str):
            return _BRANCH_ALIAS.get(v.strip().lower(), v)
        return v


class CandidateCard(BaseModel):
    expert_id: str
    name: str
    organization: str | None = None
    position: str | None = None
    degree: str | None = None
    major: str | None = None
    branch_presence_flags: dict[BranchName, bool] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    keyword_matched_counts: dict[str, int] = Field(default_factory=dict)
    top_papers: list[PublicationEvidence] = Field(default_factory=list)
    top_patents: list[IntellectualPropertyEvidence] = Field(default_factory=list)
    top_projects: list[ResearchProjectEvidence] = Field(default_factory=list)
    matched_filter_summary: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    shortlist_score: float = 0.0
    rank_score: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _backfill_legacy_branch_coverage(cls, data: Any) -> Any:
        return _backfill_branch_flags(data, target_key="branch_presence_flags")

    @computed_field
    @property
    def branch_coverage(self) -> dict[BranchName, bool]:
        return self.branch_presence_flags


class RecommendationDecision(BaseModel):
    rank: int
    expert_id: str
    name: str
    organization: str | None = None
    fit: Literal["높음", "중간", "보통"]
    reasons: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    rank_score: float = 0.0


class JudgeOutput(BaseModel):
    recommended: list[RecommendationDecision] = Field(default_factory=list)
    not_selected_reasons: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
