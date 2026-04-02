from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


BranchName = Literal["basic", "art", "pat", "pjt"]


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


class PublicationEvidence(BaseModel):
    journal_index_type: str | None = None
    publication_title: str
    journal_name: str | None = None
    publication_year_month: str | None = None
    abstract: str | None = None
    korean_keywords: list[str] = Field(default_factory=list)
    english_keywords: list[str] = Field(default_factory=list)


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

    def to_payload_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SeedExpertRecord(BaseModel):
    point_id: str
    payload: ExpertPayload
    basic_text: str
    art_text: str
    pat_text: str
    pjt_text: str


class PlannerOutput(BaseModel):
    intent_summary: str
    hard_filters: dict[str, Any] = Field(default_factory=dict)
    exclude_orgs: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)
    branch_query_hints: dict[BranchName, str] = Field(default_factory=dict)
    top_k: int = 5


class SearchHit(BaseModel):
    expert_id: str
    score: float
    payload: ExpertPayload
    branch_coverage: dict[BranchName, bool] = Field(default_factory=dict)


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
    branch_coverage: dict[BranchName, bool] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    top_papers: list[PublicationEvidence] = Field(default_factory=list)
    top_patents: list[IntellectualPropertyEvidence] = Field(default_factory=list)
    top_projects: list[ResearchProjectEvidence] = Field(default_factory=list)
    matched_filter_summary: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    shortlist_score: float = 0.0


class RecommendationDecision(BaseModel):
    rank: int
    expert_id: str
    name: str
    fit: Literal["높음", "중간", "보통"]
    reasons: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class JudgeOutput(BaseModel):
    recommended: list[RecommendationDecision] = Field(default_factory=list)
    not_selected_reasons: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)

