from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


BranchName = Literal["basic", "art", "pat", "pjt"]


class PaperEvidence(BaseModel):
    paper_nm: str
    jrnl_nm: str | None = None
    sci_slct_nm: str | None = None
    jrnl_pub_dt: str | None = None
    abstract_str: str | None = None
    kor_kywd: str | None = None
    eng_kywd: str | None = None


class PatentEvidence(BaseModel):
    ipr_invention_nm: str
    ipr_regist_type_nm: str | None = None
    ipr_regist_nat_nm: str | None = None
    aply_no: str | None = None
    aply_dt: str | None = None
    regist_no: str | None = None
    regist_dt: str | None = None


class ProjectEvidence(BaseModel):
    title1: str | None = None
    title2: str | None = None
    content1: str | None = None
    content2: str | None = None
    pjt_prfrm_org_nm: str | None = None
    rsch_mgnt_org_nm: str | None = None
    start_dt: str | None = None
    end_dt: str | None = None
    stan_yr: int | None = None

    @computed_field
    @property
    def display_title(self) -> str:
        return self.title1 or self.title2 or "Untitled project"


class EvaluationActivity(BaseModel):
    appoint_org_nm: str | None = None
    committee_nm: str | None = None
    appoint_period: str | None = None
    appoint_dt: str | None = None


class ExpertPayload(BaseModel):
    doc_id: str
    hm_nm: str
    gndr_slct_nm: str | None = None
    blng_org_nm: str | None = None
    blng_org_nm_exact: str | None = None
    blng_dept: str | None = None
    position_nm: str | None = None
    degree_slct_nm: str | None = None
    major_slct_nm: str | None = None
    country_nm: str | None = None
    career_years: int | None = None
    technical_classifications: list[str] = Field(default_factory=list)
    article_cnt: int = 0
    scie_cnt: int = 0
    patent_cnt: int = 0
    project_cnt: int = 0
    evaluation_activity_cnt: int = 0
    external_activity_cnt: int = 0
    art: list[PaperEvidence] = Field(default_factory=list)
    pat: list[PatentEvidence] = Field(default_factory=list)
    pjt: list[ProjectEvidence] = Field(default_factory=list)
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
    top_papers: list[PaperEvidence] = Field(default_factory=list)
    top_patents: list[PatentEvidence] = Field(default_factory=list)
    top_projects: list[ProjectEvidence] = Field(default_factory=list)
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

