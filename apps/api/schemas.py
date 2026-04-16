from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from apps.domain.models import RecommendationDecision


class RecommendationRequest(BaseModel):
    query: str = Field(..., description="Natural-language recommendation query")
    top_k: int | None = Field(
        default=None, ge=1, le=15, description="Maximum number of returned results"
    )
    filters_override: dict[str, Any] = Field(
        default_factory=dict, description="Explicit search filter overrides"
    )
    include_orgs: list[str] = Field(
        default_factory=list,
        description="Organizations to include during retrieval",
    )
    exclude_orgs: list[str] = Field(
        default_factory=list,
        description="Organizations to exclude during retrieval",
    )


class SearchCandidatesRequest(RecommendationRequest):
    pass


class FeedbackRequest(BaseModel):
    query: str = Field(..., description="Original user query")
    selected_expert_ids: list[str] = Field(
        default_factory=list, description="Accepted expert ids"
    )
    rejected_expert_ids: list[str] = Field(
        default_factory=list, description="Rejected expert ids"
    )
    notes: str | None = Field(None, description="Optional operator notes")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional feedback metadata"
    )


class RecommendationResponse(BaseModel):
    intent_summary: str = Field(..., description="Planner intent summary")
    applied_filters: dict[str, Any] = Field(
        ..., description="Filters applied to retrieval"
    )
    searched_branches: list[str] = Field(
        ..., description="Branches used during retrieval"
    )
    retrieved_count: int = Field(..., description="Total retrieved candidate count")
    recommendations: list[RecommendationDecision] = Field(
        ..., description="Final recommendation decisions"
    )
    data_gaps: list[str] = Field(
        ..., description="Missing or weak data observations"
    )
    not_selected_reasons: list[str] = Field(
        default_factory=list, description="Reasons for no final recommendations"
    )
    trace: dict[str, Any] = Field(..., description="Debug trace payload")


class SearchCandidateItem(BaseModel):
    expert_id: str = Field(..., description="Expert identifier")
    name: str = Field(..., description="Expert name")
    organization: str | None = Field(None, description="Affiliated organization")
    branch_presence_flags: dict[str, bool] = Field(
        ..., description="Per-branch data presence flags"
    )
    counts: dict[str, int] = Field(..., description="Evidence counts by category")
    data_gaps: list[str] = Field(..., description="Candidate-specific data gaps")
    risks: list[str] = Field(..., description="Candidate-specific risks")
    shortlist_score: float = Field(..., description="Search shortlist score")
    
    # Support Rule 추적 정보
    stable_hits: int = 0
    expanded_hits: int = 0
    support_branches: list[str] = Field(default_factory=list)


class SearchCandidatesResponse(BaseModel):
    intent_summary: str = Field(..., description="Planner intent summary")
    applied_filters: dict[str, Any] = Field(
        ..., description="Filters applied to retrieval"
    )
    searched_branches: list[str] = Field(
        ..., description="Branches used during retrieval"
    )
    retrieved_count: int = Field(..., description="Total retrieved candidate count")
    candidates: list[SearchCandidateItem] = Field(
        ..., description="Retrieved candidate cards"
    )
    trace: dict[str, Any] = Field(..., description="Debug trace payload")


class FeedbackResponse(BaseModel):
    feedback_id: int = Field(..., description="Stored feedback identifier")
    stored: bool = Field(True, description="Whether feedback was stored")


class ReadinessResponse(BaseModel):
    ready: bool = Field(..., description="Overall readiness state")
    checks: dict[str, bool] = Field(..., description="Readiness checks")
    issues: list[str] = Field(..., description="Readiness issues")
    collection_name: str = Field(..., description="Target Qdrant collection name")
    sample_point_id: str | None = Field(
        None, description="Optional validation sample point id"
    )
