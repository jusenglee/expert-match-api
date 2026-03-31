from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from apps.domain.models import RecommendationDecision


class RecommendationRequest(BaseModel):
    query: str = Field(..., description="평가위원 추천 자연어 질의")
    top_k: int | None = Field(default=None, ge=1, le=5)
    filters_override: dict[str, Any] = Field(default_factory=dict)
    exclude_orgs: list[str] = Field(default_factory=list)


class SearchCandidatesRequest(RecommendationRequest):
    pass


class FeedbackRequest(BaseModel):
    query: str
    selected_expert_ids: list[str] = Field(default_factory=list)
    rejected_expert_ids: list[str] = Field(default_factory=list)
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    intent_summary: str
    applied_filters: dict[str, Any]
    searched_branches: list[str]
    retrieved_count: int
    recommendations: list[RecommendationDecision]
    data_gaps: list[str]
    not_selected_reasons: list[str] = Field(default_factory=list)
    trace: dict[str, Any]


class SearchCandidateItem(BaseModel):
    expert_id: str
    name: str
    organization: str | None = None
    branch_coverage: dict[str, bool]
    counts: dict[str, int]
    data_gaps: list[str]
    risks: list[str]
    shortlist_score: float


class SearchCandidatesResponse(BaseModel):
    intent_summary: str
    applied_filters: dict[str, Any]
    searched_branches: list[str]
    retrieved_count: int
    candidates: list[SearchCandidateItem]
    trace: dict[str, Any]


class FeedbackResponse(BaseModel):
    feedback_id: int
    stored: bool = True


class ReadinessResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]
    issues: list[str]
    collection_name: str
    sample_point_id: str | None = None
