"""
API 요청 및 응답에 사용되는 Pydantic 데이터 모델(Schema) 정의 모듈입니다.
추천 요청, 검색 요청, 피드백, 헬스체크 등 모든 통신 규격을 정의합니다.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from apps.domain.models import RecommendationDecision


class RecommendationRequest(BaseModel):
    """최종 전문가 추천을 요청할 때 사용하는 스키마입니다."""
    query: str = Field(..., description="평가위원 추천 자연어 질의 (예: '인공지능 분야 전문가 추천')")
    top_k: int | None = Field(default=None, ge=1, le=5, description="추출할 상위 결과 개수")
    filters_override: dict[str, Any] = Field(default_factory=dict, description="검색 필터를 수동으로 지정할 때 사용")
    exclude_orgs: list[str] = Field(default_factory=list, description="추천에서 제외할 기관 목록")


class SearchCandidatesRequest(RecommendationRequest):
    """전문가 후보군 검색을 요청할 때 사용하는 스키마입니다. (RecommendationRequest와 동일)"""
    pass


class FeedbackRequest(BaseModel):
    """추천 결과에 대한 사용자 피드백을 보낼 때 사용하는 스키마입니다."""
    query: str = Field(..., description="당시 사용된 검색 질의")
    selected_expert_ids: list[str] = Field(default_factory=list, description="사용자가 선택(긍정)한 전문가 ID 목록")
    rejected_expert_ids: list[str] = Field(default_factory=list, description="사용자가 거절(부정)한 전문가 ID 목록")
    notes: str | None = Field(None, description="기타 관리자 메모")
    metadata: dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")


class RecommendationResponse(BaseModel):
    """추천 결과 응답 스키마입니다."""
    intent_summary: str = Field(..., description="LLM이 분석한 사용자 질의 의도 요약")
    applied_filters: dict[str, Any] = Field(..., description="실제로 적용된 검색 필터")
    searched_branches: list[str] = Field(..., description="검색이 수행된 데이터 브랜치 목록")
    retrieved_count: int = Field(..., description="필터링 후 검색된 전체 후보 수")
    recommendations: list[RecommendationDecision] = Field(..., description="최종 추천된 전문가 상세 정보 및 결정 근거")
    data_gaps: list[str] = Field(..., description="데이터가 부족하거나 확인이 필요한 항목")
    not_selected_reasons: list[str] = Field(default_factory=list, description="추천에서 제외된 후보들에 대한 이유")
    trace: dict[str, Any] = Field(..., description="디버깅을 위한 내부 처리 과정 추적 데이터")


class SearchCandidateItem(BaseModel):
    """후보자 목록 조회 시 개별 전문가의 정보를 담는 항목입니다."""
    expert_id: str = Field(..., description="전문가 고유 ID")
    name: str = Field(..., description="성명")
    organization: str | None = Field(None, description="소속 기관")
    branch_coverage: dict[str, bool] = Field(..., description="각 데이터 소스(논문, 특허 등)별 데이터 존재 여부")
    counts: dict[str, int] = Field(..., description="각 실적별 건수 통계")
    data_gaps: list[str] = Field(..., description="데이터 결측 항목")
    risks: list[str] = Field(..., description="잠재적 리스크 항목")
    shortlist_score: float = Field(..., description="후보 선정을 위한 기초 점수")


class SearchCandidatesResponse(BaseModel):
    """후보자 목록 조회 결과 응답 스키마입니다."""
    intent_summary: str = Field(..., description="질의 의도 요약")
    applied_filters: dict[str, Any] = Field(..., description="적용된 필터")
    searched_branches: list[str] = Field(..., description="검색 브랜치")
    retrieved_count: int = Field(..., description="검색된 전체 수")
    candidates: list[SearchCandidateItem] = Field(..., description="검색된 후보자 목록")
    trace: dict[str, Any] = Field(..., description="추적 데이터")


class FeedbackResponse(BaseModel):
    """피드백 저장 결과 응답 스키마입니다."""
    feedback_id: int = Field(..., description="저장된 피드백의 고유 번호")
    stored: bool = Field(True, description="저장 성공 여부")


class ReadinessResponse(BaseModel):
    """서비스 상세 상태(Readiness) 응답 스키마입니다."""
    ready: bool = Field(..., description="서비스 준비 완료 여부")
    checks: dict[str, bool] = Field(..., description="각 모듈별 상태 체크 결과")
    issues: list[str] = Field(..., description="발생한 문제 요약")
    collection_name: str = Field(..., description="연결된 Qdrant 컬렉션 이름")
    sample_point_id: str | None = Field(None, description="샘플 데이터 확인 여부")
