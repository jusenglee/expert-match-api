"""
전문가 추천 서비스의 핵심 비즈니스 로직을 오케스트레이션하는 모듈입니다.
질의 분석(Planner), 후보자 검색(Retriever), 다차원 평가(Judge) 과정을 통합하여
최종적인 전문가 추천 결과를 생성합니다.
"""

from __future__ import annotations

import logging
from typing import Any

from apps.core.feedback_store import FeedbackStore
from apps.domain.models import CandidateCard, JudgeOutput, PlannerOutput
from apps.recommendation.cards import CandidateCardBuilder
from apps.recommendation.judge import Judge
from apps.recommendation.planner import Planner
from apps.search.filters import QdrantFilterCompiler
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import BRANCHES

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    검색 및 추천 파이프라인을 실행하는 서비스 클래스입니다.
    외부 엔드포인트(FastAPI)와 내부 도메인 로직 사이의 가교 역할을 합니다.
    """
    def __init__(
        self,
        *,
        planner: Planner,
        retriever: QdrantHybridRetriever,
        filter_compiler: QdrantFilterCompiler,
        card_builder: CandidateCardBuilder,
        judge: Judge,
        feedback_store: FeedbackStore,
        shortlist_limit: int,
    ) -> None:
        """
        추천 서비스에 필요한 컴포넌트들을 주입받아 초기화합니다.
        
        Args:
            planner: 자연어 질의를 분석하여 검색 계획을 수립하는 플래너
            retriever: Qdrant 기반 하이브리드 검색기
            filter_compiler: 도메인 필터를 저장소 쿼리로 변환하는 컴파일러
            card_builder: 전문가 데이터를 구조화된 카드 형태로 변환하는 빌더
            judge: 후보군을 상세 평가하고 순위를 결정하는 심사역
            feedback_store: 사용자 피드백 저장소
            shortlist_limit: 최종 심시 단계로 넘어갈 후보자의 최대 수
        """
        self.planner = planner
        self.retriever = retriever
        self.filter_compiler = filter_compiler
        self.card_builder = card_builder
        self.judge = judge
        self.feedback_store = feedback_store
        self.shortlist_limit = shortlist_limit

    async def search_candidates(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        질의에 부합하는 전문가 후보군을 검색합니다. (최종 심사 단계 제외)
        """
        # 1. 자연어 질의 분석 및 필터 추출
        plan = await self.planner.plan(
            query=query,
            filters_override=filters_override,
            exclude_orgs=exclude_orgs,
            top_k=top_k,
        )
        
        # 2. 검색 필터 컴파일 (Qdrant 조건문 생성)
        query_filter = self.filter_compiler.compile(plan.hard_filters, plan.exclude_orgs)
        
        # 3. 하이브리드(Dense + Sparse) 검색 실행
        retrieval = await self.retriever.search(query=query, plan=plan, query_filter=query_filter)
        
        # 4. 검색된 히트(Hits)를 후보자 카드 객체로 변환
        cards = self.card_builder.build_small_cards(retrieval.hits, plan.hard_filters)
        
        return {
            "planner": plan,
            "query_filter": query_filter,
            "retrieved_count": len(retrieval.hits),
            "candidates": cards,
            "query_payload": retrieval.query_payload,
            "branch_queries": retrieval.branch_queries,
        }

    async def recommend(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        질의 분석부터 최종 심사까지 포함된 전체 추천 프로세스를 수행합니다.
        """
        # 1. 1차 후보군 검색
        search_result = await self.search_candidates(
            query=query,
            filters_override=filters_override,
            exclude_orgs=exclude_orgs,
            top_k=top_k,
        )
        
        plan: PlannerOutput = search_result["planner"]
        candidate_cards: list[CandidateCard] = search_result["candidates"]
        
        # 2. 심사 단계로 보낼 숏리스트(Shortlist) 구성
        shortlist = self.card_builder.shortlist(candidate_cards, self.shortlist_limit)
        
        # 3. LLM/Heuristic 기반 최종 심사 및 근거 생성
        judge_output: JudgeOutput = await self.judge.judge(query=query, plan=plan, shortlist=shortlist)

        # 4. 결과 정제 및 근거가 부족한 항목 필터링
        evidence_less_count = sum(1 for item in judge_output.recommended if not item.evidence)
        recommendations = [item for item in judge_output.recommended if item.evidence]
        not_selected_reasons = self._merge_unique_strings(judge_output.not_selected_reasons)

        # 5. 추천 결과가 없는 경우 사유 보강
        if not judge_output.recommended:
            not_selected_reasons = self._merge_unique_strings(
                not_selected_reasons,
                ["조건을 만족하는 추천 후보를 찾지 못했습니다."],
            )
        elif evidence_less_count:
            not_selected_reasons = self._merge_unique_strings(
                not_selected_reasons,
                ["근거가 충분한 추천 결과를 생성하지 못했습니다."],
            )

        if not recommendations:
            logger.info(
                "Returning empty recommendation result: query=%r retrieved_count=%d shortlist_count=%d judge_recommended_count=%d evidence_less_filtered_count=%d",
                query,
                search_result["retrieved_count"],
                len(shortlist),
                len(judge_output.recommended),
                evidence_less_count,
            )

        return {
            "intent_summary": plan.intent_summary,
            "applied_filters": plan.hard_filters,
            "searched_branches": list(BRANCHES),
            "retrieved_count": search_result["retrieved_count"],
            "recommendations": recommendations,
            "data_gaps": judge_output.data_gaps,
            "not_selected_reasons": not_selected_reasons,
            "trace": {
                "planner": plan.model_dump(mode="json"),
                "branch_queries": search_result["branch_queries"],
                "exclude_orgs": plan.exclude_orgs,
                "candidate_ids": [card.expert_id for card in candidate_cards],
                "recommendation_evidence_summary": [
                    {
                        "expert_id": item.expert_id,
                        "evidence_titles": [evidence.title for evidence in item.evidence],
                    }
                    for item in recommendations
                ],
                "query_payload": self._serialize_query_payload(search_result["query_payload"]),
            },
        }

    def save_feedback(
        self,
        *,
        query: str,
        selected_expert_ids: list[str],
        rejected_expert_ids: list[str],
        notes: str | None,
        metadata: dict[str, Any],
    ) -> int:
        """
        추천 결과에 대한 사용자의 명시적 피드백을 기록합니다.
        """
        return self.feedback_store.save_feedback(
            query=query,
            selected_expert_ids=selected_expert_ids,
            rejected_expert_ids=rejected_expert_ids,
            notes=notes,
            metadata=metadata,
        )

    @staticmethod
    def _serialize_query_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """내부 쿼리 데이터를 JSON 직렬화가 가능한 형태로 변환합니다."""
        serialized = dict(payload)
        serialized["prefetch"] = [str(item) for item in payload.get("prefetch", [])]
        query_filter = payload.get("query_filter")
        serialized["query_filter"] = str(query_filter) if query_filter else None
        serialized["query"] = str(payload.get("query"))
        return serialized

    @staticmethod
    def _merge_unique_strings(*groups: list[str]) -> list[str]:
        """여러 문자열 리스트를 중복 없이 병합합니다."""
        merged: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                normalized = item.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                merged.append(normalized)
        return merged
